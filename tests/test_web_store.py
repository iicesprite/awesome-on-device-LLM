import copy
from datetime import date
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import catalog
import note_render
import web_store
from web_store import Store, StoreError


class StoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="catalog-store-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for category in catalog.CATEGORIES:
            (self.root / category).mkdir()
        (self.root / "tags.yaml").write_text(
            "tags:\n"
            "  topic/quantization: {label: 量化, aliases: [quantization]}\n"
            "  hardware/mobile: {label: 手机, aliases: [mobile]}\n",
            encoding="utf-8",
        )
        self.store = Store(self.root)

    def metadata(self, resource_id="example", **updates):
        data = {
            "id": resource_id,
            "title": "量化研究",
            "type": "paper",
            "url": "https://example.org/" + resource_id,
            "status": "unread",
            "summary": "端侧研究",
            "tags": ["topic/quantization"],
        }
        data.update(updates)
        return data

    def add_entry(self, resource_id="example", category="papers", body=b"\n\n# Body\n", **updates):
        data = self.metadata(resource_id, type=catalog.CATEGORIES[category], **updates)
        folder = self.root / category / resource_id
        folder.mkdir()
        path = folder / "resource.md"
        path.write_bytes(b"---\n" + yaml.safe_dump(data, allow_unicode=True).encode("utf-8") + b"---" + body)
        return path

    def files(self):
        return {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def assert_error(self, status, function, *args, **kwargs):
        with self.assertRaises(StoreError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.status, status)
        return caught.exception

    def test_list_reuses_alias_search_and_intersects_filters(self):
        self.add_entry(tags=["topic/quantization", "hardware/mobile"], status="reading")
        self.add_entry("other", category="blogs")
        result = self.store.list_resources("研究", ["量化", "mobile"], "paper", "reading")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["entries"][0]["id"], "example")
        self.assertEqual(result["tags"]["topic/quantization"]["label"], "量化")
        self.assertEqual(self.store.list_resources(status="read")["total"], 0)
        self.assert_error(400, self.store.list_resources, tags=["missing"])

    def test_detail_contains_metadata_body_and_unchanged_raw_notes(self):
        path = self.add_entry(custom={"nested": [1, True]})
        notes = b"# Notes\r\n\r\nOriginal content\r\n"
        path.with_name("notes.md").write_bytes(notes)
        before = self.files()
        detail = self.store.detail("example")
        self.assertEqual(detail["revision"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(detail["entry"]["custom"], {"nested": [1, True]})
        self.assertEqual(detail["entry"]["metadata_path"], "papers/example/resource.md")
        self.assertEqual(detail["entry"]["notes_path"], "papers/example/notes.md")
        self.assertEqual(detail["notes_markdown"], notes.decode())
        self.assertIn("<h1>Body</h1>", detail["body_html"])
        self.assertEqual(before, self.files())

    def test_independent_patch_preserves_unknown_yaml_types_body_notes_and_assets(self):
        body = "\r\n\r\n# 正文\r\n```text\r\n---\r\n```\r\n尾部无换行".encode()
        data = self.metadata(
            custom={"date": date(2025, 2, 3), "binary": b"\x00\xff", "set": {"one", "two"}},
            attachments=[{"url": "https://example.org/a.pdf", "path": "resources/original.pdf"}],
            authors=["Original"],
        )
        folder = self.root / "papers" / "example"
        folder.mkdir()
        path = folder / "resource.md"
        original = b"\xef\xbb\xbf---\r\n" + yaml.safe_dump(data, allow_unicode=True).encode().replace(b"\n", b"\r\n") + b"---" + body
        path.write_bytes(original)
        notes = folder / "notes.md"
        notes.write_bytes(b"# Do not touch\r\n")
        (folder / "resources").mkdir()
        attachment = folder / "resources" / "original.pdf"
        attachment.write_bytes(b"%PDF-original")
        revision = self.store.detail("example")["revision"]
        result = self.store.update("example", revision, {"status": "reading", "summary": "已更新"})
        actual = catalog.read_frontmatter(path)
        expected = copy.deepcopy(data)
        expected.update(status="reading", summary="已更新")
        self.assertEqual(actual, expected)
        self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf---\r\n"))
        self.assertTrue(path.read_bytes().endswith(b"---" + body))
        self.assertEqual(notes.read_bytes(), b"# Do not touch\r\n")
        self.assertEqual(attachment.read_bytes(), b"%PDF-original")
        self.assertNotEqual(result["revision"], revision)
        self.assertIn("已更新", (self.root / "INDEX.md").read_text(encoding="utf-8"))
        self.assertTrue((self.root / "catalog.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertNotIn("metadata_path", actual)

    def test_remove_only_named_optional_field(self):
        path = self.add_entry(authors=["Keep"], year=2025, custom={"keep": True})
        revision = self.store.detail("example")["revision"]
        self.store.update("example", revision, {"title": "New title"}, ["year"])
        data = catalog.read_frontmatter(path)
        self.assertNotIn("year", data)
        self.assertEqual(data["authors"], ["Keep"])
        self.assertEqual(data["custom"], {"keep": True})

    def test_stale_revision_and_external_body_change_do_not_write(self):
        path = self.add_entry()
        revision = self.store.detail("example")["revision"]
        path.write_bytes(path.read_bytes() + b"external edit")
        before = self.files()
        self.assert_error(409, self.store.update, "example", revision, {"title": "Stale"})
        self.assertEqual(before, self.files())

    def test_invalid_patch_has_no_file_or_index_changes(self):
        self.add_entry()
        revision = self.store.detail("example")["revision"]
        cases = [
            ({"tags": ["unknown/tag"]}, ()),
            ({"year": True}, ()),
            ({"id": "example"}, ()),
            ({"type": "blog"}, ()),
            ({"metadata_path": "elsewhere"}, ()),
            ({"notes_path": "elsewhere"}, ()),
            ({"zotero": {}}, ()),
            ({}, ["title"]),
            ({}, ["type"]),
            ({"authors": ["A"]}, ["authors"]),
            ({"attachments": [{"url": "https://example.org/a", "path": "resources/../a"}]}, ()),
        ]
        before = self.files()
        for updates, remove_fields in cases:
            with self.subTest(updates=updates, remove_fields=remove_fields):
                self.assert_error(400, self.store.update, "example", revision, updates, remove_fields)
                self.assertEqual(before, self.files())

    def test_noop_patch_does_not_rewrite_original_yaml(self):
        self.add_entry()
        detail = self.store.detail("example")
        before = self.files()
        self.assertEqual(self.store.update("example", detail["revision"], {}), detail)
        self.assertEqual(before, self.files())

    def test_create_defaults_status_and_builds_indexes_without_notes_or_downloads(self):
        metadata = self.metadata()
        del metadata["status"]
        metadata["attachments"] = [{"url": "https://example.org/file.pdf", "path": "resources/file.pdf"}]
        detail = self.store.create(metadata)
        self.assertEqual(detail["entry"]["status"], "unread")
        self.assertIsNone(detail["notes_markdown"])
        self.assertIsNone(detail["notes_html"])
        self.assertEqual(detail["entry"]["notes_path"], "")
        folder = self.root / "papers" / "example"
        self.assertEqual({p.name for p in folder.iterdir()}, {"resource.md"})
        self.assertEqual(catalog.read_frontmatter(folder / "resource.md")["attachments"], metadata["attachments"])
        self.assertIn("量化研究", (self.root / "INDEX.md").read_text(encoding="utf-8"))
        self.assertTrue((self.root / "catalog.csv").is_file())
        self.assertNotIn("status", metadata)

    def test_create_invalid_data_initializes_no_directory(self):
        before = self.files()
        for updates in ({"tags": ["bad/tag"]}, {"id": "../escape"}, {"summary": None}, {"type": "invalid"}, {"notes_path": "notes.md"}):
            with self.subTest(updates=updates):
                self.assert_error(400, self.store.create, self.metadata(**updates))
                self.assertEqual(before, self.files())
                self.assertEqual(list((self.root / "papers").iterdir()), [])

    def test_duplicate_create_reuses_cross_category_finder_without_writes(self):
        self.add_entry("existing", category="blogs", doi="10.1234/example", title="Same work")
        before = self.files()
        error = self.assert_error(409, self.store.create, self.metadata("new", doi="https://doi.org/10.1234/EXAMPLE", title="Different title"))
        self.assertEqual(error.duplicates[0]["id"], "existing")
        self.assertEqual(before, self.files())
        self.assertFalse((self.root / "papers" / "new").exists())

    def test_existing_unindexed_directory_is_never_overwritten(self):
        folder = self.root / "papers" / "example"
        # An existing visible category folder without resource.md is an invalid catalog.
        folder.mkdir()
        (folder / "sentinel").write_bytes(b"Keep")
        before = self.files()
        self.assert_error(400, self.store.create, self.metadata())
        self.assertEqual(before, self.files())

    def test_create_index_failure_rolls_back_card_and_directory(self):
        (self.root / "INDEX.md").write_bytes(b"old index")
        (self.root / "catalog.csv").write_bytes(b"old csv")
        before = self.files()
        atomic = web_store._atomic_bytes
        failed = False

        def fail_once(path, content):
            nonlocal failed
            if path.name == "catalog.csv" and not failed:
                failed = True
                raise OSError("simulated disk failure")
            return atomic(path, content)

        with patch.object(web_store, "_atomic_bytes", side_effect=fail_once):
            self.assert_error(400, self.store.create, self.metadata())
        self.assertEqual(before, self.files())
        self.assertFalse((self.root / "papers" / "example").exists())

    def test_update_index_failure_restores_metadata_and_absent_indexes(self):
        self.add_entry()
        revision = self.store.detail("example")["revision"]
        before = self.files()
        atomic = web_store._atomic_bytes
        failed = False

        def fail_once(path, content):
            nonlocal failed
            if path.name == "catalog.csv" and not failed:
                failed = True
                raise OSError("simulated disk failure")
            return atomic(path, content)

        with patch.object(web_store, "_atomic_bytes", side_effect=fail_once):
            self.assert_error(400, self.store.update, "example", revision, {"title": "Changed"})
        self.assertEqual(before, self.files())

    def test_index_render_failure_does_not_initialize_directory(self):
        before = self.files()
        with patch.object(catalog, "render_csv", side_effect=ValueError("unrenderable")):
            self.assert_error(400, self.store.create, self.metadata())
        self.assertEqual(before, self.files())
        self.assertFalse((self.root / "papers" / "example").exists())

    def test_asset_limits_access_to_existing_resource_files(self):
        path = self.add_entry()
        resources = path.parent / "resources"
        resources.mkdir()
        asset = resources / "图.png"
        asset.write_bytes(b"image")
        self.assertEqual(self.store.asset("example", "resources/图.png"), asset)
        for relative in ("../tags.yaml", "resources/../resource.md", "resources//图.png", "resource.md", "/tags.yaml", "resources/../../tags.yaml", "resources\\图.png", "resources/.git/config", "resources/C:/file"):
            with self.subTest(relative=relative):
                self.assert_error(403, self.store.asset, "example", relative)
        self.assert_error(404, self.store.asset, "example", "resources/missing.pdf")
        self.assert_error(404, self.store.detail, "missing")
        self.assert_error(400, self.store.detail, "../example")

    def symlink(self, target, link, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Symlinks unavailable: {exc}")

    def test_asset_rejects_symlink_even_when_destination_is_inside_repository(self):
        path = self.add_entry()
        resources = path.parent / "resources"
        resources.mkdir()
        self.symlink(self.root / "tags.yaml", resources / "linked.yaml")
        self.assert_error(403, self.store.asset, "example", "resources/linked.yaml")

    @unittest.skipUnless(sys.platform == "win32", "Windows directory junction")
    def test_windows_junction_cannot_expose_private_repository_files(self):
        path = self.add_entry()
        private = self.root / ".local"
        private.mkdir()
        (private / "secret.json").write_text('{"private":true}', encoding="utf-8")
        junction = path.parent / "resources"
        subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(junction), str(private)],
            check=True, capture_output=True,
        )
        self.addCleanup(junction.rmdir)
        self.assert_error(403, self.store.asset, "example", "resources/secret.json")

    def test_symlinked_resource_file_is_not_read_or_changed(self):
        path = self.add_entry()
        target = self.root / "original.md"
        path.rename(target)
        self.symlink(target, path)
        before = target.read_bytes()
        self.assert_error(403, self.store.detail, "example")
        self.assert_error(403, self.store.update, "example", "any", {"title": "Changed"})
        self.assertEqual(before, target.read_bytes())

    def test_category_symlink_is_rejected_before_create_directory(self):
        # A category symlink is rejected before catalog.load_catalog can follow it.
        directory = self.root / "papers"
        directory.rmdir()
        destination = self.root / "actual-papers"
        destination.mkdir()
        self.symlink(destination, directory, directory=True)
        self.assert_error(403, self.store.create, self.metadata())
        self.assertEqual(list(destination.iterdir()), [])

    def test_markdown_renders_tables_fences_and_safe_relative_assets(self):
        path = self.add_entry()
        markdown = (
            "| A | B |\n| --- | --- |\n| one | two |\n\n"
            "![图](resources/图.png) [PDF](resources/paper.pdf) [笔记](notes.md)\n\n"
            "[Outside](https://example.org/path)\n\n"
            "```mermaid\ngraph TD; A-->B\n```\n"
        )
        path.with_name("notes.md").write_text(markdown, encoding="utf-8")
        rendered = self.store.detail("example")["notes_html"]
        self.assertIn("<table>", rendered)
        self.assertIn('/files/example/resources/%E5%9B%BE.png', rendered)
        self.assertIn('/files/example/resources/paper.pdf', rendered)
        self.assertIn('#resource=example&amp;tab=notes', rendered)
        self.assertIn('rel="noopener noreferrer"', rendered)
        self.assertIn('class="language-mermaid"', rendered)
        self.assertIn("graph TD; A--&gt;B", rendered)

    def test_markdown_blocks_raw_html_unsafe_urls_svg_images_and_traversal(self):
        markdown = (
            '<script>alert(1)</script>\n\n<img src=x onerror="alert(1)">\n\n'
            '[bad](javascript:alert%281%29) ![svg](resources/active.svg)\n\n'
            '[escape](resources/%2e%2e/resource.md) ![remote](https://example.org/a.png)\n\n'
            '[data](data:text/html;base64,PHNjcmlwdD4=)\n\n'
            '[network](//evil.example/a) ![event](resources/a.png%22onerror=%22bad)\n'
        )
        rendered = note_render.render_markdown(markdown, "example")
        self.assertNotIn("<script", rendered)
        self.assertNotIn("<img src=x", rendered)
        self.assertNotIn('href="javascript:', rendered)
        self.assertNotIn('href="data:', rendered)
        self.assertNotIn('href="//evil', rendered)
        self.assertNotIn('src="https://', rendered)
        self.assertNotIn('src="/files/example/resources/active.svg', rendered)
        self.assertNotIn('href="/files/example/resources/..', rendered)
        self.assertNotIn('onerror="bad"', rendered)


    def test_sidecars_and_note_frontmatter_are_not_metadata_sources(self):
        path = self.add_entry()
        sidecar = path.with_name("resource.yaml")
        sidecar.write_text("title: Not authoritative\nstatus: read\n", encoding="utf-8")
        notes = path.with_name("notes.md")
        notes.write_bytes(b"---\ntitle: Note title\n---\n# Read only\n")
        original_notes = notes.read_bytes()
        original_sidecar = sidecar.read_bytes()
        detail = self.store.detail("example")
        self.assertEqual(detail["entry"]["title"], "量化研究")
        self.assertEqual(detail["entry"]["status"], "unread")
        self.store.update("example", detail["revision"], {"status": "reading"})
        self.assertEqual(notes.read_bytes(), original_notes)
        self.assertEqual(sidecar.read_bytes(), original_sidecar)
        self.assertEqual(catalog.read_frontmatter(path)["status"], "reading")

    def test_external_edit_during_index_render_is_a_conflict_without_index_writes(self):
        path = self.add_entry()
        detail = self.store.detail("example")
        original = path.read_bytes()
        render = catalog.render_index

        def external_edit(entries, vocabulary):
            path.write_bytes(original + b"\nExternal edit\n")
            return render(entries, vocabulary)

        with patch.object(catalog, "render_index", side_effect=external_edit):
            self.assert_error(409, self.store.update, "example", detail["revision"], {"title": "Changed"})
        self.assertEqual(path.read_bytes(), original + b"\nExternal edit\n")
        self.assertFalse((self.root / "INDEX.md").exists())
        self.assertFalse((self.root / "catalog.csv").exists())

    def test_symlinked_index_prevents_creation_before_directory_initialization(self):
        target = self.root / "untouched-index.md"
        target.write_bytes(b"Do not overwrite")
        self.symlink(target, self.root / "INDEX.md")
        self.assert_error(403, self.store.create, self.metadata())
        self.assertEqual(target.read_bytes(), b"Do not overwrite")
        self.assertFalse((self.root / "papers" / "example").exists())

    def test_patch_attachment_symlink_is_rejected_without_writes(self):
        path = self.add_entry()
        resources = path.parent / "resources"
        resources.mkdir()
        self.symlink(self.root / "tags.yaml", resources / "linked.yaml")
        revision = self.store.detail("example")["revision"]
        before = self.files()
        self.assert_error(
            403, self.store.update, "example", revision,
            {"attachments": [{"url": "https://example.org/a", "path": "resources/linked.yaml"}]},
        )
        self.assertEqual(before, self.files())

    def test_patch_keeps_unknown_ordered_mapping_and_pairs_yaml_types(self):
        path = self.add_entry()
        extra = (
            b"custom_ordered: !!omap\n  - first: 1\n  - second: 2\n"
            b"custom_pairs: !!pairs\n  - repeated: 1\n  - repeated: 2\n"
            b"custom_anchor: &shared [one, two]\ncustom_alias: *shared\n"
        )
        path.write_bytes(b"---\n" + yaml.safe_dump(self.metadata()).encode() + extra + b"---\n# Body\n")
        original = catalog.read_frontmatter(path)
        detail = self.store.detail("example")
        self.store.update("example", detail["revision"], {"status": "reading", "venue": "New venue"})
        updated = catalog.read_frontmatter(path)
        for key in ("custom_ordered", "custom_pairs", "custom_anchor", "custom_alias"):
            self.assertEqual(updated[key], original[key])
        self.assertIsInstance(updated["custom_ordered"][0], tuple)
        self.assertIsInstance(updated["custom_pairs"][0], tuple)
        self.assertIs(updated["custom_anchor"], updated["custom_alias"])
        self.assertEqual(updated["venue"], "New venue")


if __name__ == "__main__":
    unittest.main()
