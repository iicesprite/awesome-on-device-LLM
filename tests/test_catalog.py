import contextlib
import csv
from datetime import date
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "catalog.py"
SPEC = importlib.util.spec_from_file_location("catalog", SCRIPT)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llm-catalog-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for category in catalog.CATEGORIES:
            (self.root / category).mkdir()
        (self.root / "tags.yaml").write_text(
            "tags:\n"
            "  topic/quantization: {label: 量化, aliases: [quantization]}\n"
            "  hardware/jetson-orin: {label: Jetson Orin, aliases: [orin]}\n",
            encoding="utf-8",
        )

    def add_entry(self, category="papers", entry_id="example", **updates):
        data = {
            "id": entry_id,
            "title": "量化研究 | A [B]",
            "type": catalog.CATEGORIES[category],
            "url": "https://example.org/paper(a)",
            "status": "unread",
            "summary": "跨设备测试\n第二行",
            "tags": ["topic/quantization", "hardware/jetson-orin"],
            "links": {"code": "https://github.com/example/project"},
            "attachments": [{"url": "https://example.org/paper.pdf", "path": "resources/paper.pdf"}],
        }
        data.update(updates)
        folder = self.root / category / entry_id
        folder.mkdir(exist_ok=True)
        path = folder / "meta.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return path

    def load(self):
        return catalog.load_catalog(self.root)

    def run_command(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = catalog.main(["--root", str(self.root), *args])
        return result, stdout.getvalue(), stderr.getvalue()

    def test_empty_catalog(self):
        self.assertEqual(self.run_command("build")[0], 0)
        self.assertIn("尚未收录", (self.root / "INDEX.md").read_text(encoding="utf-8"))
        self.assertEqual(len(list(csv.DictReader((self.root / "catalog.csv").read_text(encoding="utf-8-sig").splitlines()))), 0)

    def test_build_all_categories_and_optional_notes(self):
        for category in catalog.CATEGORIES:
            path = self.add_entry(category, category + "-entry")
            if category == "papers":
                path.with_name("notes.md").write_text("# 我的笔记", encoding="utf-8")
        result, _, error = self.run_command("build")
        self.assertEqual((result, error), (0, ""))
        index = (self.root / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("papers（1）", index)
        self.assertIn("blogs（1）", index)
        self.assertIn("docs（1）", index)
        self.assertIn("papers/papers-entry/notes.md", index)
        self.assertNotIn("blogs/blogs-entry/notes.md", index)
        self.assertIn("### topic/quantization", index)
        raw_csv = (self.root / "catalog.csv").read_bytes()
        self.assertTrue(raw_csv.startswith(b"\xef\xbb\xbf"))
        rows = list(csv.DictReader(io.StringIO(raw_csv.decode("utf-8-sig"))))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["title"], "量化研究 | A [B]")
        self.assertIn("resources/paper.pdf", rows[0]["attachments"])
        self.assertEqual(
            list(rows[0]),
            ["id", "title", "type", "url", "status", "summary", "tags", "authors",
             "year", "venue", "doi", "links", "attachments", "metadata_path", "notes_path"],
        )
        self.assertNotIn("Zotero", index)
        self.assertNotIn("zotero://", index)
        self.assertNotIn("zotero://", raw_csv.decode("utf-8-sig"))

    def test_attachments_do_not_need_to_exist(self):
        self.add_entry()
        self.assertEqual(len(self.load()[0]), 1)

    def test_markdown_escapes_table_and_link_syntax(self):
        self.add_entry()
        entries, vocabulary, _ = self.load()
        rendered = catalog.render_index(entries, vocabulary)
        self.assertIn("&#124; A &#91;B&#93;", rendered)
        self.assertIn("paper%28a%29", rendered)
        self.assertIn("跨设备测试 第二行", rendered)

    def test_search_uses_aliases_and_intersection(self):
        self.add_entry(status="read")
        self.add_entry("blogs", "other", tags=["topic/quantization"])
        entries, _, aliases = self.load()
        results = catalog.search(entries, aliases, ["量化", "ORIN"], kind="paper", status="read", query="研究")
        self.assertEqual([e["id"] for e in results], ["example"])
        self.assertEqual(catalog.search(entries, aliases, status="reading"), [])
        self.assertEqual(self.run_command("search", "--tag", "量化", "--tag", "orin")[0], 0)

    def test_unknown_search_tag_fails(self):
        self.assertEqual(self.run_command("search", "--tag", "typo")[0], 1)

    def test_yaml_date_extension_can_be_searched(self):
        self.add_entry(added=date(2026, 10, 29))
        entries, _, aliases = self.load()
        self.assertEqual(len(catalog.search(entries, aliases, query="2026-10-29")), 1)

    def test_check_does_not_write_indexes(self):
        self.add_entry()
        self.assertEqual(self.run_command("check")[0], 0)
        self.assertFalse((self.root / "INDEX.md").exists())
        self.assertFalse((self.root / "catalog.csv").exists())

    def test_build_is_deterministic(self):
        self.add_entry()
        self.run_command("build")
        before = [(self.root / name).read_bytes() for name in ("INDEX.md", "catalog.csv")]
        self.run_command("build")
        self.assertEqual(before, [(self.root / name).read_bytes() for name in ("INDEX.md", "catalog.csv")])

    def test_public_build_ignores_personal_zotero_state_and_downloads(self):
        path = self.add_entry()
        self.assertEqual(self.run_command("build")[0], 0)
        before = [(self.root / name).read_bytes() for name in ("INDEX.md", "catalog.csv")]
        local = self.root / ".local" / "zotero"
        local.mkdir(parents=True)
        for user_id, item_key in ((123, "ABCD1234"), (456, "WXYZ5678")):
            with self.subTest(user_id=user_id):
                (local / "config.json").write_text(
                    json.dumps({"api_base": "http://127.0.0.1:23119", "user_id": user_id}),
                    encoding="utf-8",
                )
                (local / "bindings.json").write_text(json.dumps({
                    "version": 1,
                    "libraries": {f"user:{user_id}": {
                        "library": {"type": "user", "id": user_id},
                        "entries": {"example": {
                            "item_key": item_key,
                            "reading_attachment_key": "PDF12345",
                            "attachments": {"resources/paper.pdf": {
                                "key": "PDF12345", "status": "available",
                            }},
                        }},
                    }},
                }), encoding="utf-8")
                (local / "index.md").write_text(
                    f"[个人条目](zotero://select/library/items/{item_key})", encoding="utf-8",
                )
                resources = path.parent / "resources"
                resources.mkdir(exist_ok=True)
                (resources / "paper.pdf").write_bytes(b"%PDF-1.7\nlocal-only")
                self.assertEqual(self.run_command("build")[0], 0)
                self.assertEqual(
                    before,
                    [(self.root / name).read_bytes() for name in ("INDEX.md", "catalog.csv")],
                )
        (local / "bindings.json").write_text("不是 JSON", encoding="utf-8")
        self.assertEqual(self.run_command("build")[0], 0)
        self.assertEqual(before, [(self.root / name).read_bytes() for name in ("INDEX.md", "catalog.csv")])

    def test_public_metadata_rejects_zotero_binding_even_when_empty(self):
        for value in (None, {}, {"item_key": "ABCD1234", "library_id": 0}):
            with self.subTest(value=value):
                self.add_entry(zotero=value)
                with self.assertRaisesRegex(catalog.CatalogError, "禁止 zotero"):
                    self.load()

    def test_public_attachments_reject_private_binding_and_status_fields(self):
        for key in ("zotero_item_key", "zotero_attachment_key", "local_status", "downloaded_on"):
            for value in (None, "", "ABCD1234"):
                with self.subTest(key=key, value=value):
                    self.add_entry(attachments=[{
                        "url": "https://example.org/paper.pdf",
                        "path": "resources/paper.pdf",
                        key: value,
                    }])
                    with self.assertRaisesRegex(catalog.CatalogError, f"禁止 {key}"):
                        self.load()

    def test_historical_zotero_provenance_does_not_create_personal_links(self):
        self.add_entry(attachments=[{
            "url": "https://example.org/paper.pdf",
            "path": "resources/paper.pdf",
            "source_zotero_item_key": "ABCD1234",
            "source_zotero_attachment_key": "PDF12345",
        }])
        self.assertEqual(self.run_command("build")[0], 0)
        index = (self.root / "INDEX.md").read_text(encoding="utf-8")
        csv_text = (self.root / "catalog.csv").read_text(encoding="utf-8-sig")
        attachment = json.loads(next(csv.DictReader(io.StringIO(csv_text)))["attachments"])[0]
        self.assertEqual(attachment["source_zotero_item_key"], "ABCD1234")
        self.assertEqual(attachment["source_zotero_attachment_key"], "PDF12345")
        self.assertNotIn("zotero://", index + csv_text)

    def test_invalid_build_preserves_existing_outputs(self):
        self.add_entry(tags=["topic/typo"])
        (self.root / "INDEX.md").write_text("previous", encoding="utf-8")
        (self.root / "catalog.csv").write_text("previous", encoding="utf-8")
        result, _, error = self.run_command("build")
        self.assertEqual(result, 1)
        self.assertIn("未知标签", error)
        for name in ("INDEX.md", "catalog.csv"):
            self.assertEqual((self.root / name).read_text(encoding="utf-8"), "previous")

    def test_duplicate_global_id(self):
        self.add_entry()
        self.add_entry("blogs")
        with self.assertRaisesRegex(catalog.CatalogError, "重复 id"):
            self.load()

    def test_directory_id_mismatch(self):
        path = self.add_entry()
        path.write_text(path.read_text(encoding="utf-8").replace("id: example", "id: different"), encoding="utf-8")
        with self.assertRaisesRegex(catalog.CatalogError, "目录名"):
            self.load()

    def test_category_type_mismatch(self):
        self.add_entry(type="blog")
        with self.assertRaisesRegex(catalog.CatalogError, "type 必须"):
            self.load()

    def test_invalid_status_url_and_duplicate_tags(self):
        cases = [({"status": "done"}, "status"), ({"status": []}, "status"),
                 ({"url": "file:///private/file"}, "url"), ({"url": "https://exa mple.org"}, "url"),
                 ({"tags": ["topic/quantization"] * 2}, "重复标签")]
        for updates, expected in cases:
            with self.subTest(updates=updates):
                self.add_entry(**updates)
                with self.assertRaisesRegex(catalog.CatalogError, expected):
                    self.load()

    def test_attachment_paths_cannot_escape_resources(self):
        for path in ("../paper.pdf", "resources/../paper.pdf", "/resources/paper.pdf", "C:/paper.pdf", "resources\\paper.pdf", "resources/", "resource/paper.pdf"):
            with self.subTest(path=path):
                self.add_entry(attachments=[{"url": "https://example.org/paper.pdf", "path": path}])
                with self.assertRaisesRegex(catalog.CatalogError, "附件"):
                    self.load()

    def test_malformed_and_duplicate_yaml(self):
        path = self.add_entry()
        for text in ("title: [broken", "id: one\nid: two\n", "- list\n"):
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(catalog.CatalogError):
                    self.load()

    def test_missing_metadata_is_reported(self):
        (self.root / "papers" / "orphan").mkdir()
        with self.assertRaisesRegex(catalog.CatalogError, "meta.yaml"):
            self.load()

    def test_alias_collision_is_reported(self):
        path = self.root / "tags.yaml"
        path.write_text(path.read_text(encoding="utf-8").replace("aliases: [orin]", "aliases: [quantization]"), encoding="utf-8")
        with self.assertRaisesRegex(catalog.CatalogError, "别名冲突"):
            self.load()


if __name__ == "__main__":
    unittest.main()
