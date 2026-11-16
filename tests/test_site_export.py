import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import catalog
import site_export
from note_render import render_markdown

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


class SiteExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "library"
        self.root.mkdir()
        for category in catalog.CATEGORIES:
            (self.root / category).mkdir()
        (self.root / "tags.yaml").write_text(yaml.safe_dump({"tags": {"target/kv-cache": {"label": "缓存", "aliases": ["KV"], "private": "不导出"}}}, allow_unicode=True), encoding="utf-8")
        self.output = self.root / "site" / ".generated"
        self.add_entry("alpha", notes="# 笔记正文\n\n只有这里出现的知识。")

    def add_entry(self, entry_id, *, category="papers", notes=None, body="", **fields):
        folder = self.root / category / entry_id
        folder.mkdir()
        data = {"id": entry_id, "title": "资源 " + entry_id, "type": catalog.CATEGORIES[category], "url": "https://example.org/article", "status": "unread", "summary": "中文摘要", "tags": ["target/kv-cache"], "authors": ["作者"], "year": 2026, **fields}
        (folder / "resource.md").write_text("---\n" + yaml.safe_dump(data, allow_unicode=True, sort_keys=False) + "---\n" + body, encoding="utf-8")
        if notes is not None:
            (folder / "notes.md").write_text(notes, encoding="utf-8")
        return folder

    def export(self, **kwargs):
        return site_export.export_site(self.root, self.output, **kwargs)

    def test_pages_catalog_and_sources_are_separate_and_read_only(self):
        folder = self.add_entry("beta", category="blogs", body="资源正文", private_field="PRIVATE-TOKEN", attachments=[{"url": "https://example.org/not-downloaded.pdf", "path": "resources/not-downloaded.pdf"}])
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.export()
        for path, data in before.items():
            self.assertEqual((self.root / path).read_bytes(), data)
        payload = json.loads((self.output / "catalog.json").read_text(encoding="utf-8"))
        entries = {entry["id"]: entry for entry in payload["entries"]}
        self.assertEqual(entries["alpha"]["notes_path"], "notes/alpha.html")
        self.assertEqual(entries["beta"]["notes_path"], "")
        self.assertEqual(entries["beta"]["status"], "unread")
        self.assertNotIn("attachments", entries["beta"])
        self.assertNotIn("private_field", entries["beta"])
        self.assertNotIn("只有这里出现", entries["alpha"]["search_text"])
        self.assertEqual(payload["tags"]["target/kv-cache"], {"label": "缓存", "aliases": ["KV"]})
        resource = (self.output / "resources" / "alpha.md").read_text(encoding="utf-8")
        notes = (self.output / "notes" / "alpha.md").read_text(encoding="utf-8")
        self.assertNotIn("笔记正文", resource)
        self.assertIn("笔记正文", notes)
        self.assertIn("?tag=target%2Fkv-cache", notes)
        self.assertIn("<ResourceCatalog />", (self.output / "index.md").read_text(encoding="utf-8"))
        self.assertFalse((folder / "resources" / "not-downloaded.pdf").exists())
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"PRIVATE-TOKEN", path.read_bytes())
        self.assertEqual(len(list(self.output.rglob("resource.md"))), 0)

    def test_base_cross_resource_links_and_blocked_pdf_links(self):
        self.add_entry("beta", notes="目标")
        source = self.root / "papers" / "alpha" / "notes.md"
        source.write_text("\n\n".join([
            "[本卡](resource.md) [本笔记](notes.md) [另一卡](../beta/resource.md) [另一笔记](../beta/notes.md)",
            "[外部](https://example.org/page) [PDF](https://example.org/paper.PDF?x=1)",
            "[arxiv](https://arxiv.org/pdf/1234.5678) [review](https://openreview.net/pdf?id=x)",
            "[附件](resources/secret.txt) [隐私](../../.local/private.md) [坏](javascript:alert(1))",
        ]), encoding="utf-8")
        self.export(base="/project/")
        result = (self.output / "notes" / "alpha.md").read_text(encoding="utf-8")
        for path in ("resources/alpha.html", "notes/alpha.html", "resources/beta.html", "notes/beta.html"):
            self.assertIn('href="/project/' + path + '"', result)
        self.assertIn('href="https://example.org/page"', result)
        for target in ("https://example.org/paper.PDF", "https://arxiv.org/pdf", "https://openreview.net/pdf", "resources/secret", ".local/private"):
            self.assertNotIn('href="' + target, result)
        self.assertIn(">PDF</a>", result)

    def test_only_opted_in_translation_is_exported_and_linked(self):
        folder = self.add_entry("pariskv", notes="# 阅读笔记")
        (folder / "part_translate.md").write_text("# 部分译文正文", encoding="utf-8")
        (self.root / "papers" / "alpha" / "part_translate.md").write_text("非公开译稿", encoding="utf-8")
        self.export(base="/project/")
        for page in ("resources/pariskv.md", "notes/pariskv.md"):
            body = (self.output / page).read_text(encoding="utf-8")
            self.assertIn('href="/project/translations/pariskv.html"', body)
        translation = (self.output / "translations" / "pariskv.md").read_text(encoding="utf-8")
        self.assertIn("部分译文正文", translation)
        self.assertIn('href="/project/notes/pariskv.html"', translation)
        self.assertFalse((self.output / "translations" / "alpha.md").exists())

    def test_only_referenced_note_rasters_are_copied(self):
        folder = self.root / "papers" / "alpha"
        assets = folder / "resources" / "assets"
        assets.mkdir(parents=True)
        for name in ("used.png", "unused.png", "card-only.png"):
            (assets / name).write_bytes(PNG)
        (assets / "disguised.png").write_bytes(b"%PDF private")
        (assets / "bad.svg").write_text("<svg onload='bad'/>", encoding="utf-8")
        (folder / "resources" / "paper.pdf").write_bytes(b"%PDF private")
        (self.root / ".local").mkdir()
        (self.root / ".local" / "secret").write_text("PRIVATE", encoding="utf-8")
        card = folder / "resource.md"
        card.write_text(card.read_text(encoding="utf-8") + "![卡图](resources/assets/card-only.png)", encoding="utf-8")
        (folder / "notes.md").write_text("![配图](resources/assets/used.png) ![缺图](resources/assets/missing.png) ![伪装](resources/assets/disguised.png) ![SVG](resources/assets/bad.svg)", encoding="utf-8")
        result = self.export(base="/repo/")
        self.assertEqual(result["assets"], 1)
        copied = self.output / "public" / "assets" / "alpha"
        self.assertEqual([p.name for p in copied.iterdir()], ["used.png"])
        self.assertEqual((copied / "used.png").read_bytes(), PNG)
        notes = (self.output / "notes" / "alpha.md").read_text(encoding="utf-8")
        self.assertIn('src="/repo/assets/alpha/used.png"', notes)
        self.assertIn("图片不可用：缺图", notes)
        self.assertIn("图片不可用：伪装", notes)
        self.assertFalse(any(p.suffix == ".pdf" for p in self.output.rglob("*")))
        self.assertFalse((self.output / ".local").exists())

    def test_html_vue_math_and_mermaid_are_inert(self):
        folder = self.add_entry("attack", title='</div><script>evil()</script>{{ danger }}', summary='" : bad\n---\n<script>x</script>', notes='---\nprivate: hidden-note-field\n---\n# 章节\n\n</div><script>evil()</script> {{ danger }}\n\n$x < y$\n\n```mermaid\ngraph TD\n A-->B\n```')
        self.export()
        page = (self.output / "notes" / "attack.md").read_text(encoding="utf-8")
        body = site_export.without_frontmatter(page)
        data = catalog.read_frontmatter(self.output / "notes" / "attack.md")
        self.assertIn("{{ danger }}", data["title"])
        self.assertIn('<div class="resource-content" v-pre>', body)
        self.assertNotIn("<script>", body)
        self.assertNotIn("hidden-note-field", body)
        self.assertIn("&lt;script&gt;", body)
        self.assertIn('class="math-source"', body)
        self.assertIn('data-math-display="inline"', body)
        self.assertIn('class="language-mermaid"', body)
        self.assertEqual(body.count("</div>"), 1)

    def test_unmarked_output_is_preserved_and_marked_output_regenerates(self):
        self.output.mkdir(parents=True)
        user_file = self.output / "mine.txt"
        user_file.write_text("keep", encoding="utf-8")
        with self.assertRaises(catalog.CatalogError):
            self.export()
        self.assertEqual(user_file.read_text(encoding="utf-8"), "keep")
        user_file.unlink()
        self.export()
        (self.output / "stale.md").write_text("old", encoding="utf-8")
        self.export()
        self.assertFalse((self.output / "stale.md").exists())

    def test_invalid_catalog_and_base_leave_previous_output_intact(self):
        self.export()
        original = (self.output / "catalog.json").read_bytes()
        for base in ("repo", "/repo", "//evil/", "/../", "/x/?bad/"):
            with self.subTest(base=base), self.assertRaises(catalog.CatalogError):
                self.export(base=base)
        card = self.root / "papers" / "alpha" / "resource.md"
        card.write_text(card.read_text(encoding="utf-8").replace("target/kv-cache", "target/unknown"), encoding="utf-8")
        with self.assertRaises(catalog.CatalogError):
            self.export()
        self.assertEqual((self.output / "catalog.json").read_bytes(), original)

    def symlink(self, target, link, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"当前环境不支持符号链接：{exc}")

    def test_symlink_image_is_not_copied(self):
        folder = self.root / "papers" / "alpha"
        assets = folder / "resources" / "assets"
        assets.mkdir(parents=True)
        outside = Path(self.temp.name) / "private.png"
        outside.write_bytes(PNG)
        self.symlink(outside, assets / "private.png")
        (folder / "notes.md").write_text("![秘密](resources/assets/private.png)", encoding="utf-8")
        result = self.export()
        self.assertEqual(result["assets"], 0)
        self.assertIn("图片不可用：秘密", (self.output / "notes" / "alpha.md").read_text(encoding="utf-8"))

    def test_symlink_resource_and_output_are_rejected(self):
        outside = Path(self.temp.name) / "private"
        outside.mkdir()
        self.symlink(outside, self.root / "blogs" / "leak", directory=True)
        with self.assertRaises(catalog.CatalogError):
            self.export()
        (self.root / "blogs" / "leak").unlink()
        self.output.parent.mkdir()
        self.symlink(outside, self.output, directory=True)
        with self.assertRaises(catalog.CatalogError):
            self.export()
        self.assertEqual(list(outside.iterdir()), [])

    def test_source_output_locations_are_rejected(self):
        for output in (self.root, self.root.parent, self.root / "papers" / "empty"):
            with self.subTest(output=output), self.assertRaises(catalog.CatalogError):
                site_export.export_site(self.root, output)

    def test_cli_environment_base_and_failure_exit(self):
        with patch.dict(os.environ, {"SITE_BASE": "/environment/"}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(site_export.main(["--root", str(self.root)]), 0)
        self.assertIn('/environment/resources/alpha.html', (self.output / "notes" / "alpha.md").read_text(encoding="utf-8"))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(site_export.main(["--root", str(self.root), "--base", "bad"]), 1)

    def test_optional_renderer_resolver_leaves_default_behavior_unchanged(self):
        source = "[笔记](notes.md) ![图](resources/assets/p.png)"
        default = render_markdown(source, "alpha")
        self.assertIn("#resource=alpha&amp;tab=notes", default)
        self.assertIn("/files/alpha/resources/assets/p.png", default)
        calls = []
        def resolver(url, *, image=False):
            calls.append((url, image))
            return None if image else "/project/notes/alpha.html"
        rendered = render_markdown(source, "alpha", url_resolver=resolver)
        self.assertEqual(calls, [("notes.md", False), ("resources/assets/p.png", True)])
        self.assertIn('href="/project/notes/alpha.html"', rendered)
        self.assertIn("图片不可用：图", rendered)
        self.assertNotIn("<img", rendered)


if __name__ == "__main__":
    unittest.main()
