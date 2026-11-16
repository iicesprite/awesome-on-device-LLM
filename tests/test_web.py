import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import web


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="resource-web-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for category in ("papers", "blogs", "docs"):
            (self.root / category).mkdir()
        (self.root / "tags.yaml").write_text(
            "tags:\n  topic/inference: {label: 推理, aliases: [inference]}\n"
            "  target/weight: {label: 权重, aliases: [weight]}\n", encoding="utf-8",
        )
        self.folder = self.root / "papers" / "example"
        self.folder.mkdir()
        self.metadata = {
            "id": "example", "type": "paper", "title": "推理研究", "url": "https://example.org/research",
            "status": "unread", "summary": "收藏理由", "tags": ["topic/inference"],
            "source_checked_on": "2026-11-06", "attachments": [],
        }
        self.card = self.folder / "resource.md"
        self.card.write_text(
            "---\n" + yaml.safe_dump(self.metadata, allow_unicode=True) + "---\n\n# 原始正文\n", encoding="utf-8",
        )
        self.server = web.WorkspaceServer(self.root, 0)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=5)

    def request(self, method, path, body=None, headers=None, authenticated=True):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        request_headers = {}
        if authenticated:
            request_headers["X-Catalog-Token"] = self.server.token
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            data = response.read()
            if response.getheader("Content-Type", "").startswith("application/json"):
                data = json.loads(data)
            return response.status, data, response.getheaders()
        finally:
            connection.close()

    def test_edit_visible_to_search_and_stale_editor_cannot_overwrite(self):
        _, detail, _ = self.request("GET", "/api/resources/example")
        status, saved, _ = self.request("PUT", "/api/resources/example", {
            "revision": detail["revision"],
            "updates": {"summary": "更新的收藏理由", "tags": ["topic/inference", "target/weight"]},
        })
        self.assertEqual(status, 200)
        self.assertNotEqual(saved["revision"], detail["revision"])
        status, found, _ = self.request("GET", "/api/catalog?tag=inference&tag=weight")
        self.assertEqual(status, 200)
        self.assertEqual([entry["id"] for entry in found["entries"]], ["example"])
        self.assertEqual(found["entries"][0]["summary"], "更新的收藏理由")
        status, _, _ = self.request("PUT", "/api/resources/example", {
            "revision": detail["revision"], "updates": {"summary": "过时覆盖"},
        })
        self.assertEqual(status, 409)
        self.assertEqual(web.catalog.read_frontmatter(self.card)["summary"], "更新的收藏理由")
        self.assertIn("更新的收藏理由", (self.root / "INDEX.md").read_text(encoding="utf-8"))

    def test_new_resource_has_empty_notes_and_duplicate_is_conflict(self):
        data = {**self.metadata, "id": "new-article", "type": "blog", "title": "另一个明确不同的文章", "url": "https://example.org/new"}
        status, detail, _ = self.request("POST", "/api/resources", {"metadata": data})
        self.assertEqual(status, 201)
        self.assertEqual(detail["entry"]["metadata_path"], "blogs/new-article/resource.md")
        self.assertEqual((self.root / "blogs/new-article/notes.md").read_bytes(), b"")
        self.assertEqual(detail["entry"]["notes_path"], "blogs/new-article/notes.md")
        status, result, _ = self.request("POST", "/api/resources", {"metadata": {**data, "id": "duplicate"}})
        self.assertEqual(status, 409)
        self.assertEqual(result["duplicates"][0]["id"], "new-article")
        self.assertFalse((self.root / "blogs/duplicate").exists())

    def test_invalid_metadata_does_not_change_resource_or_generate_indexes(self):
        before = self.card.read_bytes()
        _, detail, _ = self.request("GET", "/api/resources/example")
        status, _, _ = self.request("PUT", "/api/resources/example", {
            "revision": detail["revision"], "updates": {"tags": ["unknown/tag"]},
        })
        self.assertEqual(status, 400)
        self.assertEqual(self.card.read_bytes(), before)
        self.assertFalse((self.root / "INDEX.md").exists())

    def test_foreign_sites_and_rebinding_hosts_cannot_read_or_mutate(self):
        cases = [
            {"Host": f"attacker.example:{self.server.server_port}"},
            {"Origin": "https://attacker.example"},
            {"Sec-Fetch-Site": "cross-site"},
        ]
        before = self.card.read_bytes()
        for headers in cases:
            with self.subTest(headers=headers):
                self.assertEqual(self.request("GET", "/api/catalog", headers=headers)[0], 403)
                self.assertEqual(self.request("POST", "/api/resources", {"metadata": self.metadata}, headers=headers)[0], 403)
        self.assertEqual(self.card.read_bytes(), before)

    def test_writes_require_session_token_and_json(self):
        before = self.card.read_bytes()
        self.assertEqual(self.request("POST", "/api/resources", {}, authenticated=False)[0], 403)
        self.assertEqual(self.request("POST", "/api/resources", {}, headers={"X-Catalog-Token": "stale"})[0], 403)
        self.assertEqual(self.request("POST", "/api/resources", "{}", headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/resources", "{", headers={"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/resources", "[]", headers={"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.card.read_bytes(), before)

    def test_only_entry_attachments_are_served_and_active_content_downloads(self):
        assets = self.folder / "resources"
        assets.mkdir()
        (assets / "plot.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        (assets / "unsafe.svg").write_text('<svg onload="alert(1)"></svg>', encoding="utf-8")
        secret = self.root / ".local"
        secret.mkdir()
        (secret / "secret.json").write_text('{"private":true}', encoding="utf-8")
        self.assertEqual(self.request("GET", "/files/example/resources/plot.png")[1], b"\x89PNG\r\n\x1a\nfixture")
        status, _, headers = self.request("GET", "/files/example/resources/unsafe.svg")
        self.assertEqual(status, 200)
        self.assertIn(("Content-Disposition", "attachment"), headers)
        self.assertIn(("Content-Type", "application/octet-stream"), headers)
        for path in (
            "/.local/secret.json", "/files/example/../../../.local/secret.json",
            "/files/example/resources/%2e%2e/%2e%2e/%2e%2e/.local/secret.json",
            "/files/example/resource.md", "/files/example/resources/missing.pdf",
        ):
            with self.subTest(path=path):
                self.assertIn(self.request("GET", path)[0], (400, 403, 404))


if __name__ == "__main__":
    unittest.main()
