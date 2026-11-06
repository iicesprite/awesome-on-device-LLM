import importlib.util
from pathlib import Path
import socket
import ssl
import time
import unittest
from unittest import mock
from email.message import Message


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "resource_import.py"
SPEC = importlib.util.spec_from_file_location("resource_import", SCRIPT)
resource_import = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resource_import)


class MetadataTests(unittest.TestCase):
    def preview(self, html, url="https://example.org/article", entries=()):
        with mock.patch.object(resource_import, "fetch_html", return_value=(url, html)):
            return resource_import.preview(url, entries)

    def test_citation_precedes_og_jsonld_and_page_title(self):
        result = self.preview('''<title>Navigation title</title>
            <meta property="og:title" content="OG title">
            <meta name="citation_title" content="研究与推理">
            <meta name="citation_author" content="Alice Example">
            <meta name="citation_author" content="张三">
            <meta name="citation_publication_date" content="2023/04/12">
            <meta name="citation_journal_title" content="Explicit Journal">
            <meta name="citation_doi" content="https://doi.org/10.1234/ABC">
            <meta property="og:description" content="Original &amp; untranslated">
            <script type="application/ld+json">{"@type":"ScholarlyArticle",
              "headline":"JSON title", "author":{"name":"Other"},
              "datePublished":"2021-01-01"}</script>''')
        data = result["metadata"]
        self.assertEqual(data["title"], "研究与推理")
        self.assertRegex(data["id"], r"^resource-[0-9a-f]{12}$")
        self.assertEqual(data["authors"], ["Alice Example", "张三"])
        self.assertEqual(data["year"], 2023)
        self.assertEqual(data["venue"], "Explicit Journal")
        self.assertEqual(data["doi"], "10.1234/abc")
        self.assertEqual(data["type"], "paper")
        self.assertEqual(data["summary"], "")
        self.assertEqual(data["tags"], [])
        self.assertNotIn("links", data)
        self.assertNotIn("attachments", data)
        self.assertEqual(result["description"], "Original & untranslated")
        self.assertTrue(any("撤稿" in warning and "未核实" in warning for warning in result["warnings"]))
        self.assertTrue(any("未下载 PDF" in warning for warning in result["warnings"]))
        self.assertEqual(self.preview('<title>研究与推理</title>')["metadata"]["id"], data["id"])

    def test_jsonld_graph_and_explicit_authorship(self):
        result = self.preview('''<script type="application/ld+json">{
            "@graph":[{"@type":"Organization","name":"Not an author"},
            {"@type":"ScholarlyArticle","headline":"An actual paper",
             "author":[{"@type":"Person","name":"Named Author"},"Second Author"],
             "datePublished":"2020-06-01", "description":"English abstract",
             "identifier":{"@type":"PropertyValue","propertyID":"DOI","value":"10.1234/XYZ"}}]
            }</script>''')
        self.assertEqual(result["metadata"]["authors"], ["Named Author", "Second Author"])
        self.assertEqual(result["metadata"]["year"], 2020)
        self.assertEqual(result["metadata"]["doi"], "10.1234/xyz")
        self.assertEqual(result["description"], "English abstract")

    def test_does_not_infer_author_or_year_from_visible_text_or_url(self):
        result = self.preview('''<title>2024 Research by Famous Person</title>
            <p>Authors: Famous Person. Published 2024. github.com/example/repo</p>
            <script type="application/ld+json">{invalid}</script>''',
            url="https://example.org/2024/author/famous")
        self.assertNotIn("authors", result["metadata"])
        self.assertNotIn("year", result["metadata"])
        self.assertNotIn("links", result["metadata"])
        self.assertEqual(result["description"], "")
        self.assertEqual(result["metadata"]["status"], "unread")

    def test_og_precedes_jsonld_title_but_combines_explicit_identity(self):
        result = self.preview('''<meta property="og:title" content="Preferred title">
            <script type="application/ld+json">{"@type":"ScholarlyArticle",
            "headline":"Other title","identifier":"https://doi.org/10.5555/PAPER"}</script>''',
            entries=[{"id":"existing", "title":"Old title", "doi":"10.5555/paper",
                      "metadata_path":"docs/existing/resource.md"}])
        self.assertEqual(result["metadata"]["title"], "Preferred title")
        self.assertEqual(result["duplicates"][0]["id"], "existing")

    def test_html_and_scripts_stay_data(self):
        result = self.preview('''<meta property="og:title" content="&lt;img src=x onerror=alert(1)&gt;">
            <meta name="description" content="&lt;script&gt;bad()&lt;/script&gt;">''')
        self.assertEqual(result["metadata"]["title"], "<img src=x onerror=alert(1)>")
        self.assertEqual(result["description"], "<script>bad()</script>")

    def test_missing_title_and_fetch_failure_are_errors_not_fabricated_metadata(self):
        with self.assertRaises(resource_import.ImportError):
            self.preview("<html>no metadata</html>")
        with mock.patch.object(resource_import, "fetch_html", side_effect=resource_import.ImportError("失败")):
            with self.assertRaisesRegex(resource_import.ImportError, "失败"):
                resource_import.preview("https://example.org", [])


class DuplicateTests(unittest.TestCase):
    def test_all_categories_and_identity_strength(self):
        entries = [
            {"id":"paper", "title":"Different", "url":"https://arxiv.org/pdf/2401.12345v1.pdf", "metadata_path":"papers/paper/resource.md"},
            {"id":"blog", "title":"Different", "doi":"doi:10.1234/ABC", "metadata_path":"blogs/blog/resource.md"},
            {"id":"doc", "title":" A title ", "url":"https://example.org/doc", "metadata_path":"docs/doc/resource.md"},
        ]
        self.assertEqual(resource_import.find_duplicates({"url":"https://arxiv.org/abs/2401.12345v9"}, entries)[0]["id"], "paper")
        self.assertEqual(resource_import.find_duplicates({"url":"https://doi.org/10.1234/abc"}, entries)[0]["id"], "blog")
        self.assertEqual(resource_import.find_duplicates({"title":"a TITLE"}, entries)[0]["id"], "doc")
        self.assertEqual(entries[0]["title"], "Different")

    def test_old_arxiv_identifiers_and_metadata_links(self):
        entries = [{"id":"old", "url":"https://arxiv.org/abs/hep-th/9901001v2", "title":"old"}]
        result = resource_import.find_duplicates({"links":{"paper":"https://export.arxiv.org/pdf/hep-th/9901001v4.pdf"}}, entries)
        self.assertEqual(result[0]["id"], "old")
        self.assertEqual(resource_import.find_duplicates({"url":"https://evil.test/abs/hep-th/9901001v2"}, entries), [])

    def test_query_path_and_scheme_distinctions_are_preserved(self):
        entries = [{"id":"one", "url":"https://EXAMPLE.org:443/view?id=1#section", "title":"One"}]
        self.assertEqual(resource_import.find_duplicates({"url":"https://example.org/view?id=1#other"}, entries)[0]["id"], "one")
        for url in ("https://example.org/view?id=2", "http://example.org/view?id=1", "https://example.org/view/?id=1"):
            with self.subTest(url=url):
                self.assertEqual(resource_import.find_duplicates({"url":url}, entries), [])

    def test_distinct_arxiv_versions_match_but_distinct_ids_do_not(self):
        entries = [{"id":"one", "url":"https://arxiv.org/html/2401.12345v2"}]
        self.assertTrue(resource_import.find_duplicates({"url":"https://arxiv.org/abs/2401.12345"}, entries))
        self.assertFalse(resource_import.find_duplicates({"url":"https://arxiv.org/abs/2401.12346"}, entries))

    def test_arxiv_doi_and_arxiv_url_are_one_identity(self):
        entries = [{"id":"one", "url":"https://arxiv.org/abs/2401.12345v2"}]
        for identity in ("https://doi.org/10.48550/arXiv.2401.12345", "10.48550/arxiv.2401.12345"):
            with self.subTest(identity=identity):
                self.assertEqual(resource_import.find_duplicates({"doi":identity}, entries)[0]["id"], "one")
        self.assertEqual(resource_import.find_duplicates(
            {"url":"https://arxiv.org/pdf/2401.12345v3.pdf"},
            [{"id":"doi-entry", "doi":"10.48550/arxiv.2401.12345"}])[0]["id"], "doi-entry")

    def test_primary_url_matches_existing_alternate_sources(self):
        entry = {
            "id": "existing", "title": "Original title", "url": "https://example.org/paper",
            "links": {"publication": "https://publisher.example/article", "code": "https://example.org/shared-code"},
            "attachments": [{"url": "https://example.org/paper.pdf", "path": "resources/paper.pdf"}],
        }
        for url in ("https://publisher.example/article", "https://example.org/paper.pdf"):
            with self.subTest(url=url):
                matches = resource_import.find_duplicates({"title": "Source title", "url": url}, [entry])
                self.assertEqual([item["id"] for item in matches], ["existing"])
        unrelated = {"url": "https://example.org/different", "links": {"code": "https://example.org/shared-code"}}
        self.assertEqual(resource_import.find_duplicates(unrelated, [entry]), [])

    def test_bibtex_punctuation_does_not_hide_the_same_title(self):
        matches = resource_import.find_duplicates(
            {"title": "{SolidAttention}: {Low-Latency} Serving"},
            [{"id": "solid", "title": "SolidAttention: Low-Latency Serving"}],
        )
        self.assertEqual([item["id"] for item in matches], ["solid"])


class FakeSocket:
    def shutdown(self, _how):
        pass

    def close(self):
        pass


class FakeResponse:
    def __init__(self, data=b"<title>Test</title>", status=200, headers=None):
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = "text/html; charset=utf-8"
        for key, value in (headers or {}).items():
            if key in self.headers:
                self.headers.replace_header(key, value)
            else:
                self.headers[key] = value
        self.data = data

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read1(self, size):
        chunk, self.data = self.data[:size], self.data[size:]
        return chunk


class FakeConnection:
    def __init__(self, response):
        self.response = response
        self.sock = FakeSocket()
        self.requests = []

    def connect(self):
        pass

    def request(self, *args, **kwargs):
        self.requests.append((args, kwargs))

    def getresponse(self):
        return self.response

    def close(self):
        pass


class FetchSafetyTests(unittest.TestCase):
    def addresses(self, ip):
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        address = (ip, 443, 0, 0) if family == socket.AF_INET6 else (ip, 443)
        return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", address)]

    def test_private_mixed_mapped_and_loopback_dns_are_rejected(self):
        for ip in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "::ffff:8.8.8.8", "224.0.0.1"):
            with self.subTest(ip=ip), mock.patch.object(resource_import.socket, "getaddrinfo", return_value=self.addresses(ip)):
                with self.assertRaises(resource_import.ImportError):
                    resource_import._public_addresses("source.test", 443, time.monotonic() + 1)
        with mock.patch.object(resource_import.socket, "getaddrinfo", return_value=self.addresses("8.8.8.8") + self.addresses("10.0.0.1")):
            with self.assertRaises(resource_import.ImportError):
                resource_import._public_addresses("source.test", 443, time.monotonic() + 1)

    def test_invalid_urls_do_not_connect(self):
        for url in ("file:///etc/passwd", "ftp://example.org/a", "http://user:secret@example.org", "http://example.org:99999/", "http://example.org\\@127.0.0.1", "https://example.org/\r\nInjected:x"):
            with self.subTest(url=url), mock.patch.object(resource_import, "_connection") as connect:
                with self.assertRaises(resource_import.ImportError):
                    resource_import.fetch_html(url)
                connect.assert_not_called()

    def test_redirect_revalidates_dns_before_connection(self):
        connection = FakeConnection(FakeResponse(status=302, headers={"Location":"http://127.0.0.1/secret"}))
        with mock.patch.object(resource_import.socket, "getaddrinfo", side_effect=[self.addresses("8.8.8.8"), self.addresses("127.0.0.1")]) as resolve, mock.patch.object(resource_import, "_connection", return_value=connection) as connect:
            with self.assertRaisesRegex(resource_import.ImportError, "非公网"):
                resource_import.fetch_html("https://source.test")
            self.assertEqual(resolve.call_count, 2)
            self.assertEqual(connect.call_count, 1)

    def test_connection_pins_numeric_address_and_keeps_https_host(self):
        address = self.addresses("8.8.8.8")[0]
        record = (address[0], address[1], address[2], address[4])
        with mock.patch.object(resource_import.socket, "socket") as factory:
            connection = resource_import._connection("source.test", 443, True, record, time.monotonic() + 1)
            self.assertIsInstance(connection, resource_import.http.client.HTTPSConnection)
            self.assertEqual(connection.host, "source.test")
            self.assertEqual(connection._context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(connection._context.check_hostname)
            connection._create_connection(("source.test", 443))
            factory.return_value.connect.assert_called_once_with(("8.8.8.8", 443))
            context = mock.Mock()
            context.wrap_socket.return_value = factory.return_value
            connection._context = context
            connection.connect()
            context.wrap_socket.assert_called_once_with(factory.return_value, server_hostname="source.test")

    def test_bounded_html_and_pdf_rejection(self):
        responses = [FakeResponse(headers={"Content-Length":str(resource_import.MAX_BYTES + 1)}),
                     FakeResponse(data=b"short", headers={"Content-Length":"100"}),
                     FakeResponse(data=b"x" * (resource_import.MAX_BYTES + 1)),
                     FakeResponse(headers={"Content-Type":"application/pdf"}),
                     FakeResponse(headers={"Content-Encoding":"gzip"}),
                     FakeResponse(status=500)]
        for response in responses:
            with self.subTest(headers=str(response.headers)), mock.patch.object(resource_import, "_public_addresses", return_value=[None]), mock.patch.object(resource_import, "_connection", return_value=FakeConnection(response)):
                with self.assertRaises(resource_import.ImportError):
                    resource_import.fetch_html("https://source.test")

    def test_redirect_limit(self):
        with mock.patch.object(resource_import, "_public_addresses", return_value=[None]), mock.patch.object(resource_import, "_connection", side_effect=lambda *args: FakeConnection(FakeResponse(status=302, headers={"Location":"/again"}))) as connect:
            with self.assertRaisesRegex(resource_import.ImportError, "重定向次数"):
                resource_import.fetch_html("https://source.test")
            self.assertEqual(connect.call_count, resource_import.MAX_REDIRECTS + 1)

    def test_dns_timeout_is_an_explicit_error(self):
        result_queue = mock.Mock()
        result_queue.get.side_effect = resource_import.queue.Empty
        with mock.patch.object(resource_import.queue, "Queue", return_value=result_queue), mock.patch.object(resource_import.socket, "getaddrinfo", return_value=self.addresses("8.8.8.8")):
            with self.assertRaisesRegex(resource_import.ImportError, "解析超时"):
                resource_import._public_addresses("source.test", 443, time.monotonic() + 1)

    def test_expired_total_deadline_does_not_connect(self):
        with mock.patch.object(resource_import.time, "monotonic", side_effect=[0, 100]), mock.patch.object(resource_import, "_public_addresses", return_value=[None]), mock.patch.object(resource_import, "_connection") as connect:
            with self.assertRaisesRegex(resource_import.ImportError, "超时"):
                resource_import.fetch_html("https://source.test")
            connect.assert_not_called()

    def test_success_returns_actual_final_url_and_no_proxy_auth(self):
        first = FakeConnection(FakeResponse(status=302, headers={"Location":"/final"}))
        second = FakeConnection(FakeResponse())
        with mock.patch.object(resource_import, "_public_addresses", return_value=[None]), mock.patch.object(resource_import, "_connection", side_effect=[first, second]):
            final, text = resource_import.fetch_html("https://source.test/start")
        self.assertEqual(final, "https://source.test/final")
        self.assertEqual(text, "<title>Test</title>")
        headers = second.requests[0][1]["headers"]
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("Proxy-Authorization", headers)


if __name__ == "__main__":
    unittest.main()
