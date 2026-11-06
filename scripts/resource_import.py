"""Untrusted public-page metadata previews and catalog-wide identity matching."""
from __future__ import annotations

from collections import deque
import hashlib
from html.parser import HTMLParser
import http.client
import ipaddress
import json
import queue
import re
import socket
import ssl
import threading
import time
import unicodedata
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

MAX_BYTES = 2 * 1024 * 1024
FETCH_SECONDS = 15
MAX_REDIRECTS = 5


class ImportError(ValueError):
    """A readable failure; no synthetic metadata is returned on fetch failure."""


def _url_parts(url):
    if not isinstance(url, str) or not url or any(ord(c) <= 32 or ord(c) == 127 for c in url):
        raise ImportError("网址不能为空或包含空白、控制字符")
    try:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError()
        if parts.username is not None or parts.password is not None or "\\" in url:
            raise ValueError()
        port = parts.port if parts.port is not None else (443 if parts.scheme == "https" else 80)
        if not 1 <= port <= 65535:
            raise ValueError()
        host = parts.hostname.encode("idna").decode("ascii")
    except (ValueError, UnicodeError) as exc:
        raise ImportError("仅支持无凭证的公网 HTTP(S) 网址") from exc
    return parts, host, port


def _public_addresses(host, port, deadline):
    # getaddrinfo has no timeout API. A daemon resolver never holds up the request.
    results = queue.Queue(maxsize=1)

    def resolve():
        try:
            results.put(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except OSError as exc:
            results.put(exc)

    threading.Thread(target=resolve, daemon=True).start()
    try:
        records = results.get(timeout=max(0.001, deadline - time.monotonic()))
    except queue.Empty as exc:
        raise ImportError("域名解析超时") from exc
    if isinstance(records, Exception):
        raise ImportError("无法解析来源域名") from records
    addresses = []
    for family, socktype, protocol, _, address in records:
        try:
            ip = ipaddress.ip_address(address[0])
        except ValueError as exc:
            raise ImportError("来源地址无效") from exc
        # Reject mixed public/private answers, mapped IPv4, and scoped addresses.
        if (not ip.is_global or ip.is_multicast or getattr(ip, "ipv4_mapped", None)
                or getattr(ip, "sixtofour", None) or getattr(ip, "teredo", None) or "%" in address[0]):
            raise ImportError("禁止访问非公网地址")
        if family not in {socket.AF_INET, socket.AF_INET6}:
            raise ImportError("来源地址类型不支持")
        addresses.append((family, socktype, protocol, address))
    if not addresses:
        raise ImportError("来源域名没有公网地址")
    return addresses


def _connection(host, port, secure, address, deadline):
    connection = (http.client.HTTPSConnection(host, port, context=ssl.create_default_context())
                  if secure else http.client.HTTPConnection(host, port))

    def pinned_socket(_address, timeout=None, source_address=None):
        family, socktype, protocol, sockaddr = address
        sock = socket.socket(family, socktype, protocol)
        try:
            sock.settimeout(max(0.001, deadline - time.monotonic()))
            sock.connect(sockaddr)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ImportError("抓取超时")
            sock.settimeout(remaining)
            return sock
        except BaseException:
            sock.close()
            raise

    # HTTPSConnection still performs certificate verification and SNI for `host`.
    # Only its TCP dialer is replaced: there is no second DNS lookup or proxy.
    connection._create_connection = pinned_socket
    connection.timeout = max(0.001, deadline - time.monotonic())
    return connection


def fetch_html(url):
    """Fetch HTML with pinned public DNS, per-hop checks and one total deadline."""
    deadline = time.monotonic() + FETCH_SECONDS
    current = url
    try:
        for hop in range(MAX_REDIRECTS + 1):
            parts, host, port = _url_parts(current)
            addresses = _public_addresses(host, port, deadline)
            if time.monotonic() >= deadline:
                raise ImportError("抓取超时")
            connection = _connection(host, port, parts.scheme == "https", addresses[0], deadline)
            timer = None
            try:
                connection.connect()
                # Closing a socket alone need not interrupt a blocking read on all OSes.
                active_socket = connection.sock

                def expire():
                    try:
                        active_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    active_socket.close()

                timer = threading.Timer(max(0.001, deadline - time.monotonic()), expire)
                timer.daemon = True
                timer.start()
                path = quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~")
                if parts.query:
                    path += "?" + quote(parts.query, safe="%=&?/:@!$'()*+,;~-._")
                connection.request("GET", path, headers={
                    "User-Agent": "LocalResourceCatalog/1.0", "Accept": "text/html,application/xhtml+xml",
                    "Accept-Encoding": "identity", "Connection": "close",
                })
                response = connection.getresponse()
                if response.status in {301, 302, 303, 307, 308}:
                    location = response.getheader("Location")
                    if not location:
                        raise ImportError("来源重定向缺少目标网址")
                    if hop == MAX_REDIRECTS:
                        raise ImportError("来源重定向次数过多")
                    current = urljoin(current, location)
                    continue
                if not 200 <= response.status < 300:
                    raise ImportError(f"来源返回 HTTP {response.status}")
                content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    raise ImportError("来源不是 HTML 网页；不下载或解析 PDF，请手动填写")
                if response.getheader("Content-Encoding", "identity").lower() not in {"identity", ""}:
                    raise ImportError("来源使用不支持的压缩编码")
                length = response.getheader("Content-Length")
                if length and (not length.isdecimal() or int(length) > MAX_BYTES):
                    raise ImportError("来源网页超过大小限制或长度无效")
                data = bytearray()
                while True:
                    if time.monotonic() >= deadline:
                        raise ImportError("抓取超时")
                    chunk = response.read1(min(65536, MAX_BYTES + 1 - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
                    if len(data) > MAX_BYTES:
                        raise ImportError("来源网页超过大小限制")
                if time.monotonic() >= deadline:
                    raise ImportError("抓取超时")
                if length and len(data) != int(length):
                    raise ImportError("来源响应不完整，未读取完整 HTML")
                charset = response.headers.get_content_charset() or "utf-8"
                try:
                    text = data.decode(charset, errors="replace")
                except LookupError:
                    text = data.decode("utf-8", errors="replace")
                return current, text
            finally:
                if timer is not None:
                    timer.cancel()
                connection.close()
        raise ImportError("来源重定向次数过多")
    except ImportError:
        raise
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise ImportError("抓取来源失败（连接、证书、超时或响应格式错误），可手动补录") from exc


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def _doi(value):
    if not isinstance(value, str):
        return ""
    value = unquote(value).strip()
    value = re.sub(r"^doi:\s*", "", value, flags=re.I)
    if value.lower().startswith(("https://", "http://")):
        try:
            parts = urlsplit(value)
            if parts.hostname not in {"doi.org", "dx.doi.org", "www.doi.org"}:
                return ""
            value = parts.path.lstrip("/")
        except ValueError:
            return ""
    return value.casefold() if re.fullmatch(r"10\.\d{4,9}/[^\s?#]+", value, re.I) else ""


def _arxiv(value):
    if not isinstance(value, str):
        return ""
    value = value.strip()
    doi = _doi(value)
    if doi.startswith("10.48550/arxiv."):
        value = doi[len("10.48550/arxiv."):]
    if value.lower().startswith(("https://", "http://")):
        try:
            parts = urlsplit(value)
            if (parts.hostname or "").lower() not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
                return ""
            match = re.fullmatch(r"/(?:abs|pdf|html)/(.+?)(?:\.pdf)?/?", parts.path)
            if not match:
                return ""
            value = match[1]
        except ValueError:
            return ""
    value = re.sub(r"^arxiv:\s*", "", value, flags=re.I)
    match = re.fullmatch(r"(\d{4}\.\d{4,5}|[a-z][a-z.\-]*/\d{7})(?:v\d+)?", value, re.I)
    return match[1].casefold() if match else ""


def _normalized_url(value):
    try:
        parts, host, port = _url_parts(value)
    except ImportError:
        return ""
    authority = f"[{host}]" if ":" in host else host.lower()
    if port != (443 if parts.scheme == "https" else 80):
        authority += f":{port}"
    # Keep the scheme, path spelling, trailing slash and every query parameter.
    return urlunsplit((parts.scheme, authority, parts.path or "/", parts.query, ""))


def _identities(metadata):
    values = [metadata.get("url"), metadata.get("doi"), metadata.get("arxiv")]
    links = metadata.get("links", {})
    if isinstance(links, dict):
        values.extend(links.values())
    return ({key for value in values if (key := _doi(value))},
            {key for value in values if (key := _arxiv(value))})


def _title_key(value):
    return re.sub(r"[\W_]", "", unicodedata.normalize("NFKC", _text(value)).casefold())


def _source_urls(metadata):
    values = [metadata.get("url")]
    links = metadata.get("links", {})
    if isinstance(links, dict):
        values.extend(links.values())
    attachments = metadata.get("attachments", [])
    if isinstance(attachments, list):
        values.extend(item.get("url") for item in attachments if isinstance(item, dict))
    return {key for value in values if (key := _normalized_url(value))}


def find_duplicates(metadata, entries):
    """Pure, cross-category match; strong identity, exact URL, then exact title."""
    dois, arxivs = _identities(metadata)
    url = _normalized_url(metadata.get("url"))
    sources = _source_urls(metadata)
    title = _title_key(metadata.get("title"))
    matches = []
    for entry in entries:
        other_dois, other_arxivs = _identities(entry)
        if dois & other_dois:
            reason = "DOI 相同"
        elif arxivs & other_arxivs:
            reason = "arXiv 标识相同（忽略版本号）"
        elif (url and url in _source_urls(entry)) or _normalized_url(entry.get("url")) in sources:
            # Match a primary source to known alternatives, not shared code links.
            reason = "来源网址与已有入口相同"
        elif title and title == _title_key(entry.get("title")):
            reason = "标题相同，请核实是否同一资源"
        else:
            continue
        matches.append({"id": entry.get("id", ""), "title": entry.get("title", ""),
                        "metadata_path": str(entry.get("metadata_path", "")), "reason": reason})
    return matches


class _Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta = {}
        self.title = []
        self.jsonld = []
        self.in_title = False
        self.script = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            name = (attrs.get("name") or attrs.get("property") or "").lower()
            content = _text(attrs.get("content"))
            if name and content:
                self.meta.setdefault(name, []).append(content)
        elif tag == "title":
            self.in_title = True
        elif tag == "script" and (attrs.get("type") or "").lower() == "application/ld+json":
            self.script = []

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag == "script" and self.script is not None:
            self.jsonld.append("".join(self.script))
            self.script = None

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        if self.script is not None:
            self.script.append(data)

    def first(self, *names):
        return next((self.meta[name][0] for name in names if self.meta.get(name)), "")


def _article(page):
    accepted = {"Article", "ScholarlyArticle", "TechArticle", "BlogPosting", "NewsArticle"}
    for raw in page.jsonld:
        try:
            payload = json.loads(raw)
        except (ValueError, RecursionError):
            continue
        pending = deque(payload if isinstance(payload, list) else [payload])
        while pending:
            node = pending.popleft()
            if not isinstance(node, dict):
                continue
            kind = node.get("@type", [])
            kinds = [kind] if isinstance(kind, str) else kind if isinstance(kind, list) else []
            if any(isinstance(item, str) and item.rsplit("/", 1)[-1] in accepted for item in kinds):
                return node
            graph = node.get("@graph")
            if isinstance(graph, list):
                pending.extend(graph)
    return {}


def _year(value):
    match = re.match(r"^(\d{4})(?:$|[-/. ])", _text(value))
    return int(match[1]) if match and int(match[1]) > 0 else None


def preview(url, entries):
    final_url, html = fetch_html(url)
    page = _Page()
    try:
        page.feed(html)
        page.close()
    except (ValueError, RecursionError) as exc:
        raise ImportError("来源 HTML 无法解析") from exc
    article = _article(page)
    title = page.first("citation_title", "og:title") or _text(article.get("headline") or article.get("name")) or _text("".join(page.title))
    if not title:
        raise ImportError("网页未提供可识别标题，请手动填写")
    doi = _doi(page.first("citation_doi", "dc.identifier", "dc.identifier.doi")) or _doi(final_url) or _doi(url)
    if not doi:
        identifiers = article.get("identifier", [])
        if not isinstance(identifiers, list):
            identifiers = [identifiers]
        for identifier in identifiers:
            if isinstance(identifier, dict):
                identifier = identifier.get("value") if _text(identifier.get("propertyID")).lower() == "doi" else identifier.get("@id")
            doi = _doi(identifier)
            if doi:
                break
    kind = article.get("@type", [])
    kinds = [kind] if isinstance(kind, str) else kind if isinstance(kind, list) else []
    scholarly = "ScholarlyArticle" in kinds or bool(page.first("citation_title", "citation_journal_title", "citation_conference_title"))
    is_paper = scholarly or bool(doi or _arxiv(final_url) or _arxiv(url))
    resource_type = "paper" if is_paper else "doc" if "TechArticle" in kinds else "blog"
    slug = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()).strip("-")[:80].rstrip("-")
    if not slug:
        slug = "resource-" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:12]
    metadata = {"id": slug, "title": title, "type": resource_type, "url": final_url,
                "status": "unread", "summary": "", "tags": []}
    if doi:
        metadata["doi"] = doi
    authors = list(page.meta.get("citation_author", []))
    if not authors:
        people = article.get("author", [])
        if not isinstance(people, list):
            people = [people]
        for person in people:
            name = _text(person.get("name")) if isinstance(person, dict) else _text(person)
            if name:
                authors.append(name)
    if not authors:
        authors = list(page.meta.get("author", []) or page.meta.get("dc.creator", []))
    if authors:
        metadata["authors"] = list(dict.fromkeys(authors))
    year = _year(page.first("citation_publication_date", "citation_date", "dc.date", "article:published_time")) or _year(article.get("datePublished"))
    if year:
        metadata["year"] = year
    venue = page.first("citation_journal_title", "citation_conference_title")
    if not venue and isinstance(article.get("isPartOf"), dict):
        venue = _text(article["isPartOf"].get("name"))
    if venue:
        metadata["venue"] = venue
    description = page.first("citation_abstract", "og:description", "description") or _text(article.get("description"))
    warnings = ["仅抓取并解析返回的 HTML 页面元数据；未下载 PDF，未核实全文内容或代码链接。",
                "作者、年份等仅按来源声明提取，未独立核实；撤稿状态、最新版本及版本差异未核实。",
                "来源内容不可信；请核对标题、类型与 ID，并自行填写中文摘要和标签。"]
    if not is_paper and "TechArticle" not in kinds:
        warnings.append("来源未明确提供资源分类，当前类型为待确认的博客建议。")
    return {"metadata": metadata, "description": description, "source_url": final_url,
            "warnings": warnings, "duplicates": find_duplicates(metadata, entries)}
