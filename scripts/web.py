#!/usr/bin/env python3
"""Serve the local resource workspace; run with uv run --locked python scripts/web.py."""
from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
import sys
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

import catalog
import resource_import
from web_store import Store, StoreError

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
MAX_REQUEST = 1024 * 1024
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/vendor/mermaid.min.js": ("vendor/mermaid.min.js", "text/javascript; charset=utf-8"),
    "/vendor/katex/katex.min.js": ("vendor/katex/katex.min.js", "text/javascript; charset=utf-8"),
    "/vendor/katex/katex.min.css": ("vendor/katex/katex.min.css", "text/css; charset=utf-8"),
}
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self'; font-src 'self'; connect-src 'self'; "
    "object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


class WorkspaceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root, port=8765):
        self.store = Store(Path(root).resolve())
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), Handler)


class Handler(BaseHTTPRequestHandler):
    server: WorkspaceServer
    server_version = "ResourceWorkspace/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def send_headers(self, status, content_type, length, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", CSP)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def json_response(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False, default=str, allow_nan=False).encode("utf-8")
        self.send_headers(status, "application/json; charset=utf-8", len(raw))
        self.wfile.write(raw)

    def check_request(self, mutation=False):
        port = self.server.server_port
        host = self.headers.get("Host", "")
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            raise StoreError("仅允许通过本机地址访问服务", status=403)
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{host}":
            raise StoreError("不接受跨站请求", status=403)
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise StoreError("不接受跨站请求", status=403)
        if mutation and not hmac.compare_digest(
            self.headers.get("X-Catalog-Token", "").encode("utf-8"), self.server.token.encode("ascii")
        ):
            raise StoreError("写入令牌已失效，请刷新页面后重试", status=403)

    def read_json(self):
        if self.headers.get_content_type() != "application/json":
            raise StoreError("请求必须使用 application/json", status=415)
        if self.headers.get("Transfer-Encoding"):
            raise StoreError("不支持分块请求", status=400)
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise StoreError("无效的请求长度") from exc
        if size < 1 or size > MAX_REQUEST:
            raise StoreError("请求为空或超过 1 MiB", status=413)
        raw = self.rfile.read(size)
        if len(raw) != size:
            raise StoreError("请求内容不完整")
        try:
            value = json.loads(raw, parse_constant=self.reject_constant)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise StoreError("请求不是有效的 JSON") from exc
        if not isinstance(value, dict):
            raise StoreError("请求内容必须是 JSON 对象")
        return value

    @staticmethod
    def reject_constant(value):
        raise ValueError(f"无效数字：{value}")

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def do_PUT(self):
        self.dispatch("PUT")

    def dispatch(self, method):
        try:
            self.check_request(mutation=method != "GET")
            parts = urlsplit(self.path)
            path = parts.path
            if method == "GET":
                self.get(path, parse_qs(parts.query, keep_blank_values=True))
            else:
                self.mutate(method, path, self.read_json())
        except StoreError as exc:
            payload = {"error": str(exc)}
            if getattr(exc, "duplicates", None):
                payload["duplicates"] = exc.duplicates
            self.json_response(payload, exc.status)
        except resource_import.ImportError as exc:
            self.json_response({"error": str(exc)}, 502)
        except catalog.CatalogError as exc:
            self.json_response({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, UnicodeError) as exc:
            self.log_error("File/network error: %s", exc)
            self.json_response({"error": "文件或网络操作失败，请查看服务终端；已编辑的内容仍保留在表单中。"}, 500)

    def get(self, path, query):
        if path == "/api/catalog":
            result = self.server.store.list_resources(
                query=query.get("q", [""])[0], tags=query.get("tag", []),
                kind=query.get("type", [None])[0] or None,
                status=query.get("status", [None])[0] or None,
            )
            self.json_response({**result, "token": self.server.token})
            return
        match = re.fullmatch(r"/api/resources/([a-z0-9]+(?:-[a-z0-9]+)*)", path)
        if match:
            self.json_response(self.server.store.detail(match[1]))
            return
        if path in STATIC:
            filename, content_type = STATIC[path]
            self.send_file(WEB_ROOT / filename, content_type)
            return
        font = re.fullmatch(r"/vendor/katex/fonts/(KaTeX_[A-Za-z0-9_-]+\.(woff2|woff|ttf))", path)
        if font:
            content_type = {"woff2": "font/woff2", "woff": "font/woff", "ttf": "font/ttf"}[font[2]]
            self.send_file(WEB_ROOT / "vendor" / "katex" / "fonts" / font[1], content_type)
            return
        match = re.fullmatch(r"/files/([a-z0-9]+(?:-[a-z0-9]+)*)/(.+)", path)
        if match:
            asset = self.server.store.asset(match[1], unquote(match[2]))
            mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
            inline = mime in {"image/png", "image/jpeg", "image/gif", "image/webp", "image/avif", "application/pdf"}
            extra = {"Content-Disposition": "inline" if inline else "attachment"}
            # Even a mislabeled attachment cannot execute scripts when opened directly.
            self.send_file(asset, mime if inline else "application/octet-stream", extra, sandbox=True)
            return
        raise StoreError("未找到页面或资源", status=404)

    def send_file(self, path, content_type, extra=None, sandbox=False):
        if not path.is_file():
            raise StoreError("本地文件不存在；附件记录不代表已下载", status=404)
        with path.open("rb") as stream:
            size = path.stat().st_size
            headers = dict(extra or {})
            if sandbox:
                headers["Content-Security-Policy"] = "sandbox; default-src 'none'; style-src 'unsafe-inline'"
            self.send_headers(200, content_type, size, headers)
            while chunk := stream.read(64 * 1024):
                self.wfile.write(chunk)

    def mutate(self, method, path, body):
        if method == "POST" and path == "/api/import/preview":
            url = body.get("url")
            if not isinstance(url, str) or not url.strip():
                raise StoreError("请填写完整的 HTTP(S) 网址")
            entries = self.server.store.list_resources()["entries"]
            self.json_response(resource_import.preview(url.strip(), entries))
            return
        if method == "POST" and path == "/api/resources":
            self.json_response(self.server.store.create(body.get("metadata")), 201)
            return
        match = re.fullmatch(r"/api/resources/([a-z0-9]+(?:-[a-z0-9]+)*)", path)
        if method == "PUT" and match:
            self.json_response(self.server.store.update(
                match[1], body.get("revision"), body.get("updates"), body.get("remove_fields", []),
            ))
            return
        raise StoreError("未找到操作入口", status=404)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=catalog.DEFAULT_ROOT, help="资源仓库路径")
    parser.add_argument("--port", type=int, default=8765, help="本机端口（默认 8765）")
    parser.add_argument("--open", action="store_true", help="启动后打开默认浏览器")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("端口必须在 0..65535 之间")
    try:
        with WorkspaceServer(args.root, args.port) as server:
            address = f"http://127.0.0.1:{server.server_port}"
            print(f"资源工作台：{address}\n仓库：{args.root.resolve()}\n仅监听本机；Ctrl+C 停止。", flush=True)
            if args.open:
                webbrowser.open(address)
            server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except (OSError, catalog.CatalogError, StoreError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
