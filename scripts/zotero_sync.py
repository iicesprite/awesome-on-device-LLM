#!/usr/bin/env python3
"""准备经认证 MCP 导入，验证回执，并维护当前账户的本机 Zotero 索引。

本脚本只读 Zotero；写入仅由已认证 MCP 执行生成的 import.js。
"""
import argparse
from copy import deepcopy
from http.client import HTTPException
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import unicodedata
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse
from urllib.request import (HTTPRedirectHandler, ProxyHandler, Request, build_opener,
                            url2pathname)

import catalog

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_API_BASE = "http://127.0.0.1:23119"
LOCAL = ROOT / ".local" / "zotero"
KEY = re.compile(r"[A-Z0-9]{8}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def api_base(value):
    require(isinstance(value, str) and not any(c.isspace() for c in value), "api_base 格式无效")
    parsed = urlparse(value)
    try:
        host = parsed.hostname
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        loopback = False
    require(parsed.scheme in ("http", "https") and loopback and
            not parsed.username and not parsed.password and not parsed.query and
            not parsed.fragment and parsed.path in ("", "/"),
            "api_base 必须是无凭据、查询或路径的回环 HTTP(S) 源地址（不含 /api）")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("api_base 端口无效") from exc
    return value.rstrip("/")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("本机 Zotero API 不应重定向")


def get(route, config):
    """所有常规请求都绑定数值用户 ID；仅 configure 允许发现 users/0。"""
    url = f"{config['api_base']}/api/users/{config['user_id']}" + route
    request = Request(url, headers={"Zotero-API-Version": "3"})
    with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=30) as response:
        raw = response.read().decode("utf-8")
    return raw.strip() if route.endswith("/file/view/url") else json.loads(raw)


def library(config):
    return {"type": "user", "id": config["user_id"]}


def verify_account(config):
    require(type(config.get("user_id")) is int and config["user_id"] > 0,
            "需要已登录的 Zotero 个人账户及正整数 user_id")
    require(api_base(config["api_base"]) == config["api_base"], "api_base 格式无效")
    rows = get("/items?format=json&limit=1", config)
    require(isinstance(rows, list), "Zotero 账户验证响应格式无效")
    for row in rows:
        identity = row.get("library", {})
        require(identity.get("type") == "user" and identity.get("id") == config["user_id"],
                "Zotero 当前账户与配置不符")
    # Local API rejects a numeric ID other than the logged-in user's, even
    # when this list is empty. Never fall back to users/0 here.
    return config


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(text)
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def save_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def configure(api_base_value=None, user_id=None):
    previous = json.loads((LOCAL / "config.json").read_text(encoding="utf-8")) if (LOCAL / "config.json").exists() else {}
    base = api_base(api_base_value or previous.get("api_base", DEFAULT_API_BASE))
    if user_id is None:
        rows = get("/items?format=json&limit=1", {"api_base": base, "user_id": 0})
        require(isinstance(rows, list) and rows,
                "空库无法自动发现账户；请登录 Zotero 并以 configure --user-id 正整数 显式配置")
        identity = rows[0].get("library", {})
        require(identity.get("type") == "user", "仅支持 Zotero 个人库")
        user_id = identity.get("id")
    config = verify_account({"api_base": base, "user_id": user_id})
    save_json(LOCAL / "config.json", config)
    if previous.get("user_id") != user_id and (LOCAL / "index.md").exists():
        # Do not leave the previous account's links presented as the active index.
        atomic_text(LOCAL / "index.md", "# 本机 Zotero 索引\n\n账户已切换；请运行 zotero_sync.py index 生成当前账户索引。\n")
    print(f"已配置个人库 user:{user_id}（要求 Zotero 保持登录）")
    return config


def load_config(auto=False):
    path = LOCAL / "config.json"
    if not path.exists():
        require(auto, "请先运行 zotero_sync.py configure")
        return configure()
    return verify_account(json.loads(path.read_text(encoding="utf-8")))


def load_bindings(config):
    verify_account(config)
    path = LOCAL / "bindings.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"version": 1, "libraries": {}}
    require(data.get("version") == 1 and isinstance(data.get("libraries"), dict), "本机 bindings.json 格式无效")
    scope = data["libraries"].get(f"user:{config['user_id']}")
    if scope is not None:
        require(scope.get("library") == library(config) and isinstance(scope.get("entries"), dict), "本机绑定账户无效")
    return data, scope["entries"] if scope else {}


def inventory(config):
    result = []
    while True:
        page = get(f"/items/top?format=json&limit=100&start={len(result)}", config)
        result.extend(page)
        if len(page) < 100:
            return [x for x in result if x["data"].get("itemType") not in ("attachment", "note", "annotation")]


def valid_key(key):
    require(isinstance(key, str) and KEY.fullmatch(key), "Zotero key 格式无效")
    return key


def readable(path):
    try:
        with path.open("rb") as stream:
            return bool(stream.read(1))
    except OSError:
        return False


def attachment_available(key, config):
    try:
        uri = get(f"/items/{valid_key(key)}/file/view/url", config)
        parsed = urlparse(uri)
        return parsed.scheme == "file" and parsed.netloc in ("", "localhost") and readable(Path(url2pathname(parsed.path)))
    except HTTPError as exc:
        if exc.code in (400, 404):
            return False
        raise


def title_key(value):
    return re.sub(r"[\W_]", "", unicodedata.normalize("NFKC", value).casefold())


def identities(data):
    result = set()
    identity_links = {"arxiv", "doi", "preprint", "publication", "publisher",
                      "published", "fulltext", "html", "pdf", "paper"}
    links = [value for name, value in data.get("links", {}).items() if name.casefold() in identity_links]
    values = [data.get("doi", ""), data.get("DOI", ""), data.get("url", ""), *links, data.get("extra", "")]
    for value in values:
        for identity in re.findall(r"10\.\d{4,9}/[^\s<>]+", value, re.I):
            result.add(identity.rstrip(".,").casefold())
        for value in re.findall(r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv:)\s*((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7}))", value, re.I):
            result.add("arxiv:" + value.casefold())
    return result


def same_identity(entry, data):
    left, right = identities(entry), identities(data)
    if left & right:
        return True
    # A shared identity wins first; conflicting explicit identities cannot be
    # rescued by a same-title match.
    return not (left and right) and title_key(entry["title"]) == title_key(data.get("title", ""))


def match(entry, items, binding=None):
    key = (binding or {}).get("item_key")
    if key:
        found = [x for x in items if x["key"] == valid_key(key)]
        require(len(found) == 1 and same_identity(entry, found[0]["data"]),
                f"{entry['id']}: 本机绑定失效或身份不符，停止以避免重复导入")
        return found[0]
    found = [x for x in items if identities(entry) & identities(x["data"])]
    if not found:
        found = [x for x in items if same_identity(entry, x["data"])]
    require(len(found) <= 1, f"{entry['id']}: 多个 Zotero 候选，请先确认")
    return found[0] if found else None


def public_url(value):
    parsed = urlparse(value)
    require(parsed.scheme in ("http", "https") and parsed.hostname and
            not parsed.username and not parsed.password, "下载地址必须是公开 HTTP(S) URL")
    for address in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM):
        require(ipaddress.ip_address(address[4][0]).is_global, "不允许从非公网地址下载 PDF")
    return value


class PublicRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_pdf(url, path):
    """只下载声明的确切来源，不推测或改写版本 URL。"""
    public_url(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with build_opener(ProxyHandler({}), PublicRedirect()).open(url, timeout=30) as response:
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
                name = stream.name
                magic = response.read(5)
                require(magic == b"%PDF-", "下载内容不是 PDF（缺少 %PDF- 文件头）")
                stream.write(magic)
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def paper_entries():
    return [e for e in catalog.load_catalog(ROOT)[0] if e["type"] == "paper"]


def prepare(download=False):
    config = load_config(auto=True)
    _, bindings = load_bindings(config)
    items = inventory(config)
    jobs = []
    for entry in paper_entries():
        binding = bindings.get(entry["id"], {})
        zotero = match(entry, items, binding)
        children = get(f"/items/{zotero['key']}/children?format=json", config) if zotero else []
        files, missing = [], []
        used_keys = {}
        attachments = [a for a in entry.get("attachments", []) if a["path"].lower().endswith(".pdf")]
        for attachment in attachments:
            relative = attachment["path"]
            path = ROOT / Path(entry["metadata_path"]).parent / relative
            source_url = attachment.get("downloaded_from", attachment["url"])
            version = attachment.get("downloaded_version", attachment.get("version"))
            saved = binding.get("attachments", {}).get(relative, {})
            key = saved.get("key")
            candidates = []
            if key:
                candidates = [x for x in children if x["key"] == valid_key(key)]
                require(len(candidates) == 1 and candidates[0]["data"].get("itemType") == "attachment" and
                        candidates[0]["data"].get("parentItem") == zotero["key"] and
                        candidates[0]["data"].get("contentType") == "application/pdf",
                        f"{entry['id']}: 本机附件绑定失效 {relative}")
            else:
                unique_source = sum(a.get("downloaded_from", a["url"]) == source_url for a in attachments) == 1
                # URL alone is not a version identity. Require an exact declared
                # version title or filename too.
                candidates = [x for x in children if x["data"].get("itemType") == "attachment" and
                              x["data"].get("parentItem") == zotero["key"] and
                              x["data"].get("contentType") == "application/pdf" and
                              x["data"].get("url") == source_url and
                              ((version and x["data"].get("title") == version) or
                               (unique_source and x["data"].get("filename") == path.name))]
            require(len(candidates) <= 1, f"{entry['id']}: PDF 候选不唯一 {relative}")
            available_key = candidates[0]["key"] if candidates and attachment_available(candidates[0]["key"], config) else None
            if available_key:
                signature = (source_url, version, bool(key))
                require(available_key not in used_keys or
                        (key == available_key and used_keys[available_key] == signature),
                        f"{entry['id']}: 不同版本不能自动合并到同一 Zotero PDF")
                used_keys[available_key] = signature
            reason = "Zotero 附件文件不可读，且仓库缺少可读文件" if candidates else "仓库缺少可读文件，未找到可确认版本的 Zotero 附件"
            if not available_key and not readable(path) and download:
                try:
                    # Historical missing versions may have an unversioned source;
                    # downloading that URL must not silently relabel today's PDF.
                    historical_missing = saved.get("legacy_metadata", {}).get("local_status") == "missing-in-zotero"
                    require(not historical_missing or version,
                            "历史缺失版本不能从可能变化的来源自动重建；请补充明确版本文件")
                    require(not attachment.get("source_zotero_attachment_key") or version,
                            "历史来源未注明确切版本，不能将当前下载冒充旧文件")
                    download_pdf(source_url, path)
                except (ValueError, OSError, HTTPException) as exc:
                    reason += f"；下载失败：{exc}"
            if available_key or readable(path):
                files.append({"relative_path": relative, "file": str(path.resolve()) if readable(path) else None,
                              "url": source_url, "title": version or path.stem,
                              "key": available_key})
            else:
                missing.append({"relative_path": relative, "reason": reason})
        jobs.append({"id": entry["id"], "title": entry["title"], "url": entry["url"],
                     "doi": entry.get("doi", ""), "links": entry.get("links", {}), "year": entry.get("year"),
                     "venue": entry.get("venue", ""), "authors": entry.get("authors", []),
                     "tags": entry["tags"], "key": zotero["key"] if zotero else None,
                     "files": files, "missing_files": missing,
                     "before_tags": zotero["data"].get("tags", []) if zotero else []})
    verify_account(config)
    payload = {"library": library(config), "collection": "awesome-on-device-LLM", "papers": jobs}
    save_json(LOCAL / "plan.json", payload)
    template = (ROOT / "scripts" / "zotero_import.js").read_text(encoding="utf-8")
    atomic_text(LOCAL / "import.js", "const plan = " + json.dumps(payload, ensure_ascii=False) + ";\n" + template)
    print(f"准备 {len(jobs)} 篇：复用 {sum(bool(x['key']) for x in jobs)}，新建 {sum(not x['key'] for x in jobs)}")
    print(f"可读 PDF {sum(len(x['files']) for x in jobs)}；缺失 PDF {sum(len(x['missing_files']) for x in jobs)}；仅元数据 {sum(not x['files'] for x in jobs)} 篇")
    for job in jobs:
        for missing in job["missing_files"]:
            print(f"  {job['id']} {missing['relative_path']}：{missing['reason']}")
    print(f"Zotero 尚未改写；通过已认证 MCP 的 zotero_script 执行：{LOCAL / 'import.js'}")
    return payload


def primary_pdf(entry, attachments):
    available = {path: saved["key"] for path, saved in attachments.items() if saved.get("status") == "available"}
    notes = ROOT / entry.get("notes_path", "")
    if notes.is_file():
        paths = re.findall(r"\]\((?:\./)?(resources/[^)#?]+\.pdf)(?:[#?][^)]*)?\)", notes.read_text(encoding="utf-8"), re.I)
        if paths and unquote(paths[0]) in available:
            return available[unquote(paths[0])]
    return next(iter(available.values()), None)


def checked_attachment(key, parent, config):
    child = get(f"/items/{valid_key(key)}?format=json", config)["data"]
    require(child.get("itemType") == "attachment" and child.get("parentItem") == parent and
            child.get("contentType") == "application/pdf" and not child.get("deleted"),
            "PDF 父条目或类型不符")
    return attachment_available(key, config)


def render_personal_index(config, entries, bindings):
    lines = ["# 本机 Zotero 索引", "", f"个人账户：user:{config['user_id']}。仅供本机使用；不提交到 Git。", "",
             "| 论文 | 标签 | Zotero 条目 | 阅读 PDF | 附件状态 | 笔记 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for entry in entries:
        binding = bindings.get(entry["id"], {})
        key = binding.get("item_key")
        current = {}
        if key:
            zotero = get(f"/items/{valid_key(key)}?format=json", config)
            require(same_identity(entry, zotero["data"]) and not zotero["data"].get("deleted"),
                    f"{entry['id']}: 本机绑定失效，拒绝发布旧链接")
        for attachment in entry.get("attachments", []):
            path = attachment["path"]
            if not path.lower().endswith(".pdf"):
                continue
            saved = binding.get("attachments", {}).get(path, {})
            available = False
            if key and saved.get("key"):
                try:
                    available = checked_attachment(saved["key"], key, config)
                except HTTPError as exc:
                    if exc.code not in (400, 404):
                        raise
                except ValueError:
                    pass
            current[path] = {"key": saved.get("key"), "status": "available" if available else "missing"}
        primary = primary_pdf(entry, current)
        item_link = catalog.link("条目", f"zotero://select/library/items/{key}") if key else "未绑定"
        pdf_link = catalog.link("阅读 PDF", f"zotero://open-pdf/library/items/{primary}") if primary else "无可读 PDF"
        status = "；".join(f"{path}：{'可读' if state['status'] == 'available' else '缺失'}" for path, state in current.items()) or "无 PDF 附件"
        notes = catalog.link("笔记", Path(os.path.relpath(ROOT / entry["notes_path"], LOCAL)).as_posix()) if entry.get("notes_path") else "—"
        cells = [catalog.link(entry["title"], entry["url"]), catalog.md("、".join(sorted(entry["tags"]))), item_link,
                 pdf_link, catalog.md(status), notes]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def index():
    config = load_config()
    _, bindings = load_bindings(config)
    text = render_personal_index(config, paper_entries(), bindings)
    verify_account(config)
    atomic_text(LOCAL / "index.md", text)
    print(f"已生成当前账户本机索引：{LOCAL / 'index.md'}")


def unique_rows(rows, field, message):
    require(isinstance(rows, list) and all(isinstance(row, dict) and isinstance(row.get(field), str) for row in rows), message)
    require(len({row[field] for row in rows}) == len(rows), message)
    return {row[field]: row for row in rows}


def record(receipt):
    config = load_config()
    plan = json.loads((LOCAL / "plan.json").read_text(encoding="utf-8"))
    rows = json.loads(receipt.read_text(encoding="utf-8"))
    require(plan.get("library") == library(config) and isinstance(rows, dict) and
            rows.get("library") == library(config), "计划、回执与当前账户必须一致；请重新 prepare")
    entries = {e["id"]: e for e in paper_entries()}
    planned = unique_rows(plan.get("papers"), "id", "计划论文 ID 缺失或重复")
    received = unique_rows(rows.get("items"), "id", "回执论文 ID 缺失或重复")
    require(set(planned) == set(entries) == set(received), "计划与回执需包含当前全部论文")
    container, old = load_bindings(config)
    bindings = deepcopy(old)
    item_keys, attachment_keys = set(), {}
    for identity, entry in entries.items():
        job, row = planned[identity], received[identity]
        require(job.get("title") == entry["title"] and job.get("url") == entry["url"] and
                job.get("doi", "") == entry.get("doi", "") and
                set(job.get("tags", [])) == set(entry["tags"]), f"{identity}: 计划已过期，请重新 prepare")
        key = valid_key(row.get("key"))
        require(key not in item_keys, "回执条目 key 重复")
        item_keys.add(key)
        require(not job.get("key") or job["key"] == key, f"{identity}: 回执改变既有条目绑定")
        zotero = get(f"/items/{key}?format=json", config)["data"]
        require(zotero.get("itemType") not in ("attachment", "note", "annotation") and not zotero.get("deleted") and
                same_identity(entry, zotero), f"{identity}: 回执论文身份不符")
        require(set(entry["tags"]).issubset({tag["tag"] for tag in zotero.get("tags", [])}), f"{identity}: Zotero 缺少规范标签")
        expected = unique_rows(job.get("files"), "relative_path", "计划 PDF 路径重复或缺失")
        missing = unique_rows(job.get("missing_files"), "relative_path", "计划缺失 PDF 路径重复或缺失")
        files = unique_rows(row.get("files"), "relative_path", "回执 PDF 路径重复或缺失")
        paths = {a["path"] for a in entry.get("attachments", []) if a["path"].lower().endswith(".pdf")}
        require(set(expected).isdisjoint(missing) and set(expected) | set(missing) == paths and set(files) == set(expected),
                f"{identity}: 回执附件与计划不一致，或计划附件已过期")
        declared = {a["path"]: a for a in entry.get("attachments", []) if a["path"] in paths}
        for path, value in expected.items():
            version = declared[path].get("downloaded_version", declared[path].get("version"))
            require(value.get("url") == declared[path].get("downloaded_from", declared[path]["url"]) and
                    value.get("title") == (version or Path(path).stem),
                    f"{identity}: PDF 来源或版本已改变，请重新 prepare")
        saved = deepcopy(old.get(identity, {}))
        saved["item_key"] = key
        saved["attachments"] = {}
        for path, value in missing.items():
            previous = old.get(identity, {}).get("attachments", {}).get(path, {})
            saved["attachments"][path] = {**previous, "status": "missing", "reason": value["reason"]}
        for path, value in files.items():
            attachment_key = valid_key(value.get("key"))
            version = declared[path].get("downloaded_version", declared[path].get("version"))
            signature = (key, expected[path]["url"], version, expected[path].get("key"))
            require(attachment_key not in attachment_keys or
                    (expected[path].get("key") == attachment_key and attachment_keys[attachment_key] == signature),
                    "回执附件 key 重复，不能合并不同版本")
            attachment_keys[attachment_key] = signature
            require(not expected[path].get("key") or expected[path]["key"] == attachment_key, f"{identity}: 回执改变既有 PDF 绑定")
            require(checked_attachment(attachment_key, key, config), f"{identity}: Zotero PDF 本机文件不可读")
            previous = old.get(identity, {}).get("attachments", {}).get(path, {})
            saved["attachments"][path] = {**previous, "key": attachment_key, "status": "available",
                                          "source_url": expected[path]["url"]}
            saved["attachments"][path].pop("reason", None)
        saved.pop("reading_attachment_key", None)
        primary = primary_pdf(entry, saved["attachments"])
        if primary:
            saved["reading_attachment_key"] = primary
        bindings[identity] = saved
    # Generate/validate all links before committing any local state. Never touch
    # public metadata, and retain other accounts' bindings byte-for-byte in value.
    text = render_personal_index(config, list(entries.values()), bindings)
    verify_account(config)
    container["libraries"][f"user:{config['user_id']}"] = {"library": library(config), "entries": bindings}
    save_json(LOCAL / "bindings.json", container)
    atomic_text(LOCAL / "index.md", text)
    print(f"已验证并保存 {len(entries)} 篇本机绑定；本机索引：{LOCAL / 'index.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    configure_parser = sub.add_parser("configure", help="配置回环 API 和已登录个人账户")
    configure_parser.add_argument("--api-base")
    configure_parser.add_argument("--user-id", type=int)
    prepare_parser = sub.add_parser("prepare", help="生成 MCP 导入计划；默认不下载")
    prepare_parser.add_argument("--download", action="store_true", help="尝试下载声明的公网 PDF；逐文件报告失败")
    record_parser = sub.add_parser("record", help="验证回执并保存本机绑定和索引")
    record_parser.add_argument("receipt", type=Path)
    sub.add_parser("index", help="复查文件可读性并生成当前账户本机索引")
    args = parser.parse_args()
    try:
        if args.command == "configure":
            configure(args.api_base, args.user_id)
        elif args.command == "prepare":
            prepare(args.download)
        elif args.command == "record":
            record(args.receipt)
        else:
            index()
    except (ValueError, OSError, catalog.CatalogError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
