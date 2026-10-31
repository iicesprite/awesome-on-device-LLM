#!/usr/bin/env python3
"""Validate resource metadata, build indexes, and search the local catalog."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import quote, urlsplit

if __name__ == "__main__":
    # Keep Chinese CLI output readable in Windows terminals and redirected output.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import yaml
except ModuleNotFoundError:
    raise SystemExit("缺少 PyYAML。请在仓库根目录执行 uv sync，或使用 uv run python scripts/catalog.py 运行。")


CATEGORIES = {"papers": "paper", "blogs": "blog", "docs": "doc"}
TYPE_LABELS = {"paper": "论文", "blog": "博客", "doc": "文档"}
STATUS_LABELS = {"unread": "未读", "reading": "在读", "read": "已读"}
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
TAG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*/[a-z0-9]+(?:-[a-z0-9]+)*\Z")
DEFAULT_ROOT = Path(__file__).resolve().parent.parent


class CatalogError(ValueError):
    """An actionable catalog validation error."""


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently keeping the last value."""

    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise CatalogError("YAML 字段名必须是字符串")
            if key in mapping:
                raise CatalogError(f"重复的 YAML 字段：{key}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def require(condition, message):
    if not condition:
        raise CatalogError(message)


def is_text(value):
    return isinstance(value, str) and bool(value.strip())


def is_url(value):
    if not is_text(value) or any(c.isspace() for c in value):
        return False
    try:
        parts = urlsplit(value)
        return parts.scheme in {"https", "http"} and bool(parts.hostname)
    except ValueError:
        return False


def read_yaml(path):
    try:
        result = yaml.load(path.read_text(encoding="utf-8-sig"), Loader=UniqueLoader)
        require(isinstance(result, dict), "文件内容必须是 YAML 映射")
        return result
    except (OSError, UnicodeError, yaml.YAMLError, CatalogError) as exc:
        raise CatalogError(f"{path}: {exc}") from exc


def load_tags(root):
    vocabulary = read_yaml(root / "tags.yaml").get("tags")
    require(isinstance(vocabulary, dict) and vocabulary, "tags.yaml: tags 必须是非空映射")
    aliases = {}
    for tag, info in vocabulary.items():
        require(bool(TAG.fullmatch(tag)), f"tags.yaml: 非法标签 {tag!r}，应为 dimension/value")
        require(isinstance(info, dict) and is_text(info.get("label")), f"{tag}: 缺少 label")
        names = info.get("aliases", [])
        require(isinstance(names, list) and all(is_text(x) for x in names), f"{tag}: aliases 必须是字符串列表")
        for name in [tag, info["label"], *names]:
            key = name.casefold()
            require(key not in aliases or aliases[key] == tag, f"tags.yaml: 标签别名冲突 {name!r}")
            aliases[key] = tag
    return vocabulary, aliases


def validate_entry(data, folder, category, vocabulary):
    require("zotero" not in data, "公开元数据禁止 zotero 绑定；请使用 .local/zotero/bindings.json")
    for key in ("id", "title", "type", "url", "status", "summary", "tags"):
        require(key in data, f"缺少字段 {key}")
    entry_id = data["id"]
    require(is_text(entry_id) and bool(SLUG.fullmatch(entry_id)), "id 应为小写英文、数字和连字符")
    require(entry_id == folder.name, f"id {entry_id!r} 与目录名 {folder.name!r} 不一致")
    require(is_text(data["title"]), "title 不能为空")
    require(data["type"] == CATEGORIES[category], f"{category}/ 中的 type 必须为 {CATEGORIES[category]}")
    require(is_url(data["url"]), "url 必须是完整的 HTTP(S) 网址，且不能包含空白字符")
    require(isinstance(data["status"], str) and data["status"] in STATUS_LABELS, "status 必须为 unread / reading / read")
    require(isinstance(data["summary"], str), "summary 必须是字符串，可以为空")
    tags = data["tags"]
    require(isinstance(tags, list) and all(is_text(t) for t in tags), "tags 必须是字符串列表")
    require(len(set(tags)) == len(tags), "tags 中有重复标签")
    for tag in tags:
        require(tag in vocabulary, f"未知标签 {tag!r}；请使用规范标签，或先在 tags.yaml 中添加")
    links = data.get("links", {})
    require(isinstance(links, dict), "links 必须是名称到网址的映射")
    for name, url in links.items():
        require(is_text(name) and is_url(url), f"links.{name} 必须是完整的 HTTP(S) 网址")
    attachments = data.get("attachments", [])
    require(isinstance(attachments, list), "attachments 必须是列表")
    paths = set()
    for item in attachments:
        require(isinstance(item, dict) and is_url(item.get("url")), "每个 attachment 必须有完整的 url")
        for key in ("zotero_item_key", "zotero_attachment_key", "local_status", "downloaded_on"):
            require(key not in item, f"公开附件禁止 {key}；请使用 .local/zotero/bindings.json")
        path = item.get("path")
        require(is_text(path) and "\\" not in path and ":" not in path, "附件 path 应使用相对路径和正斜杠")
        parts = PurePosixPath(path).parts
        require(len(parts) >= 2 and parts[0] == "resources" and ".." not in parts and not path.endswith("/"), "附件 path 必须指向当前条目的 resources/ 内的文件")
        require(path not in paths, f"附件 path 重复：{path}")
        paths.add(path)
    if "authors" in data:
        require(isinstance(data["authors"], list) and all(is_text(a) for a in data["authors"]), "authors 必须是字符串列表")
    if "year" in data:
        require(type(data["year"]) is int and 1 <= data["year"] <= 9999, "year 必须是四位以内的正整数")
    for key in ("venue", "doi"):
        if key in data:
            require(is_text(data[key]), f"{key} 必须是非空字符串；不适用时省略")


def load_catalog(root):
    vocabulary, aliases = load_tags(root)
    entries = []
    ids = set()
    for category in CATEGORIES:
        directory = root / category
        require(directory.is_dir(), f"缺少分类目录：{directory}")
        for folder in sorted(directory.iterdir()):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            path = folder / "meta.yaml"
            data = read_yaml(path)
            try:
                validate_entry(data, folder, category, vocabulary)
                require(data["id"] not in ids, f"跨目录重复 id：{data['id']}")
            except CatalogError as exc:
                raise CatalogError(f"{path}: {exc}") from exc
            ids.add(data["id"])
            entry = dict(data)
            entry["metadata_path"] = path.relative_to(root).as_posix()
            notes = folder / "notes.md"
            entry["notes_path"] = notes.relative_to(root).as_posix() if notes.is_file() else ""
            entries.append(entry)
    return sorted(entries, key=lambda e: (e["type"], e["id"])), vocabulary, aliases


def md(value):
    value = html.escape(str(value), quote=False).replace("\r", " ").replace("\n", " ")
    for char, entity in (("\\", "&#92;"), ("|", "&#124;"), ("[", "&#91;"), ("]", "&#93;"), ("`", "&#96;"), ("*", "&#42;"), ("_", "&#95;")):
        value = value.replace(char, entity)
    return value


def link(label, url):
    return f"[{md(label)}]({quote(url, safe='/:%?=&#+@~.-')})"


def table(entries):
    lines = ["| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for entry in entries:
        tags = "、".join(sorted(entry["tags"])) or "—"
        notes = link("笔记", entry["notes_path"]) if entry["notes_path"] else "—"
        cells = [link(entry["title"], entry["url"]), TYPE_LABELS[entry["type"]], md(entry["summary"]), md(tags), STATUS_LABELS[entry["status"]], link(entry["id"], entry["metadata_path"]), notes]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_index(entries, vocabulary):
    lines = ["# 资源索引", "", "<!-- 自动生成；请修改各条目的 meta.yaml，然后运行 uv run python scripts/catalog.py build。 -->", "", f"共 **{len(entries)}** 条资源。按类型浏览，或在下方按标签交叉查阅。", "", "[项目说明](README.md) · [CSV 索引](catalog.csv) · [标签词表](tags.yaml)", ""]
    if not entries:
        lines += ["尚未收录资源。按 README 添加第一条记录后，重新生成索引。", ""]
        return "\n".join(lines)
    for category, kind in CATEGORIES.items():
        subset = [e for e in entries if e["type"] == kind]
        lines += [f"## {category}（{len(subset)}）", ""]
        lines += table(subset) if subset else ["暂无条目。"]
        lines.append("")
    used_tags = sorted({tag for e in entries for tag in e["tags"]})
    if used_tags:
        lines += ["## 按标签浏览", ""]
    for tag in used_tags:
        subset = [e for e in entries if tag in e["tags"]]
        lines += [f"### {md(tag)} · {md(vocabulary[tag]['label'])}（{len(subset)}）", "", *table(subset), ""]
    return "\n".join(lines)


CSV_FIELDS = ["id", "title", "type", "url", "status", "summary", "tags", "authors", "year", "venue", "doi", "links", "attachments", "metadata_path", "notes_path"]


def render_csv(entries):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for entry in entries:
        row = {key: entry.get(key, "") for key in CSV_FIELDS}
        row["tags"] = ";".join(sorted(entry["tags"]))
        row["authors"] = ";".join(entry.get("authors", []))
        for key, default in (("links", {}), ("attachments", [])):
            row[key] = json.dumps(entry.get(key, default), ensure_ascii=False, sort_keys=True, default=str)
        writer.writerow(row)
    return output.getvalue()


def search(entries, aliases, tags=(), kind=None, status=None, query=""):
    canonical = set()
    for tag in tags:
        require(tag.casefold() in aliases, f"未知搜索标签：{tag}")
        canonical.add(aliases[tag.casefold()])
    return [e for e in entries if canonical.issubset(e["tags"])
            and (not kind or e["type"] == kind)
            and (not status or e["status"] == status)
            and (not query or query.casefold() in json.dumps(e, ensure_ascii=False, default=str).casefold())]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="仓库路径（默认由脚本位置确定）")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="校验元数据并生成 INDEX.md 和 catalog.csv")
    commands.add_parser("check", help="只校验，不写入文件；不检查网络或附件下载状态")
    query = commands.add_parser("search", help="搜索本地元数据；多个筛选条件取交集")
    query.add_argument("--tag", action="append", default=[], help="规范标签、中文名称或别名；可重复")
    query.add_argument("--type", choices=TYPE_LABELS)
    query.add_argument("--status", choices=STATUS_LABELS)
    query.add_argument("--query", default="", help="元数据关键字（不区分大小写，不搜索笔记正文）")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        entries, vocabulary, aliases = load_catalog(root)
        if args.command == "build":
            index = render_index(entries, vocabulary)
            catalog = render_csv(entries)
            (root / "INDEX.md").write_text(index, encoding="utf-8", newline="\n")
            (root / "catalog.csv").write_text(catalog, encoding="utf-8-sig", newline="\n")
            print(f"已生成 INDEX.md 和 catalog.csv：{len(entries)} 条资源")
        elif args.command == "check":
            print(f"校验通过：{len(entries)} 条资源，{len(vocabulary)} 个规范标签")
        else:
            matches = search(entries, aliases, args.tag, args.type, args.status, args.query)
            print("\n".join(table(matches)))
            print(f"\n找到 {len(matches)} 条资源")
        return 0
    except (CatalogError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
