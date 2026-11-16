#!/usr/bin/env python3
"""Export public VitePress pages without modifying the resource library."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from urllib.parse import quote, unquote, urlsplit

import yaml

try:
    from . import catalog, note_render
except ImportError:
    import catalog
    import note_render

MARKER = ".site-export"
MARKER_CONTENT = "awesome-on-device-llm site export v1\n"
PUBLIC_FIELDS = ("id", "title", "type", "url", "status", "summary", "tags", "authors", "year", "venue", "doi")
RASTER = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".bmp", ".ico"}
# Only these explicitly selected companion documents are public.
TRANSLATIONS = {"pariskv": "part_translate.md"}


def plain_path(path):
    """Reject symlinks and Windows reparse points in every existing component."""
    path = Path(os.path.abspath(path))
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise catalog.CatalogError(f"不发布符号链接或 junction：{part}")
    return path


def normalize_base(base):
    if not re.fullmatch(r"/(?:[A-Za-z0-9_.~-]+/)*", base) or any(p in {".", ".."} for p in base.split("/")):
        raise catalog.CatalogError("base 必须是正斜杠首尾的站点路径，例如 / 或 /repo/")
    return base


def preflight(root):
    plain_path(root / "tags.yaml")
    for category in catalog.CATEGORIES:
        directory = plain_path(root / category)
        if not directory.is_dir():
            raise catalog.CatalogError(f"缺少分类目录：{directory}")
        for folder in directory.iterdir():
            if folder.name.startswith("."):
                continue
            plain_path(folder)
            if folder.is_dir():
                plain_path(folder / "resource.md")
                plain_path(folder / "notes.md")


def without_frontmatter(source):
    lines = source.lstrip("\ufeff").splitlines(keepends=True)
    if lines and lines[0].rstrip("\r\n") == "---":
        for index in range(1, len(lines)):
            if lines[index].rstrip("\r\n") == "---":
                return "".join(lines[index + 1:])
    return source


def pdf_url(parsed):
    path = unquote(parsed.path).lower()
    host = (parsed.hostname or "").lower()
    paper_host = any(host == domain or host.endswith("." + domain) for domain in ("arxiv.org", "openreview.net"))
    return ".pdf" in unquote(parsed.geturl()).lower() or (paper_host and "/pdf" in path)


def raster_bytes(path):
    data = path.read_bytes()
    suffix = path.suffix.lower()
    valid = {
        ".png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": data.startswith(b"\xff\xd8\xff"),
        ".jpeg": data.startswith(b"\xff\xd8\xff"),
        ".gif": data.startswith((b"GIF87a", b"GIF89a")),
        ".webp": data.startswith(b"RIFF") and data[8:12] == b"WEBP",
        ".bmp": data.startswith(b"BM"),
        ".ico": data.startswith(b"\x00\x00\x01\x00"),
        ".avif": data[4:8] == b"ftyp" and any(brand in data[8:32] for brand in (b"avif", b"avis")),
    }
    return data if valid.get(suffix, False) else None


class Resolver:
    def __init__(self, root, entry, entries, base, assets):
        self.root, self.entry, self.base, self.assets = root, entry, base, assets
        self.folder = root / entry["metadata_path"]
        self.folder = self.folder.parent
        self.allow_images = False
        self.pages = {}
        for item in entries:
            self.pages[item["metadata_path"]] = item["resource_path"]
            folder = PurePosixPath(item["metadata_path"]).parent
            if item["notes_path"]:
                self.pages[(folder / "notes.md").as_posix()] = item["notes_path"]
            if item.get("translation_path"):
                self.pages[(folder / TRANSLATIONS[item["id"]]).as_posix()] = item["translation_path"]

    def __call__(self, value, *, image=False):
        if not isinstance(value, str) or "\\" in value or any(ord(c) < 32 for c in value):
            return None
        try:
            parsed = urlsplit(value)
            if parsed.scheme or parsed.netloc:
                if not image and parsed.scheme.lower() in {"http", "https"} and parsed.netloc and not pdf_url(parsed):
                    return value
                return None
        except ValueError:
            return None
        path = unquote(parsed.path)
        if "\\" in path or any(ord(c) < 32 for c in path) or path.startswith("/"):
            return None
        fragment = "#" + quote(unquote(parsed.fragment), safe="-._~") if parsed.fragment else ""
        if not image and not path:
            return fragment or None
        if image:
            if not self.allow_images:
                return None
            parts = path.removeprefix("./").split("/")
            if len(parts) < 3 or parts[:2] != ["resources", "assets"] or any(not p or p.startswith(".") or ":" in p for p in parts):
                return None
            relative = PurePosixPath(*parts[2:])
            if relative.suffix.lower() not in RASTER:
                return None
            try:
                source = plain_path(self.folder.joinpath(*parts))
                if not source.is_file():
                    return None
                target = f"assets/{self.entry['id']}/{relative.as_posix()}"
                if target not in self.assets:
                    data = raster_bytes(source)
                    if data is None:
                        return None
                    self.assets[target] = data
                return self.base + quote(target, safe="/")
            except (OSError, catalog.CatalogError):
                return None
        # Only known public Markdown pages may be navigated locally.
        if parsed.query:
            return None
        try:
            target = self.folder.joinpath(*PurePosixPath(path).parts).resolve().relative_to(self.root).as_posix()
        except (OSError, ValueError):
            return None
        page = self.pages.get(target)
        return self.base + page + fragment if page else None


def page(entry, vocabulary, body, resolver, *, suffix=""):
    escape = html.escape
    title = entry["title"] + suffix
    frontmatter = yaml.safe_dump({"title": title, "description": entry["summary"]}, allow_unicode=True, sort_keys=False).strip()
    chunks = [f"<p>{escape(entry['summary'])}</p>"]
    details = [catalog.TYPE_LABELS[entry["type"]], catalog.STATUS_LABELS[entry["status"]]]
    details.extend(entry.get("authors", []))
    for key in ("year", "venue", "doi"):
        if key in entry:
            details.append(str(entry[key]))
    chunks.append("<p>" + " · ".join(escape(value) for value in details) + "</p>")
    source = resolver(entry["url"])
    chunks.append(f'<p><a href="{escape(source, quote=True)}" rel="noopener noreferrer">主来源</a></p>' if source else '<p>主来源为 PDF，不在本站提供附件链接。</p>')
    navigation = [("全部资源", resolver.base)]
    navigation.append(("资源详情", resolver.base + entry["resource_path"]))
    if entry["notes_path"]:
        navigation.append(("阅读笔记", resolver.base + entry["notes_path"]))
    if entry.get("translation_path"):
        navigation.append(("部分译文", resolver.base + entry["translation_path"]))
    chunks.append('<nav aria-label="资源导航">' + " · ".join(f'<a href="{escape(url, quote=True)}">{label}</a>' for label, url in navigation) + "</nav>")
    chunks.append('<p class="resource-tags">' + " ".join(f'<a href="{escape(resolver.base + "?tag=" + quote(tag, safe=""), quote=True)}">{escape(vocabulary[tag]["label"])}</a>' for tag in entry["tags"]) + "</p>")
    chunks.append(note_render.render_markdown(body, entry["id"], url_resolver=resolver))
    # A single HTML block prevents Markdown re-parsing HTML/code blank lines.
    content = "\n".join(chunks).replace("\r", "").replace("\n", "&#10;")
    heading = '# <span v-pre>' + "".join(f"&#{ord(char)};" for char in title) + "</span> {#page-title}"
    return f'---\n{frontmatter}\n---\n\n{heading}\n\n<div class="resource-content" v-pre>{content}</div>\n'


def prepare_output(root, output):
    output = plain_path(output)
    if output == root or output in root.parents or any(output.is_relative_to(root / category) for category in catalog.CATEGORIES):
        raise catalog.CatalogError("输出目录不能覆盖资源库或分类源文件")
    if output.exists():
        if not output.is_dir():
            raise catalog.CatalogError("输出路径必须是目录")
        children = list(output.iterdir())
        if children:
            marker = plain_path(output / MARKER)
            if not marker.is_file() or marker.read_text(encoding="utf-8") != MARKER_CONTENT:
                raise catalog.CatalogError("拒绝覆盖非空目录：缺少本站生成标记")
            for folder, directories, files in os.walk(output, followlinks=False):
                for name in directories + files:
                    plain_path(Path(folder) / name)
    return output


def export_site(root=catalog.DEFAULT_ROOT, output=None, base="/"):
    root = plain_path(root)
    base = normalize_base(base)
    output = prepare_output(root, Path(output) if output is not None else root / "site" / ".generated")
    preflight(root)
    entries, vocabulary, _ = catalog.load_catalog(root)
    public = []
    for entry in entries:
        item = {key: entry[key] for key in PUBLIC_FIELDS if key in entry}
        item["resource_path"] = f"resources/{entry['id']}.html"
        item["notes_path"] = f"notes/{entry['id']}.html" if entry["notes_path"] else ""
        if filename := TRANSLATIONS.get(entry["id"]):
            translation = plain_path((root / entry["metadata_path"]).parent / filename)
            if translation.is_file():
                item["translation_path"] = f"translations/{entry['id']}.html"
        item["search_text"] = " ".join(str(value) for key in PUBLIC_FIELDS for value in (item.get(key, []) if isinstance(item.get(key), list) else [item.get(key, "")]))
        public.append(item)
    assets, pages = {}, {}
    for original, item in zip(entries, public):
        # Source paths are internal to the exporter, never included in JSON.
        resolver_entry = {**item, "metadata_path": original["metadata_path"]}
        resolver_entries = [{**p, "metadata_path": source["metadata_path"]} for source, p in zip(entries, public)]
        resolver = Resolver(root, resolver_entry, resolver_entries, base, assets)
        source = root / original["metadata_path"]
        pages[f"resources/{item['id']}.md"] = page(item, vocabulary, without_frontmatter(source.read_text(encoding="utf-8-sig")), resolver)
        if original["notes_path"]:
            resolver.allow_images = True
            pages[f"notes/{item['id']}.md"] = page(item, vocabulary, without_frontmatter((root / original["notes_path"]).read_text(encoding="utf-8-sig")), resolver, suffix=" · 阅读笔记")
        if item.get("translation_path"):
            resolver.allow_images = True
            translation = source.with_name(TRANSLATIONS[item["id"]])
            pages[f"translations/{item['id']}.md"] = page(item, vocabulary, without_frontmatter(translation.read_text(encoding="utf-8-sig")), resolver, suffix=" · 部分译文")
    pages["index.md"] = '---\nlayout: page\ntitle: 端侧 LLM 资源库\n---\n\n<ResourceCatalog />\n'
    pages["catalog.json"] = json.dumps({"entries": public, "tags": {tag: {"label": info["label"], "aliases": info.get("aliases", [])} for tag, info in vocabulary.items()}}, ensure_ascii=False, indent=2) + "\n"
    # Validate and read all sources before replacing any previous generated output.
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / MARKER).write_text(MARKER_CONTENT, encoding="utf-8")
    for name, text in pages.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    for name, data in assets.items():
        destination = output / "public" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return {"entries": len(entries), "notes": sum(bool(item["notes_path"]) for item in public), "assets": len(assets), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=catalog.DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base", default=os.environ.get("SITE_BASE", "/"))
    args = parser.parse_args(argv)
    try:
        result = export_site(args.root, args.output, args.base)
    except (catalog.CatalogError, OSError, UnicodeError) as exc:
        print(f"导出失败：{exc}", file=sys.stderr)
        return 1
    print(f"已导出 {result['entries']} 条资源、{result['notes']} 份笔记、{result['assets']} 张配图至 {result['output']}。")
    print("输出仅供 VitePress 构建；不含 PDF、私人文件或本地附件。请勿手工编辑生成目录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
