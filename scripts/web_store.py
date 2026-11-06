"""Thread-safe local catalog operations; resource.md remains the only metadata source."""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import stat
import tempfile
import threading

import yaml

import catalog
from note_render import render_markdown
from resource_import import find_duplicates


class StoreError(ValueError):
    def __init__(self, message, status=400, duplicates=None):
        super().__init__(message)
        self.status = status
        if duplicates is not None:
            self.duplicates = duplicates


_REQUIRED = frozenset({"id", "title", "type", "url", "status", "summary", "tags"})
_DERIVED = frozenset({"metadata_path", "notes_path"})
_PROTECTED = frozenset({"id", "type"}) | _DERIVED


def _atomic_bytes(path, content):
    """Use the same sibling-tempfile/replace convention as zotero_sync.atomic_text."""
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(content)
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def _frontmatter_parts(raw):
    """Keep the delimiter's trailing newline and every subsequent body byte intact."""
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    lines = raw[len(bom):].splitlines(keepends=True)
    if not lines or lines[0].rstrip(b"\r\n") != b"---":
        raise StoreError("resource.md 缺少 YAML frontmatter")
    closing = next((i for i in range(1, len(lines)) if lines[i].rstrip(b"\r\n") == b"---"), None)
    if closing is None:
        raise StoreError("resource.md 的 YAML frontmatter 未闭合")
    newline = lines[0][3:] or b"\n"
    end = len(bom) + sum(len(line) for line in lines[:closing]) + 3
    return bom, newline, raw[end:]


def _card(data, raw=None, updates=None, remove_fields=()):
    bom, newline, body = _frontmatter_parts(raw) if raw is not None else (b"", b"\n", b"\n")
    try:
        if raw is None:
            header = yaml.safe_dump(data, allow_unicode=True, sort_keys=False).encode("utf-8")
        else:
            # Preserve untouched YAML nodes, including !!omap/!!pairs and aliases.
            # A safe_dump of the constructed Python values loses those YAML types.
            start = len(bom) + 3 + len(newline)
            end = len(raw) - len(body) - 3
            document = yaml.compose(raw[start:end].decode("utf-8"), Loader=catalog.UniqueLoader)
            changed = updates or {}
            removed = set(remove_fields)
            fields = []
            seen = set()
            for key, value in document.value:
                name = key.value
                if name in removed:
                    continue
                if name in changed:
                    value = yaml.compose(yaml.safe_dump(data[name], allow_unicode=True, sort_keys=False))
                    seen.add(name)
                fields.append((key, value))
            for name in changed:
                if name not in seen:
                    key = yaml.ScalarNode("tag:yaml.org,2002:str", name)
                    value = yaml.compose(yaml.safe_dump(data[name], allow_unicode=True, sort_keys=False))
                    fields.append((key, value))
            document.value = fields
            header = yaml.serialize(document, allow_unicode=True).encode("utf-8")
    except (yaml.YAMLError, TypeError, ValueError) as exc:
        raise StoreError(f"无法序列化元数据：{exc}") from exc
    return bom + b"---" + newline + header.replace(b"\n", newline) + b"---" + body


class Store:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._lock = threading.RLock()

    def _safe(self, path):
        path = Path(path)
        try:
            relative = path.relative_to(self.root)
        except ValueError as exc:
            raise StoreError("路径超出仓库范围", 403) from exc
        if any(part in {".", ".."} for part in relative.parts):
            raise StoreError("禁止路径穿越", 403)
        current = self.root
        for part in relative.parts:
            current = current / part
            try:
                info = current.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise StoreError("不允许通过符号链接或目录联接访问资源", 403)
        try:
            path.resolve().relative_to(self.root)
        except (ValueError, OSError, RuntimeError) as exc:
            raise StoreError("路径超出仓库范围", 403) from exc
        return path

    def _load(self):
        try:
            self._safe(self.root / "tags.yaml")
            for category in catalog.CATEGORIES:
                directory = self._safe(self.root / category)
                if directory.is_dir():
                    for folder in directory.iterdir():
                        if folder.name.startswith("."):
                            continue
                        self._safe(folder)
                        if folder.is_dir():
                            self._safe(folder / "resource.md")
                            self._safe(folder / "notes.md")
            return catalog.load_catalog(self.root)
        except (catalog.CatalogError, OSError, UnicodeError) as exc:
            raise StoreError(str(exc)) from exc

    def _find(self, resource_id, entries):
        if not isinstance(resource_id, str) or not catalog.SLUG.fullmatch(resource_id):
            raise StoreError("非法资源 id", 400)
        for entry in entries:
            if entry["id"] == resource_id:
                return entry
        raise StoreError("未找到该资源", 404)

    def _raw(self, entry):
        path = self._safe(self.root / entry["metadata_path"])
        try:
            return path, path.read_bytes()
        except OSError as exc:
            raise StoreError(f"无法读取资源：{exc}") from exc

    def _detail(self, entry, raw, notes=True):
        _, _, body = _frontmatter_parts(raw)
        try:
            text = body.decode("utf-8")
            notes_markdown = None
            if notes and entry["notes_path"]:
                notes_markdown = self._safe(self.root / entry["notes_path"]).read_bytes().decode("utf-8-sig")
            return {
                "entry": copy.deepcopy(entry),
                "revision": hashlib.sha256(raw).hexdigest(),
                "body_html": render_markdown(text, entry["id"]),
                "notes_html": render_markdown(notes_markdown, entry["id"]) if notes_markdown is not None else None,
                "notes_markdown": notes_markdown,
            }
        except (OSError, UnicodeError) as exc:
            raise StoreError(f"无法读取笔记或正文：{exc}") from exc

    def list_resources(self, query="", tags=(), kind=None, status=None):
        with self._lock:
            entries, vocabulary, aliases = self._load()
            try:
                found = catalog.search(entries, aliases, tags=tags, kind=kind, status=status, query=query)
            except (catalog.CatalogError, TypeError, AttributeError) as exc:
                raise StoreError(str(exc)) from exc
            return {"entries": found, "tags": vocabulary, "total": len(found)}

    def detail(self, resource_id):
        with self._lock:
            entries, _, _ = self._load()
            entry = self._find(resource_id, entries)
            _, raw = self._raw(entry)
            return self._detail(entry, raw)

    def _validate(self, data, folder, category, vocabulary):
        try:
            catalog.validate_entry(data, folder, category, vocabulary)
            for item in data.get("attachments", []):
                attachment = self._safe(folder / item["path"])
                try:
                    attachment.resolve().relative_to((folder / "resources").resolve())
                except ValueError as exc:
                    raise StoreError("附件路径超出当前资源的 resources/ 目录", 403) from exc
        except (catalog.CatalogError, TypeError, ValueError) as exc:
            if isinstance(exc, StoreError):
                raise
            raise StoreError(str(exc)) from exc

    def _commit(self, writes, new_folder=None):
        snapshots = {}
        attempted = []
        made_folder = False
        try:
            for path in writes:
                self._safe(path)
                snapshots[path] = path.read_bytes() if path.exists() else None
            if new_folder is not None:
                self._safe(new_folder)
                new_folder.mkdir()
                made_folder = True
            for path, content in writes.items():
                self._safe(path)
                attempted.append(path)
                _atomic_bytes(path, content)
        except (OSError, StoreError) as exc:
            rollback_errors = []
            for path in reversed(attempted):
                try:
                    if snapshots[path] is None:
                        path.unlink(missing_ok=True)
                    else:
                        _atomic_bytes(path, snapshots[path])
                except OSError as rollback_exc:
                    rollback_errors.append(str(rollback_exc))
            if made_folder:
                try:
                    new_folder.rmdir()
                except OSError as rollback_exc:
                    rollback_errors.append(str(rollback_exc))
            message = f"保存失败，已恢复原文件：{exc}"
            if rollback_errors:
                message = f"保存失败且回滚未完成，请检查文件权限：{exc}；" + "；".join(rollback_errors)
            raise StoreError(message, status=exc.status if isinstance(exc, StoreError) else 400) from exc

    def _writes(self, path, raw, entries, vocabulary):
        try:
            return {
                path: raw,
                self._safe(self.root / "INDEX.md"): catalog.render_index(entries, vocabulary).encode("utf-8"),
                self._safe(self.root / "catalog.csv"): catalog.render_csv(entries).encode("utf-8-sig"),
            }
        except (TypeError, ValueError) as exc:
            if isinstance(exc, StoreError):
                raise
            raise StoreError(f"无法生成索引：{exc}") from exc

    def update(self, resource_id, revision, updates, remove_fields=()):
        with self._lock:
            entries, vocabulary, _ = self._load()
            entry = self._find(resource_id, entries)
            path, raw = self._raw(entry)
            if not isinstance(revision, str) or revision != hashlib.sha256(raw).hexdigest():
                raise StoreError("资源已被修改，请重新载入后再保存", 409)
            if not isinstance(updates, dict) or not all(isinstance(key, str) for key in updates):
                raise StoreError("updates 必须是字段映射")
            if not isinstance(remove_fields, (list, tuple)) or not all(isinstance(key, str) for key in remove_fields):
                raise StoreError("remove_fields 必须是字段名列表")
            if _PROTECTED.intersection(updates) or (_REQUIRED | _DERIVED).intersection(remove_fields):
                raise StoreError("不能修改 id、type、派生路径或删除必填字段")
            if set(updates).intersection(remove_fields):
                raise StoreError("同一字段不能同时修改和移除")
            # Load original YAML, never serialize a JSON projection supplied by the browser.
            try:
                data = catalog.read_frontmatter(path)
            except catalog.CatalogError as exc:
                raise StoreError(str(exc)) from exc
            for key in remove_fields:
                data.pop(key, None)
            data.update(copy.deepcopy(updates))
            category = path.relative_to(self.root).parts[0]
            self._validate(data, path.parent, category, vocabulary)
            if not updates and not remove_fields:
                return self._detail(entry, raw)
            replacement = dict(data, metadata_path=entry["metadata_path"], notes_path=entry["notes_path"])
            new_raw = _card(data, raw, updates=updates, remove_fields=remove_fields)
            result = self._detail(replacement, new_raw)
            changed = [replacement if item["id"] == resource_id else item for item in entries]
            writes = self._writes(path, new_raw, changed, vocabulary)
            # Catch external edits made while preparing render/index output.
            if self._raw(entry)[1] != raw:
                raise StoreError("资源已被修改，请重新载入后再保存", 409)
            self._commit(writes)
            return result

    def create(self, metadata):
        with self._lock:
            entries, vocabulary, _ = self._load()
            if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
                raise StoreError("metadata 必须是字段映射")
            if _DERIVED.intersection(metadata):
                raise StoreError("派生路径不属于元数据")
            data = copy.deepcopy(metadata)
            data.setdefault("status", "unread")
            kind = data.get("type")
            category = next((name for name, value in catalog.CATEGORIES.items() if value == kind), None)
            if category is None:
                raise StoreError("type 必须为 paper / blog / doc")
            resource_id = data.get("id")
            if not isinstance(resource_id, str) or not catalog.SLUG.fullmatch(resource_id):
                raise StoreError("id 应为小写英文、数字和连字符")
            folder = self._safe(self.root / category / resource_id)
            self._validate(data, folder, category, vocabulary)
            duplicates = find_duplicates(data, entries)
            if duplicates:
                raise StoreError("该资源已存在，请打开已有条目", 409, duplicates)
            if any(item["id"] == resource_id for item in entries) or folder.exists():
                raise StoreError("该 id 或资源目录已存在，请使用其他 id", 409)
            path = folder / "resource.md"
            entry = dict(data, metadata_path=path.relative_to(self.root).as_posix(), notes_path="")
            raw = _card(data)
            result = self._detail(entry, raw, notes=False)
            changed = sorted([*entries, entry], key=lambda item: (item["type"], item["id"]))
            writes = self._writes(path, raw, changed, vocabulary)
            self._commit(writes, new_folder=folder)
            return result

    def asset(self, resource_id, relative_path):
        with self._lock:
            entries, _, _ = self._load()
            entry = self._find(resource_id, entries)
            if not isinstance(relative_path, str) or "\\" in relative_path or ":" in relative_path:
                raise StoreError("附件路径必须位于该资源的 resources/ 目录", 403)
            parts = relative_path.split("/")
            if len(parts) < 2 or parts[0] != "resources" or any(not part or part.startswith(".") for part in parts):
                raise StoreError("附件路径必须位于该资源的 resources/ 目录", 403)
            folder = (self.root / entry["metadata_path"]).parent
            path = self._safe(folder.joinpath(*parts))
            if not path.is_file():
                raise StoreError("附件不存在", 404)
            return path
