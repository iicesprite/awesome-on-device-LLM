"""Render safe Markdown and inert TeX fragments for the local KaTeX renderer."""

from __future__ import annotations

import html
from pathlib import PurePosixPath
from urllib.parse import quote, unquote, urlsplit
from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.utils import is_code_block


_RASTER = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".bmp", ".ico"})


def _closing_marker(source, marker, start, stop):
    """Find a delimiter whose backslash is not itself escaped."""
    end = source.find(marker, start, stop)
    while end >= 0:
        preceding = end - 1
        while preceding >= 0 and source[preceding] == "\\":
            preceding -= 1
        if (end - preceding - 1) % 2 == 0:
            return end
        end = source.find(marker, end + len(marker), stop)
    return -1


def _bracket_inline(state, silent):
    opening = state.src[state.pos:state.pos + 2]
    if opening not in {r"\(", r"\["}:
        return False
    closing = r"\)" if opening == r"\(" else r"\]"
    start = state.pos + 2
    end = _closing_marker(state.src, closing, start, state.posMax)
    if end < 0 or not state.src[start:end].strip():
        return False
    if not silent:
        kind = "math_inline" if opening == r"\(" else "math_inline_double"
        token = state.push(kind, "math", 0)
        token.content = state.src[start:end]
        token.markup = opening
    state.pos = end + 2
    return True


def _bracket_block(state, start_line, end_line, silent):
    if is_code_block(state, start_line):
        return False
    start = state.bMarks[start_line] + state.tShift[start_line]
    if not state.src.startswith(r"\[", start):
        return False
    lines = []
    for line in range(start_line, end_line):
        if line > start_line and state.sCount[line] < state.blkIndent and not state.isEmpty(line):
            return False
        begin = state.bMarks[line] + state.tShift[line]
        text = state.src[begin:state.eMarks[line]]
        if line == start_line:
            text = text[2:]
        end = _closing_marker(text, r"\]", 0, len(text))
        if end >= 0:
            if text[end + 2:].strip():
                return False  # An inline rule keeps surrounding paragraph text.
            lines.append(text[:end])
            if not "".join(lines).strip():
                return False
            if not silent:
                token = state.push("math_block", "math", 0)
                token.block = True
                token.content = "\n".join(lines)
                token.markup = r"\["
                token.map = [start_line, line + 1]
                state.line = line + 1
            return True
        lines.append(text)
    return False


def _math_fragment(renderer, tokens, index, options, env):
    token = tokens[index]
    display = "inline" if token.type == "math_inline" else "block"
    tag = "div" if token.block else "span"
    # TeX is text, never HTML or an attribute value. KaTeX consumes textContent.
    source = html.escape(token.content.strip())
    return f'<{tag} class="math-source" data-math-display="{display}">{source}</{tag}>'


def _destination(value, resource_id, image=False):
    if not isinstance(value, str) or any(ord(c) < 32 for c in value) or "\\" in value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme or parsed.netloc:
        if not image and parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
            return value
        return None
    path = unquote(parsed.path)
    if "\\" in path or any(ord(c) < 32 for c in path):
        return None
    if not image and path in {"notes.md", "./notes.md"}:
        return f"#resource={quote(resource_id, safe='')}&tab=notes"
    if not image and not path and parsed.fragment:
        return "#" + quote(unquote(parsed.fragment), safe="-._~")
    if path.startswith("./"):
        path = path[2:]
    parts = path.split("/")
    if len(parts) < 2 or parts[0] != "resources" or any(not p or p in {".", ".."} or ":" in p or p.startswith(".") for p in parts):
        return None
    if image and PurePosixPath(path).suffix.lower() not in _RASTER:
        return None
    target = f"/files/{quote(resource_id, safe='')}/{quote(path, safe='/')}"
    if parsed.query:
        target += "?" + quote(unquote(parsed.query), safe="=&-._~")
    if parsed.fragment:
        target += "#" + quote(unquote(parsed.fragment), safe="-._~")
    return target


def render_markdown(source, resource_id):
    parser = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")
    parser.use(dollarmath_plugin, allow_labels=False, double_inline=True)
    parser.inline.ruler.before("escape", "math_bracket_inline", _bracket_inline)
    parser.block.ruler.before(
        "fence", "math_bracket_block", _bracket_block,
        {"alt": ["paragraph", "reference", "blockquote", "list"]},
    )
    for kind in ("math_inline", "math_inline_double", "math_block"):
        parser.add_render_rule(kind, _math_fragment)
    tokens = parser.parse(source)

    def secure(items):
        for token in items:
            if token.type in {"link_open", "image"}:
                image = token.type == "image"
                attribute = "src" if image else "href"
                original = token.attrGet(attribute) or ""
                destination = _destination(original, resource_id, image=image)
                if destination is None:
                    token.attrs.pop(attribute, None)
                    token.attrSet("title", "已阻止不安全或不受支持的链接")
                else:
                    token.attrSet(attribute, destination)
                    if not image and urlsplit(destination).scheme:
                        token.attrSet("rel", "noopener noreferrer")
            if token.children:
                secure(token.children)

    secure(tokens)
    return parser.renderer.render(tokens, parser.options, {})
