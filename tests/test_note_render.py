from html.parser import HTMLParser
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from note_render import render_markdown


class ParsedMarkdown(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.formulas = []
        self.elements = []
        self.current = None
        self.feed(render_markdown(source, "example"))
        self.close()

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))
        display = dict(attrs).get("data-math-display")
        if display:
            self.current = {"display": display, "source": ""}
            self.formulas.append(self.current)

    def handle_endtag(self, tag):
        if tag in {"span", "div"}:
            self.current = None

    def handle_data(self, text):
        if self.current is not None:
            self.current["source"] += text


class NoteRenderTests(unittest.TestCase):
    def test_four_delimiters_preserve_tex_and_display_mode(self):
        parsed = ParsedMarkdown(
            r"行内 $x^2$ 和 \(\alpha_1\)。" + "\n\n"
            r"$$\frac{a}{b}$$" + "\n\n"
            "\\[\n\\begin{aligned}\nx &= 1 \\\\\ny &= 2\n\\end{aligned}\n\\]"
        )
        self.assertEqual(parsed.formulas, [
            {"display": "inline", "source": "x^2"},
            {"display": "inline", "source": r"\alpha_1"},
            {"display": "block", "source": r"\frac{a}{b}"},
            {"display": "block", "source": "\\begin{aligned}\nx &= 1 \\\\\ny &= 2\n\\end{aligned}"},
        ])

    def test_display_math_in_a_paragraph_keeps_surrounding_text(self):
        source = r"前文 \[x_i^2\] 中间 $$y_i$$ 后文"
        parsed = ParsedMarkdown(source)
        self.assertEqual(parsed.formulas, [
            {"display": "block", "source": "x_i^2"},
            {"display": "block", "source": "y_i"},
        ])
        rendered = render_markdown(source, "example")
        for text in ("前文", "中间", "后文"):
            self.assertIn(text, rendered)

    def test_bracket_blocks_respect_quote_and_list_prefixes(self):
        for source in (
            "> \\[\n> \\frac{a}{b}\n> + c\n> \\]",
            "- 项目\n\n  \\[\n  \\frac{a}{b}\n  + c\n  \\]",
        ):
            with self.subTest(source=source):
                self.assertEqual(ParsedMarkdown(source).formulas, [
                    {"display": "block", "source": "\\frac{a}{b}\n+ c"},
                ])

    def test_inline_math_survives_markdown_containers(self):
        source = "**加粗 \\(x_1\\)**\n\n- $y_2$\n\n| 公式 |\n| --- |\n| \\(z_3\\) |"
        self.assertEqual(ParsedMarkdown(source).formulas, [
            {"display": "inline", "source": "x_1"},
            {"display": "inline", "source": "y_2"},
            {"display": "inline", "source": "z_3"},
        ])

    def test_code_and_escaped_delimiters_are_not_math(self):
        sources = [
            r"`$x$` `\(x\)` `\[x\]` `$$x$$`",
            "```tex\n$x$\n\\(x\\)\n\\[x\\]\n$$x$$\n```",
            "    \\[x\\]\n\n    $$x$$",
            r"\$x\$ 和 \\(x\\) 和 \\[x\\]",
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(ParsedMarkdown(source).formulas, [])

    def test_escaped_closing_bracket_does_not_end_formula(self):
        parsed = ParsedMarkdown(r"\(a\\)b\) 然后 $c$")
        self.assertEqual(parsed.formulas, [
            {"display": "inline", "source": r"a\\)b"},
            {"display": "inline", "source": "c"},
        ])

    def test_unclosed_delimiters_do_not_swallow_later_paragraphs(self):
        for opening in ("$", "$$", r"\(", r"\["):
            source = opening + "x\n\n## 后续章节\n\n仍然可读"
            with self.subTest(opening=opening):
                parsed = ParsedMarkdown(source)
                self.assertEqual(parsed.formulas, [])
                self.assertIn("h2", [tag for tag, _ in parsed.elements])
                self.assertIn("仍然可读", render_markdown(source, "example"))

    def test_tex_cannot_inject_html_or_attributes(self):
        payload = r'\text{</span><img src=x onerror="bad"><script>alert(1)</script>&}'
        for opening, closing in (("$", "$"), ("$$", "$$"), (r"\(", r"\)"), (r"\[", r"\]")):
            with self.subTest(opening=opening):
                parsed = ParsedMarkdown(opening + payload + closing)
                self.assertEqual([item["source"] for item in parsed.formulas], [payload])
                self.assertFalse(any(tag in {"script", "img", "iframe"} for tag, _ in parsed.elements))
                self.assertFalse(any("onerror" in attrs for _, attrs in parsed.elements))


if __name__ == "__main__":
    unittest.main()
