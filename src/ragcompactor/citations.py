"""Formatting helpers for readable citation references."""

from __future__ import annotations

import re


_LINE_CITATION = re.compile(
    r"【(?P<source>\d+)†L(?P<start>\d+)(?:-L?(?P<end>\d+))?】"
)
_LINE_CITATION_GROUP = re.compile(
    r"(?P<prefix>\s*)(?P<citations>(?:【\d+†L\d+(?:-L?\d+)?】)+)"
)


def format_citations(text: str) -> str:
    """Render compact line citations as readable source and line references."""

    def readable_reference(match: re.Match[str]) -> str:
        source = match.group("source")
        start = match.group("start")
        end = match.group("end")
        lines = f"line {start}" if end is None else f"lines {start}–{end}"
        return f"Source {source}, {lines}"

    def replace(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        if not prefix and match.start() > 0:
            prefix = " "
        references = [
            readable_reference(citation)
            for citation in _LINE_CITATION.finditer(match.group("citations"))
        ]
        return f"{prefix}({'; '.join(references)})"

    return _LINE_CITATION_GROUP.sub(replace, text)
