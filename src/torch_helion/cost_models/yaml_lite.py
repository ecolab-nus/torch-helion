"""YAML reader for Loom perf files.

Uses PyYAML when available; otherwise a small parser covering the subset
the perf files use (nested mappings, lists of mappings, quoted scalars,
anchors ``&name`` and aliases ``*name``).
"""

from __future__ import annotations

import re
from typing import Any

try:  # pragma: no cover - exercised when PyYAML is installed
    import yaml as _yaml
except Exception:  # pragma: no cover
    _yaml = None


def load_yaml(text: str) -> Any:
    if _yaml is not None:
        return _yaml.safe_load(text)
    return _MiniYaml(text).parse()


class _MiniYaml:
    def __init__(self, text: str) -> None:
        self.lines: list[tuple[int, str]] = []
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith('"') else raw.rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            self.lines.append((indent, line.strip()))
        self.pos = 0
        self.anchors: dict[str, Any] = {}

    def parse(self) -> Any:
        return self._block(0)

    def _block(self, indent: int) -> Any:
        if self.pos >= len(self.lines):
            return {}
        if self.lines[self.pos][1].startswith("- "):
            return self._list(indent)
        return self._map(indent)

    def _list(self, indent: int) -> list:
        out = []
        while self.pos < len(self.lines) and self.lines[self.pos][0] == indent and self.lines[self.pos][1].startswith("- "):
            ind, line = self.lines[self.pos]
            rest = line[2:].strip()
            self.pos += 1
            if ":" in rest and not rest.startswith(('"', "'")):
                # inline mapping entry: "- key: value" followed by deeper keys
                self.lines.insert(self.pos, (indent + 2, rest))
                out.append(self._map(indent + 2))
            else:
                out.append(self._scalar(rest))
        return out

    def _map(self, indent: int) -> dict:
        out: dict[str, Any] = {}
        while self.pos < len(self.lines) and self.lines[self.pos][0] == indent and not self.lines[self.pos][1].startswith("- "):
            ind, line = self.lines[self.pos]
            key, _, rest = line.partition(":")
            key = key.strip()
            rest = rest.strip()
            self.pos += 1
            anchor = None
            m = re.match(r"&(\w+)\s*(.*)", rest)
            if m:
                anchor, rest = m.group(1), m.group(2).strip()
            if rest:
                value = self._scalar(rest)
            elif self.pos < len(self.lines) and self.lines[self.pos][0] > indent:
                value = self._block(self.lines[self.pos][0])
            else:
                value = None
            if anchor:
                self.anchors[anchor] = value
            out[key] = value
        return out

    def _scalar(self, text: str) -> Any:
        if text.startswith("*"):
            return self.anchors[text[1:].strip()]
        if text.startswith("[") and text.endswith("]"):
            return [self._scalar(p.strip()) for p in text[1:-1].split(",") if p.strip()]
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            return text[1:-1]
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        if re.fullmatch(r"-?\d+\.\d*", text):
            return float(text)
        if text in ("true", "True"):
            return True
        if text in ("false", "False"):
            return False
        return text
