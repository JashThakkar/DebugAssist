from __future__ import annotations
from typing import Optional

import re

_WHITESPACE_RE = re.compile(r"\s+")
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\(?:[^\\\n]+\\)*[^\\\n]+")
_UNIX_PATH_RE = re.compile(r"(?:/[^/\n]+)+")
_LINE_NUMBER_RE = re.compile(r"\bline\s+\d+\b", flags=re.IGNORECASE)
_PLAIN_INT_RE = re.compile(r"\b\d+\b")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_QUOTED_STR_RE = re.compile(r"(['\"])(?:(?=(\\?))\2.)*?\1")


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    t = text.strip().lower()

    t = _WINDOWS_PATH_RE.sub("<PATH>", t)
    t = _UNIX_PATH_RE.sub("<PATH>", t)

    t = _LINE_NUMBER_RE.sub("line <LINE>", t)

    t = _HEX_RE.sub("<HEX>", t)

    t = _QUOTED_STR_RE.sub("<STR>", t)

    t = _PLAIN_INT_RE.sub("<NUM>", t)

    t = _WHITESPACE_RE.sub(" ", t).strip()
    
    return t


def combine_inputs(error_text: str, code: Optional[str] = None) -> str:
    err = error_text or ""
    
    if code and code.strip():
        return f"{err}\n<CODE>\n{code}"
    
    return err
