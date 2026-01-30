from __future__ import annotations

import re
from typing import Optional

_RULES: list[tuple[str, re.Pattern]] = [
    ("import_error", re.compile(r"\bmoduleNotFoundError\b", re.IGNORECASE)),
    ("import_error", re.compile(r"\bimporterror\b", re.IGNORECASE)),
    ("import_error", re.compile(r"\bno module named\b", re.IGNORECASE)),
    ("import_error", re.compile(r"\bcannot import name\b", re.IGNORECASE)),

    ("syntax_error", re.compile(r"\bsyntaxerror\b", re.IGNORECASE)),
    ("syntax_error", re.compile(r"\bindentationerror\b", re.IGNORECASE)),
    ("syntax_error", re.compile(r"\bunexpected indent\b", re.IGNORECASE)),
    ("syntax_error", re.compile(r"\bexpected an indented block\b", re.IGNORECASE)),

    ("type_error", re.compile(r"\btypeerror\b", re.IGNORECASE)),
    ("type_error", re.compile(r"\bnot callable\b", re.IGNORECASE)),
    ("type_error", re.compile(r"\bunsupported operand type", re.IGNORECASE)),
    ("type_error", re.compile(r"\bhas no len\(\)\b", re.IGNORECASE)),

    ("value_error", re.compile(r"\bvalueerror\b", re.IGNORECASE)),
    ("value_error", re.compile(r"\binvalid literal for int\(\)", re.IGNORECASE)),
    ("value_error", re.compile(r"\bcould not convert string to float\b", re.IGNORECASE)),
    ("value_error", re.compile(r"\blist\.remove\(x\): x not in list\b", re.IGNORECASE)),

    ("attribute_error", re.compile(r"\battributeerror\b", re.IGNORECASE)),
    ("attribute_error", re.compile(r"\bhas no attribute\b", re.IGNORECASE)),
    ("attribute_error", re.compile(r"\bnonetype\b.*\bhas no attribute\b", re.IGNORECASE)),

    ("key_error", re.compile(r"\bkeyerror\b", re.IGNORECASE)),
    ("key_error", re.compile(r"^\s*['\"][A-Za-z0-9_ -]{1,40}['\"]\s*$", re.MULTILINE)),

    ("index_error", re.compile(r"\bindexerror\b", re.IGNORECASE)),
    ("index_error", re.compile(r"\blist index out of range\b", re.IGNORECASE)),

    ("file_error", re.compile(r"\bfilenotfounderror\b", re.IGNORECASE)),
    ("file_error", re.compile(r"\bpermissionerror\b", re.IGNORECASE)),
    ("file_error", re.compile(r"\bno such file or directory\b", re.IGNORECASE)),
    ("file_error", re.compile(r"\bpermission denied\b", re.IGNORECASE)),

    ("zero_division", re.compile(r"\bzerodivisionerror\b", re.IGNORECASE)),
    ("zero_division", re.compile(r"\bdivision by zero\b", re.IGNORECASE)),
    ("zero_division", re.compile(r"\binteger division or modulo by zero\b", re.IGNORECASE)),

    ("connection_error", re.compile(r"\brequests\.exceptions\.timeout\b", re.IGNORECASE)),
    ("connection_error", re.compile(r"\brequests\.exceptions\.connectionerror\b", re.IGNORECASE)),
    ("connection_error", re.compile(r"\bread timed out\b", re.IGNORECASE)),
    ("connection_error", re.compile(r"\bconnection refused\b", re.IGNORECASE)),
]


def rule_predict(error_text: str) -> Optional[str]:
    if not error_text or not error_text.strip():
        return None

    text = error_text.strip()
    
    for family, pattern in _RULES:
        if pattern.search(text):
            return family
        
    return None
