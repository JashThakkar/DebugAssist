from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import random
import re
import pandas as pd
import typer


app = typer.Typer(add_completion=False)

ERROR_FAMILIES = [
    "import_error",
    "syntax_error",
    "type_error",
    "value_error",
    "attribute_error",
    "key_error",
    "index_error",
    "file_error",
    "zero_division",
    "connection_error",
]

MODULES = [
    "requests", "numpy", "pandas", "flask", "django", "matplotlib",
    "sklearn", "yaml", "typer", "joblib", "bs4", "lxml"
]

FILES = [
    "main.py", "app.py", "script.py", "server.py", "utils.py",
    "src/handler.py", "src/service.py", "src/helpers.py",
    "project/module.py"
]

FUNCS = [
    "run", "main", "handler", "process", "parse", "load_data",
    "save_results", "compute", "transform", "validate"
]

VARS = [
    "data", "items", "result", "user", "payload", "config", "count",
    "total", "value", "x", "y", "idx", "key"
]

KEYS = [
    "user_id", "email", "name", "age", "items", "token", "id",
    "status", "created_at", "updated_at"
]

PATHS = [
    "data/input.csv", "data/users.json", "configs/app.yaml", "logs/app.log",
    "C:\\Users\\User\\Desktop\\input.txt",
    "/home/user/project/data/input.txt",
]

HOSTS = [
    "api.example.com", "localhost", "127.0.0.1", "example.org",
    "service.internal"
]

URLS = [
    "https://api.example.com/v1/users",
    "https://example.org/data",
    "http://localhost:8000/health",
    "https://service.internal/api",
]

STRINGS = ["abc", "12a", "None", "TRUE", "3.14.15", "01-32-2025"]

def _rand_line() -> int:
    return random.randint(1, 250)

def _choice(xs: List[str]) -> str:
    return random.choice(xs)

def _maybe_truncate(trace: str) -> str:
    """
    Simulate real-world paste: sometimes users paste full traceback,
    sometimes only the last lines.
    """
    r = random.random()
    lines = trace.strip("\n").splitlines()
    
    if len(lines) <= 3:
        return trace.strip("\n")

    if r < 0.20:
        return lines[-1]
    
    if r < 0.40:
        return "\n".join(lines[-2:])
    
    if r < 0.55:
        if lines[0].startswith("Traceback"):
            return "\n".join(lines[1:])
        
    return trace.strip("\n")

def _sanitize_fix(fix: str) -> str:
    return re.sub(r"\s+", " ", fix).strip()

@dataclass
class TemplateSpec:
    templates: List[str]
    fix_texts: List[str]

def _specs() -> Dict[str, TemplateSpec]:
    return {
        "import_error": TemplateSpec(
            templates=[
                """Traceback (most recent call last):
                    File "{file}", line {line}, in <module>
                        import {module}
                    ModuleNotFoundError: No module named '{module}'""",
                                    """Traceback (most recent call last):
                    File "{file}", line {line}, in <module>
                        from {module} import {name}
                    ImportError: cannot import name '{name}' from '{module}'""",
            ],
            fix_texts=[
                "Install the missing dependency: python -m pip install <module>; verify the correct virtual environment is active; restart the interpreter/kernel.",
                "Check the module version and import path; ensure the symbol exists in the installed package; avoid naming your file the same as the package."
            ],
        ),

        "syntax_error": TemplateSpec(
            templates=[
                """Traceback (most recent call last):
                    File "{file}", line {line}
                        {bad_line}
                    SyntaxError: invalid syntax""",
                                    """Traceback (most recent call last):
                    File "{file}", line {line}
                        {bad_line}
                    IndentationError: unexpected indent""",
            ],
            fix_texts=[
                "Check the indicated line for missing punctuation (':', ')', ']', quotes) or incomplete statements; comment out recent edits to isolate.",
                "Fix indentation consistency (spaces vs tabs); ensure blocks align; use 4 spaces per indent and reformat the file."
            ],
        ),

        "type_error": TemplateSpec(
            templates=[
                """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        {var} = {var} + {other}
                    TypeError: unsupported operand type(s) for +: '{t1}' and '{t2}'""",
                                    """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        {var}()
                    TypeError: '{t1}' object is not callable""",
            ],
            fix_texts=[
                "Inspect types with type(x); convert/cast to compatible types before operation; validate inputs (e.g., int(), float(), str()).",
                "You may be shadowing a function name with a variable; rename the variable or ensure you're calling a function, not a string/list/dict."
            ],
        ),

        "value_error": TemplateSpec(
            templates=[
                """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        {var} = int('{s}')
                    ValueError: invalid literal for int() with base 10: '{s}'""",
                                    """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        {var}.remove({num})
                    ValueError: list.remove(x): x not in list""",
            ],
            fix_texts=[
                "Validate/clean the string before casting; use try/except around parsing; confirm the expected format.",
                "Check whether the value exists before removing; use 'if x in list:'; verify list contents and logic."
            ],
        ),

        "attribute_error": TemplateSpec(
            templates=[
                """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        {var}.{attr}()
                    AttributeError: '{t1}' object has no attribute '{attr}'""",
                                    """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        {var}.split(',')
                    AttributeError: 'NoneType' object has no attribute 'split'""",
            ],
            fix_texts=[
                "Print the object and type(obj) before the failing line; confirm the attribute exists; check spelling and expected object type.",
                "Add a None-check before calling methods; ensure the variable is initialized and assigned the expected value before use."
            ],
        ),

        "key_error": TemplateSpec(
            templates=[
                """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        {var} = {dictname}['{key}']
                    KeyError: '{key}'""",
                                    """ERROR: Failed to process request
                    '{key}'""",
            ],
            fix_texts=[
                "Print dictionary keys and confirm the key exists; use dict.get(key, default) when appropriate; normalize key formatting (case/whitespace).",
                "If this came from logs, treat it like a missing dict key; add guard logic and verify upstream data shape."
            ],
        ),

        "index_error": TemplateSpec(
            templates=[
                """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        {var} = {listname}[{idx}]
                    IndexError: list index out of range""",
                                    """list index out of range""",
            ],
            fix_texts=[
                "Check list length with len(list); guard bounds; review loop conditions for off-by-one errors; handle empty lists.",
                "If the traceback is missing, still treat it as an IndexError; add bounds checks and validate inputs."
            ],
        ),

        "file_error": TemplateSpec(
            templates=[
                """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        f = open('{path}', 'r')
                    FileNotFoundError: [Errno 2] No such file or directory: '{path}'""",
                                    """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        f = open('{path}', 'w')
                    PermissionError: [Errno 13] Permission denied: '{path}'""",
            ],
            fix_texts=[
                "Print the absolute path and working directory; confirm the file exists; use pathlib to build paths; ensure correct relative path.",
                "Write to a permitted directory; check file/folder permissions; avoid protected OS paths; run with correct permissions if necessary."
            ],
        ),

        "zero_division": TemplateSpec(
            templates=[
                """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        {var} = {num} / 0
                    ZeroDivisionError: division by zero""",
                                    """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        {var} = {num} // 0
                    ZeroDivisionError: integer division or modulo by zero""",
            ],
            fix_texts=[
                "Guard denominators (if denom == 0); validate input ranges; handle empty/zero values before division.",
                "Check that a divisor is never zero; add fallback logic or filtering for invalid values."
            ],
        ),

        "connection_error": TemplateSpec(
            templates=[
                """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        r = requests.get('{url}', timeout={timeout})
                    requests.exceptions.Timeout: HTTPSConnectionPool(host='{host}', port=443): Read timed out.""",
                                    """Traceback (most recent call last):
                    File "{file}", line {line}, in {func}
                        r = requests.get('{url}')
                    requests.exceptions.ConnectionError: Failed to establish a new connection: [Errno 111] Connection refused""",
            ],
            fix_texts=[
                "Increase timeout; verify network connectivity/DNS; confirm the service is up; add retries/backoff; check proxy settings.",
                "Confirm the host/port is correct and reachable; check the server is running; validate firewall rules; try curl/ping for diagnostics."
            ],
        ),
    }

def _render(template: str) -> str:
    module = _choice(MODULES)
    file = _choice(FILES)
    func = _choice(FUNCS)
    var = _choice(VARS)
    other = _choice(VARS)
    name = random.choice(["get", "post", "Client", "Session", "DataFrame", "load", "dump"])
    t1 = random.choice(["int", "str", "list", "dict", "NoneType", "float", "bool"])
    t2 = random.choice(["int", "str", "list", "dict", "NoneType", "float", "bool"])
    attr = random.choice(["split", "items", "get", "append", "read", "to_json", "keys"])
    dictname = random.choice(["payload", "data", "row", "obj", "record"])
    listname = random.choice(["items", "results", "values", "rows"])
    key = _choice(KEYS)
    idx = random.randint(0, 25)
    s = _choice(STRINGS)
    num = random.randint(0, 999)
    path = _choice(PATHS)
    host = _choice(HOSTS)
    url = _choice(URLS)
    timeout = random.choice([1, 2, 3, 5, 10])

    bad_line = random.choice([
        "if x == 3 print(x)",
        "def func(x)\n        return x",
        "for i in range(10)\n    print(i)",
        "print('hello'",
        "my_list = [1, 2, 3",
        "return return x",
    ])

    rendered = template.format(
        module=module,
        file=file,
        line=_rand_line(),
        func=func,
        var=var,
        other=other,
        name=name,
        t1=t1,
        t2=t2,
        attr=attr,
        dictname=dictname,
        listname=listname,
        key=key,
        idx=idx,
        s=s,
        num=num,
        path=path,
        host=host,
        url=url,
        timeout=timeout,
        bad_line=bad_line,
    )
    return _maybe_truncate(rendered)

def _plan_counts(total: int | None, per_class: int | None) -> Dict[str, int]:
    if (total is None and per_class is None) or (total is not None and per_class is not None):
        raise typer.BadParameter("Provide exactly one of --total or --per-class")

    if per_class is not None:
        if per_class <= 0:
            raise typer.BadParameter("--per-class must be > 0")
        
        return {fam: per_class for fam in ERROR_FAMILIES}

    assert total is not None
    
    if total <= 0:
        raise typer.BadParameter("--total must be > 0")

    base = total // len(ERROR_FAMILIES)
    rem = total % len(ERROR_FAMILIES)
    counts = {fam: base for fam in ERROR_FAMILIES}

    fams = ERROR_FAMILIES[:]
    random.shuffle(fams)
    
    for i in range(rem):
        counts[fams[i]] += 1
        
    return counts

@app.command()
def main(
    total: int = typer.Option(None, help="Total number of rows to generate across all classes (e.g., 300)."),
    per_class: int = typer.Option(None, help="Number of rows per error family (e.g., 50)."),
    out: Path = typer.Option(Path("data/debug_cases.csv"), help="Output CSV path."),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
) -> None:

    random.seed(seed)

    specs = _specs()
    counts = _plan_counts(total=total, per_class=per_class)

    rows: List[Dict[str, str]] = []
    next_id = 1

    for family, n in counts.items():
        spec = specs[family]
        
        for _ in range(n):
            template = random.choice(spec.templates)
            fix = _sanitize_fix(random.choice(spec.fix_texts))
            err_text = _render(template)

            rows.append({
                "id": str(next_id),
                "error_text": err_text,
                "error_family": family,
                "fix_text": fix,
            })
            
            next_id += 1

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

if __name__ == "__main__":
    app()
