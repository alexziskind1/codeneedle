"""Parse a source file and enumerate named functions with enough body lines to test."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


MIN_BODY_LINES = 20
BONUS_CAP = 40  # extra lines past the primary 20 that count toward the "blue" bonus


@dataclass
class FunctionTarget:
    name: str
    start_line: int           # 1-indexed line of first body line (after the opening brace)
    body_lines: list[str]     # body lines starting at start_line, excluding the closing brace line
    language: str = "js"      # "js" or "py" — controls the prompt wording
    source_path: Path | None = None  # which file this came from (for multi-file corpora)

    @property
    def primary_lines(self) -> list[str]:
        return self.body_lines[:MIN_BODY_LINES]

    @property
    def bonus_lines(self) -> list[str]:
        return self.body_lines[MIN_BODY_LINES:MIN_BODY_LINES + BONUS_CAP]


@dataclass
class Source:
    """A combined corpus: one or more files concatenated for a single benchmark run."""
    files: list[Path]
    text: str                       # full text fed to the model
    targets: list[FunctionTarget]
    language: str

    @property
    def display_name(self) -> str:
        if len(self.files) == 1:
            return self.files[0].name
        return f"{len(self.files)} files from {self.files[0].parent}"


def language_of(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".js", ".mjs", ".cjs"):
        return "js"
    if suffix == ".py":
        return "py"
    if suffix == ".rs":
        return "rs"
    if suffix in (".cpp", ".cc", ".cxx", ".c", ".hpp", ".h"):
        return "cpp"
    if suffix in (".ts", ".tsx", ".mts", ".cts"):
        return "ts"
    raise ValueError(f"Unsupported file type: {suffix!r}. Supported: .js, .py, .rs, .cpp, .ts")


def extract(path: Path) -> list[FunctionTarget]:
    source = path.read_text()
    lang = language_of(path)
    extractors = {
        "js": _extract_js,
        "ts": _extract_js,
        "py": _extract_py,
        "rs": _extract_rs,
        "cpp": _extract_cpp,
    }
    targets = extractors[lang](source)
    for t in targets:
        t.language = lang
        t.source_path = path
    return targets


def load_source_glob(
    directory: Path,
    glob: str,
    limit: int | None = None,
) -> Source:
    """Glob a directory for source files, concatenate them, extract all targets.

    Files are concatenated with comment-marker headers so the model can see file
    boundaries. All files must be the same language. Across files, duplicate
    function names are deduplicated (first occurrence wins) so the prompt is
    unambiguous when looked up by name.
    """
    paths = sorted(p for p in directory.glob(glob) if p.is_file())
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no files match {directory}/{glob}")

    lang = language_of(paths[0])
    for p in paths[1:]:
        if language_of(p) != lang:
            raise ValueError(
                f"mixed languages in glob: {paths[0]} is {lang}, {p} is {language_of(p)}"
            )

    parts: list[str] = []
    targets: list[FunctionTarget] = []
    seen_names: set[str] = set()
    line_offset = 0

    for p in paths:
        text = p.read_text()
        header = _file_header(lang, p)
        parts.append(header)
        parts.append(text)
        if not text.endswith("\n"):
            parts.append("\n")
        parts.append("\n")  # blank line between files

        header_line_count = header.count("\n")
        for t in extract(p):
            if t.name in seen_names:
                # Skip cross-file collisions — prompt would be ambiguous by name.
                continue
            seen_names.add(t.name)
            t.start_line += line_offset + header_line_count
            targets.append(t)

        line_offset += header.count("\n") + text.count("\n") + (0 if text.endswith("\n") else 1) + 1

    combined = "".join(parts)
    return Source(files=paths, text=combined, targets=targets, language=lang)


def _file_header(lang: str, path: Path) -> str:
    marker = "#" if lang == "py" else "//"
    return f"{marker} ====== {path} ======\n"


# --- JavaScript ---------------------------------------------------------------


def _extract_js(source: str) -> list[FunctionTarget]:
    import esprima

    try:
        tree = esprima.parseModule(
            source, options={"loc": True, "tolerant": True}
        )
    except Exception:
        tree = esprima.parseScript(
            source, options={"loc": True, "tolerant": True}
        )

    lines = source.splitlines()
    targets: list[FunctionTarget] = []
    seen: set[str] = set()

    def emit(name: str, block) -> None:
        if name in seen:
            return
        brace_line = block.loc.start.line  # line of '{'
        close_line = block.loc.end.line    # line of '}'
        if close_line - brace_line < MIN_BODY_LINES + 1:
            return
        # lines strictly between { and }
        body = lines[brace_line:close_line - 1]
        if len(body) < MIN_BODY_LINES:
            return
        seen.add(name)
        targets.append(
            FunctionTarget(
                name=name,
                start_line=brace_line + 1,
                body_lines=body,
            )
        )

    def hint_for(parent_type: str | None, key: str, parent) -> str | None:
        if parent_type == "VariableDeclarator" and key == "init":
            pid = getattr(parent, "id", None)
            if pid is not None and getattr(pid, "type", None) == "Identifier":
                return pid.name
        elif parent_type == "AssignmentExpression" and key == "right":
            left = getattr(parent, "left", None)
            if left is None:
                return None
            if getattr(left, "type", None) == "Identifier":
                return left.name
            if getattr(left, "type", None) == "MemberExpression":
                prop = getattr(left, "property", None)
                if prop is not None and getattr(prop, "type", None) == "Identifier":
                    return prop.name
        elif parent_type == "Property" and key == "value":
            k = getattr(parent, "key", None)
            if k is not None and getattr(k, "type", None) == "Identifier":
                return k.name
            if k is not None and getattr(k, "type", None) == "Literal":
                return str(k.value)
        elif parent_type == "MethodDefinition" and key == "value":
            k = getattr(parent, "key", None)
            if k is not None and getattr(k, "type", None) == "Identifier":
                return k.name
        return None

    def walk(node, name_hint: str | None = None) -> None:
        if node is None or not hasattr(node, "type"):
            return
        t = node.type

        if t == "FunctionDeclaration":
            nm = (node.id.name if getattr(node, "id", None) else None) or name_hint
            body = getattr(node, "body", None)
            if nm and body is not None and body.type == "BlockStatement":
                emit(nm, body)
        elif t == "FunctionExpression":
            nm = (
                (node.id.name if getattr(node, "id", None) else None)
                or name_hint
            )
            body = getattr(node, "body", None)
            if nm and body is not None and body.type == "BlockStatement":
                emit(nm, body)
        elif t == "ArrowFunctionExpression":
            body = getattr(node, "body", None)
            if name_hint and body is not None and body.type == "BlockStatement":
                emit(name_hint, body)

        # recurse
        for key, val in vars(node).items():
            if key == "loc":
                continue
            if isinstance(val, list):
                for item in val:
                    if hasattr(item, "type"):
                        walk(item, hint_for(t, key, node))
            elif hasattr(val, "type"):
                walk(val, hint_for(t, key, node))

    walk(tree)
    return targets


# --- Python -------------------------------------------------------------------


def _extract_py(source: str) -> list[FunctionTarget]:
    import ast

    tree = ast.parse(source)
    lines = source.splitlines()
    targets: list[FunctionTarget] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in seen or not node.body:
            continue
        start = node.body[0].lineno
        end = max(getattr(n, "end_lineno", n.lineno) for n in node.body)
        body = lines[start - 1:end]
        if len(body) < MIN_BODY_LINES:
            continue
        seen.add(node.name)
        targets.append(
            FunctionTarget(name=node.name, start_line=start, body_lines=body)
        )
    return targets


# --- Rust ---------------------------------------------------------------------


import re

_RS_FN_RE = re.compile(
    r"^\s*"
    r"(?:pub(?:\s*\([^)]*\))?\s+)?"                                      # pub / pub(crate) / pub(in path)
    r"(?:(?:default|const|async|unsafe|extern(?:\s+\"[^\"]*\")?)\s+)*"   # qualifiers, any combo/order
    r"fn\s+(\w+)",                                                       # fn name (generics/params may follow)
)

# Raw string opener: optional b/c byte/c-string prefix, r, zero-or-more #, ".
# Closer is '"' + same number of #. (Hashless r"..." and byte br"..." included.)
_RS_RAW_STRING_OPEN = re.compile(r'[bc]?r(#*)"')


def _rs_char_literal_end(line: str, i: int) -> int | None:
    """If `line[i]` ('\\'') opens a Rust char literal, return the index just past
    its closing quote; otherwise None.

    Char literals (`'x'`, `'\\n'`, `'\\''`, `'\\u{1F600}'`) carry `{ } " \\` that must
    NOT be treated as braces or string delimiters. Lifetimes (`'a`, `'static`) and
    the label syntax (`'outer: loop`) open with the same quote but have no closing
    one — those return None so the caller treats the quote as an ordinary char.
    """
    n = len(line)
    j = i + 1
    if j >= n:
        return None
    if line[j] == "\\":
        j += 1
        if j < n and line[j] == "u" and j + 1 < n and line[j + 1] == "{":
            close = line.find("}", j + 2)
            if close == -1:
                return None
            j = close + 1
        else:
            j += 1  # the single escaped char (\n, \t, \\, \', \" …)
        return j + 1 if j < n and line[j] == "'" else None
    # Simple one-char literal: 'X'. Anything else (identifier start) is a lifetime.
    return i + 3 if j + 1 < n and line[j + 1] == "'" else None


def _rs_significant_chars(lines: list[str], start: int):
    """Yield (line_index, char) for every *code* character from line `start`,
    skipping line comments, nestable /* */ block comments, string and raw-string
    literals, and char literals.

    Block-comment nesting depth and raw-string state carry across lines (Rust
    block comments nest, unlike C); ordinary strings and char literals stay
    within a single line.
    """
    block_depth = 0
    in_raw = False
    raw_closer = ""
    i = start
    while i < len(lines):
        line = lines[i]
        n = len(line)
        col = 0
        while col < n:
            if in_raw:
                ci = line.find(raw_closer, col)
                if ci == -1:
                    col = n
                else:
                    col = ci + len(raw_closer)
                    in_raw = False
                continue
            if block_depth > 0:
                open_i = line.find("/*", col)
                close_i = line.find("*/", col)
                if close_i == -1 and open_i == -1:
                    col = n
                elif open_i != -1 and (close_i == -1 or open_i < close_i):
                    block_depth += 1
                    col = open_i + 2
                else:
                    block_depth -= 1
                    col = close_i + 2
                continue
            ch = line[col]
            if ch == "/" and col + 1 < n and line[col + 1] == "/":
                break
            if ch == "/" and col + 1 < n and line[col + 1] == "*":
                block_depth += 1
                col += 2
                continue
            if ch == '"':
                col += 1
                while col < n and line[col] != '"':
                    if line[col] == "\\":
                        col += 1
                    col += 1
                col += 1
                continue
            # Raw string: r"…", r#"…"#, and byte/c-string forms br"…", cr"…".
            # Guard on the preceding char so an identifier ending in r/b/c (e.g.
            # `for`, `myr`) isn't mistaken for a raw-string prefix.
            if ch in "rbc" and (col == 0 or not (line[col - 1].isalnum() or line[col - 1] == "_")):
                m = _RS_RAW_STRING_OPEN.match(line, col)
                if m:
                    hashes = len(m.group(1))
                    raw_closer = '"' + "#" * hashes
                    col = m.end()
                    ci = line.find(raw_closer, col)
                    if ci == -1:
                        in_raw = True
                        col = n
                    else:
                        col = ci + len(raw_closer)
                    continue
            if ch == "'":
                end = _rs_char_literal_end(line, col)
                if end is not None:
                    col = end
                    continue
            yield i, ch
            col += 1
        i += 1


def _rs_block_end(lines: list[str], bo: int) -> int:
    """Given the line holding a block's opening '{', return the line index of the
    matching '}'. Returns `bo` if the block never closes."""
    depth = 0
    for li, ch in _rs_significant_chars(lines, bo):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return li
    return bo


def _rs_find_body_brace(lines: list[str], start: int) -> int | None:
    """From a fn-head line, find the line index of the body's opening '{'.

    The signature may span lines (multi-line params, return type, where clause).
    Tracks () and [] nesting so a ';' inside an array type (`[u8; 32]`, common in
    parameter and return types) isn't mistaken for the ';' that ends a bodyless
    declaration. A '{' at bracket-depth 0 opens the body; a ';' there means it was
    only a declaration (e.g. a trait method signature) — return None.
    """
    bracket = 0
    for li, ch in _rs_significant_chars(lines, start):
        if ch in "([":
            bracket += 1
        elif ch in ")]":
            if bracket > 0:
                bracket -= 1
        elif ch == "{" and bracket == 0:
            return li
        elif ch == ";" and bracket == 0:
            return None
    return None


_RS_MACRO_RULES_RE = re.compile(r"^\s*macro_rules!\s")


def _extract_rs(source: str) -> list[FunctionTarget]:
    lines = source.splitlines()
    targets: list[FunctionTarget] = []
    seen: set[str] = set()
    i = 0
    n = len(lines)
    while i < n:
        # Skip a macro_rules! body wholesale — the `fn` tokens inside are
        # templates with $metavariables, not real definitions. (Functions
        # *generated* by attribute macros aren't in the source text at all.)
        if _RS_MACRO_RULES_RE.match(lines[i]):
            bo = _rs_find_body_brace(lines, i)
            if bo is None:
                i += 1
                continue
            close_li = _rs_block_end(lines, bo)
            i = close_li + 1 if close_li > bo else i + 1
            continue
        m = _RS_FN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        bo = _rs_find_body_brace(lines, i)
        if bo is None:
            i += 1
            continue
        close_li = _rs_block_end(lines, bo)
        body = lines[bo + 1 : close_li]
        if len(body) >= MIN_BODY_LINES and name not in seen:
            seen.add(name)
            targets.append(
                FunctionTarget(name=name, start_line=bo + 2, body_lines=body)
            )
        i = close_li + 1 if close_li > bo else bo + 1
    return targets


# --- C / C++ ------------------------------------------------------------------


# Matches the start of a function definition. Allows a leading "::" or
# namespace-qualified return type and a qualified function name. The parameter
# list may continue onto following lines — only the opening "(" needs to be here.
# A negative lookahead rejects control-flow keywords so statements like
# "if (...)" / "for (...)" are not mistaken for definitions.
_CPP_FN_HEAD_RE = re.compile(
    r"^\s*(?!(?:if|for|while|switch|catch|return|else|do|sizeof)\b)"
    r"(?:[\w:~][\w\s:*&<>,]*?\s[\s*&]*)?"   # optional return type / specifiers, ending in whitespace
    r"((?:\w+::)*~?\w+)\s*\(",              # qualified function name + open paren
)

_CPP_CONTROL = {"if", "for", "while", "switch", "catch", "return", "else", "do", "sizeof"}

# Raw string opener: optional encoding prefix, R, ", delimiter (no parens/space/
# backslash), then "(". Closer is ")" + delimiter + '"'. May span lines.
_CPP_RAW_OPEN = re.compile(r'(?:u8|u|U|L)?R"([^()\\ ]*)\(')


def _cpp_significant_chars(lines: list[str], start: int):
    """Yield (line_index, char) for every *code* character from line `start`,
    skipping string literals, char literals, raw strings, and // and /* */ comments.

    Without this, braces/semicolons inside `"{ }"`, `'}'`, a raw string
    `R"({)"`, or comments are miscounted — e.g. a logf with a `"{\\"key\\":..."`
    string would inflate the brace depth and over- or under-run the function body.
    Block-comment and raw-string state are carried across lines; ordinary C/C++
    string and char literals do not span lines, so they are scanned within a line.
    """
    in_block = False
    in_raw = False
    raw_closer = ""
    i = start
    while i < len(lines):
        line = lines[i]
        n = len(line)
        col = 0
        while col < n:
            if in_raw:
                ci = line.find(raw_closer, col)
                if ci == -1:
                    col = n
                else:
                    col = ci + len(raw_closer)
                    in_raw = False
                continue
            if in_block:
                end = line.find("*/", col)
                if end == -1:
                    col = n
                else:
                    col = end + 2
                    in_block = False
                continue
            ch = line[col]
            if ch == "/" and col + 1 < n and line[col + 1] == "/":
                break  # line comment — skip the rest of this line
            if ch == "/" and col + 1 < n and line[col + 1] == "*":
                in_block = True
                col += 2
                continue
            # Raw string R"delim( ... )delim" — detect before the plain '"' path.
            if ch in "RuUL" and (col == 0 or not (line[col - 1].isalnum() or line[col - 1] == "_")):
                rm = _CPP_RAW_OPEN.match(line, col)
                if rm:
                    raw_closer = ")" + rm.group(1) + '"'
                    col = rm.end()
                    ci = line.find(raw_closer, col)
                    if ci == -1:
                        in_raw = True
                        col = n
                    else:
                        col = ci + len(raw_closer)
                    continue
            if ch == '"' or ch == "'":
                quote = ch
                col += 1
                while col < n and line[col] != quote:
                    if line[col] == "\\":
                        col += 1
                    col += 1
                col += 1
                continue
            yield i, ch
            col += 1
        i += 1


def _cpp_signature_brace(lines: list[str], i: int) -> int | None:
    """From a function-head candidate at line i, find the opening body brace.

    Balances parens across lines for the (possibly multi-line) parameter list,
    then scans forward for the first "{" (the body) or ";" (a declaration /
    prototype, no body). Returns the line index of the "{", or None if no body.
    String/char/comment contents are ignored throughout.
    """
    chars = _cpp_significant_chars(lines, i)
    paren = 0
    opened = False
    # Phase 1: balance the parameter-list parens.
    for _li, ch in chars:
        if ch == "(":
            paren += 1
            opened = True
        elif ch == ")":
            paren -= 1
        if opened and paren == 0:
            break
    if not (opened and paren == 0):
        return None
    # Phase 2: resume the same scan. The first "{" opens the body; a ";" means
    # it was only a declaration. Qualifiers (const/override/noexcept/trailing-
    # return/init-list) carry no ";" so the first "{" wins.
    for li, ch in chars:
        if ch == "{":
            return li
        if ch == ";":
            return None
    return None


def _extract_cpp(source: str) -> list[FunctionTarget]:
    lines = source.splitlines()
    targets: list[FunctionTarget] = []
    seen: set[str] = set()
    i = 0
    n = len(lines)
    while i < n:
        m = _CPP_FN_HEAD_RE.match(lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        if name.split("::")[-1] in _CPP_CONTROL:
            i += 1
            continue
        bo = _cpp_signature_brace(lines, i)
        if bo is None:
            i += 1
            continue
        # Brace-count the body (string/char/comment-aware) from the "{" line
        # until depth returns to 0 — that line holds the closing brace.
        depth = 0
        close_li = bo
        for li, ch in _cpp_significant_chars(lines, bo):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    close_li = li
                    break
        body = lines[bo + 1 : close_li]
        if len(body) >= MIN_BODY_LINES and name not in seen:
            seen.add(name)
            targets.append(
                FunctionTarget(
                    name=name,
                    start_line=bo + 2,
                    body_lines=body,
                )
            )
        i = close_li + 1 if close_li > bo else bo + 1
    return targets


# --- Sampling -----------------------------------------------------------------


def stratified_sample(
    targets: list[FunctionTarget],
    total_lines: int,
    k: int = 16,
    seed: int = 42,
) -> list[FunctionTarget]:
    """Sample k targets spread across file position — tests recall at all depths, not just the tail."""
    if len(targets) <= k:
        return list(targets)
    targets = sorted(targets, key=lambda t: t.start_line)
    rng = random.Random(seed)
    buckets: list[list[FunctionTarget]] = [[] for _ in range(k)]
    for t in targets:
        idx = min(k - 1, (t.start_line * k) // max(1, total_lines))
        buckets[idx].append(t)
    chosen: list[FunctionTarget] = []
    for b in buckets:
        if b:
            chosen.append(rng.choice(b))
    chosen_names = {t.name for t in chosen}
    pool = [t for t in targets if t.name not in chosen_names]
    rng.shuffle(pool)
    while len(chosen) < k and pool:
        chosen.append(pool.pop())
    return chosen[:k]
