import re
from pathlib import Path

FLAG = "--allow-unsupported-compiler"


def _find_matching_paren(text: str, start_paren: int) -> int | None:
    depth = 0
    i = start_paren
    in_str = None
    escape = False

    while i < len(text):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
        else:
            if ch in ("'", '"'):
                in_str = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1

    return None


def _add_flag_to_list(list_text: str) -> tuple[str, bool]:
    if FLAG in list_text:
        return list_text, False

    stripped = list_text.rstrip()
    if stripped.endswith("]"):
        insert_at = stripped.rfind("]")
        prefix = stripped[:insert_at].rstrip()
        if prefix.endswith("["):
            new_text = stripped[:insert_at] + f"'{FLAG}'" + stripped[insert_at:]
        else:
            new_text = stripped[:insert_at] + f", '{FLAG}'" + stripped[insert_at:]
        return new_text, True

    return list_text, False


def _patch_call_text(call_text: str) -> tuple[str, bool]:
    if "extra_cuda_cflags" in call_text:
        list_match = re.search(
            r"(extra_cuda_cflags\s*=\s*)(\[[^\]]*?\])",
            call_text,
            flags=re.S,
        )
        if list_match:
            list_text = list_match.group(2)
            new_list_text, changed = _add_flag_to_list(list_text)
            if changed:
                new_call_text = (
                    call_text[: list_match.start(2)]
                    + new_list_text
                    + call_text[list_match.end(2):]
                )
                return new_call_text, True
        return call_text, False

    insert_text = f"extra_cuda_cflags=['{FLAG}']"
    insert_at = call_text.rfind(")")
    inner = call_text[call_text.find("(") + 1: insert_at].strip()
    if inner == "" or inner.endswith(","):
        sep = ""
    else:
        sep = ", "
    new_call_text = call_text[:insert_at] + sep + insert_text + call_text[insert_at:]
    return new_call_text, True


def _patch_load_calls(text: str) -> tuple[str, int, int]:
    pattern = re.compile(r"\btorch\.utils\.cpp_extension\.load\s*\(")
    replacements = []
    found = 0
    modified = 0

    for match in pattern.finditer(text):
        start_paren = match.end() - 1
        end_paren = _find_matching_paren(text, start_paren)
        if end_paren is None:
            continue

        found += 1
        call_text = text[match.start(): end_paren + 1]
        new_call_text, changed = _patch_call_text(call_text)
        if changed:
            modified += 1
            replacements.append((match.start(), end_paren + 1, new_call_text))

    if not replacements:
        return text, found, modified

    new_text = text
    for start, end, new_call_text in reversed(replacements):
        new_text = new_text[:start] + new_call_text + new_text[end:]

    return new_text, found, modified


def _flag_present(text: str) -> bool:
    return FLAG in text


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    op_dir = repo_root / "external" / "stylegan-nada" / "ZSSGAN" / "op"

    if not op_dir.is_dir():
        print(f"ERROR: ops directory not found: {op_dir}")
        return 1

    py_files = sorted(op_dir.glob("*.py"))
    if not py_files:
        print(f"ERROR: no Python files found in: {op_dir}")
        return 1

    patched_files = []
    total_found = 0
    total_modified = 0
    target_files = ["fused_act.py", "upfirdn2d.py"]
    missing_targets = []
    targets_already_patched = True

    for name in target_files:
        target_path = op_dir / name
        if not target_path.is_file():
            missing_targets.append(str(target_path))
            targets_already_patched = False
            continue
        if not _flag_present(target_path.read_text(encoding="utf-8")):
            targets_already_patched = False

    if targets_already_patched:
        print("Already patched.")
        return 0

    for path in py_files:
        original = path.read_text(encoding="utf-8")
        updated, found, modified = _patch_load_calls(original)
        total_found += found
        total_modified += modified
        if modified:
            path.write_text(updated, encoding="utf-8")
            patched_files.append((str(path), modified))

    if missing_targets:
        print("ERROR: missing required files:")
        for path in missing_targets:
            print(f"  {path}")
        return 1

    if total_modified == 0:
        print("ERROR: no load() calls were modified.")
        return 2

    for name in target_files:
        target_path = op_dir / name
        if not _flag_present(target_path.read_text(encoding="utf-8")):
            print(f"ERROR: flag missing after patch: {target_path}")
            return 3

    print("Patched files:")
    for path, count in patched_files:
        print(f"  {path} (modified {count} load() call(s))")
    print(f"Total modified load() calls: {total_modified}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
