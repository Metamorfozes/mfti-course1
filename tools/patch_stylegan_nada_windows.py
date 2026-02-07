import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    REPO_ROOT / "external" / "stylegan-nada" / "ZSSGAN" / "op" / "fused_act.py",
    REPO_ROOT / "external" / "stylegan-nada" / "ZSSGAN" / "op" / "upfirdn2d.py",
]

FILE_UTILS = REPO_ROOT / "external" / "stylegan-nada" / "ZSSGAN" / "utils" / "file_utils.py"


def patch_load_call(text: str) -> tuple[str, int]:
    """
    Add extra_cuda_cflags=['--allow-unsupported-compiler'] into load(...) calls
    if not present. Works even if load call spans multiple lines.
    """
    modified = 0

    # Find each 'load(' occurrence and its matching ')'
    idx = 0
    while True:
        start = text.find("load(", idx)
        if start == -1:
            break

        # Walk forward to match parentheses for this call
        i = start + len("load(")
        depth = 1
        in_str = False
        str_ch = ""
        while i < len(text) and depth > 0:
            ch = text[i]
            if in_str:
                if ch == str_ch and text[i - 1] != "\\":
                    in_str = False
            else:
                if ch in ("'", '"'):
                    in_str = True
                    str_ch = ch
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
            i += 1

        end = i  # position after ')'
        call = text[start:end]

        if "extra_cuda_cflags" not in call:
            # Insert before the last ')'
            insert = ",\n    extra_cuda_cflags=['--allow-unsupported-compiler']"
            # try to keep indentation if call already multi-line
            if "\n" in call:
                patched_call = call[:-1] + insert + "\n)"
            else:
                patched_call = call[:-1] + insert + ")"
            text = text[:start] + patched_call + text[end:]
            modified += 1
            idx = start + len(patched_call)
        else:
            idx = end

    return text, modified


def patch_file_utils(text: str) -> tuple[str, int]:
    """
    torchvision.utils.make_grid signature changed: 'range' -> 'value_range'
    Patch save_image call accordingly if needed.
    """
    modified = 0
    # replace "range=(" kwarg inside save_image/make_grid usage
    if re.search(r"\brange\s*=\s*\(", text):
        text = re.sub(r"\brange\s*=\s*\(", "value_range=(", text)
        modified += 1
    return text, modified


def main() -> int:
    total_mods = 0

    for p in TARGETS:
        if not p.exists():
            print(f"Not found: {p}")
            continue
        src = p.read_text(encoding="utf-8")
        dst, mods = patch_load_call(src)
        if mods:
            p.write_text(dst, encoding="utf-8")
        total_mods += mods
        print(f"{p}: load() patched = {mods}")

    if FILE_UTILS.exists():
        src = FILE_UTILS.read_text(encoding="utf-8")
        dst, mods = patch_file_utils(src)
        if mods:
            FILE_UTILS.write_text(dst, encoding="utf-8")
        total_mods += mods
        print(f"{FILE_UTILS}: file_utils patched = {mods}")
    else:
        print(f"Not found: {FILE_UTILS}")

    if total_mods == 0:
        print("ERROR: no load() calls were modified.")
        return 2

    print(f"OK: total patches applied = {total_mods}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
