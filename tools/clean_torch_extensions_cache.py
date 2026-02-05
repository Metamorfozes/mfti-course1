import os
import shutil
from pathlib import Path


def main() -> int:
    user_profile = os.environ.get("USERPROFILE", "")
    if not user_profile:
        print("ERROR: USERPROFILE is not set.")
        return 1

    cache_dir = Path(user_profile) / ".cache" / "torch_extensions"

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Removed: {cache_dir}")
    else:
        print(f"Not found: {cache_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
