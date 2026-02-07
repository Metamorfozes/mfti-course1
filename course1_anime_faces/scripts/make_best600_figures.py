from __future__ import annotations

from pathlib import Path
import re
from typing import List

from PIL import Image, ImageOps, ImageDraw


RUN_ROOT = Path("course1_anime_faces/results/B_run_1024_fast_sanity")
FINAL_ROOT = Path("course1_anime_faces/final/B_run_1024_fast_sanity_best600")
SAMPLES_DIR = FINAL_ROOT / "samples_best"
FIGURES_DIR = FINAL_ROOT / "figures"

TARGET_STEPS = [300, 400, 500, 600, 650, 700]


def parse_step(path: Path) -> int | None:
    m = re.match(r"^dst_(\d{6})\.jpg$", path.name)
    if not m:
        return None
    return int(m.group(1))


def list_available_samples() -> List[Path]:
    candidates = [p for p in (RUN_ROOT / "sample").glob("dst_*.jpg") if p.is_file()]
    return sorted(candidates, key=lambda p: parse_step(p) or -1)


def nearest_by_step(available: List[Path], target: int) -> Path:
    return min(
        available,
        key=lambda p: (abs((parse_step(p) or 0) - target), (parse_step(p) or 0)),
    )


def ensure_selected_samples() -> List[Path]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    available = list_available_samples()
    if not available:
        raise FileNotFoundError(f"No sample images found in {(RUN_ROOT / 'sample').as_posix()}")

    selected: List[Path] = []
    for step in TARGET_STEPS:
        src = nearest_by_step(available, step)
        dst = SAMPLES_DIR / src.name
        Image.open(src).save(dst)
        selected.append(dst)
    return selected


def make_grid(images: List[Path], out_path: Path, cols: int = 3) -> None:
    opened = [Image.open(p).convert("RGB") for p in images]
    if not opened:
        raise ValueError("No images to compose")

    tile_w = min(img.width for img in opened)
    tile_h = min(img.height for img in opened)
    rows = (len(opened) + cols - 1) // cols

    margin = 20
    label_h = 36
    canvas_w = cols * tile_w + (cols + 1) * margin
    canvas_h = rows * (tile_h + label_h) + (rows + 1) * margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for idx, (img, path) in enumerate(zip(opened, images)):
        row = idx // cols
        col = idx % cols
        x = margin + col * (tile_w + margin)
        y = margin + row * (tile_h + label_h + margin)

        fitted = ImageOps.fit(img, (tile_w, tile_h), method=Image.Resampling.LANCZOS)
        canvas.paste(fitted, (x, y))

        step = parse_step(path)
        label = f"step {step:06d}" if step is not None else path.name
        draw.text((x, y + tile_h + 8), label, fill=(20, 20, 20))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_best_single(selected: List[Path], out_path: Path) -> Path:
    available = sorted(selected, key=lambda p: parse_step(p) or -1)
    best = min(available, key=lambda p: (abs((parse_step(p) or 0) - 600), (parse_step(p) or 0)))

    image = Image.open(best).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return best


def main() -> None:
    selected = ensure_selected_samples()
    selected = sorted(selected, key=lambda p: parse_step(p) or -1)

    grid_path = FIGURES_DIR / "grid_best_steps.png"
    best_path = FIGURES_DIR / "best_single.png"

    make_grid(selected, grid_path, cols=3)
    best_src = make_best_single(selected, best_path)

    selected_names = ", ".join(p.name for p in selected)
    print(f"Selected samples: {selected_names}")
    print(f"Best single source: {best_src.name}")
    print(f"Saved: {grid_path.as_posix()}")
    print(f"Saved: {best_path.as_posix()}")


if __name__ == "__main__":
    main()
