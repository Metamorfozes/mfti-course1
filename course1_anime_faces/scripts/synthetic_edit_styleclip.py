from __future__ import annotations

import argparse
import json
import random
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]

STYLEGAN_NADA_ROOT = REPO_ROOT / "external" / "stylegan-nada"
ZSSGAN_DIR = STYLEGAN_NADA_ROOT / "ZSSGAN"

for p in [STYLEGAN_NADA_ROOT, ZSSGAN_DIR]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
DEFAULT_OUT_DIR = REPO_ROOT / "course1_anime_faces" / "final" / "edits_synthetic"

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StyleCLIP-like synthetic image edits with a trained StyleGAN-NADA checkpoint.")
    parser.add_argument("--run_dir", required=True, help="Run directory with args.json and checkpoint/*.pt.")
    parser.add_argument("--ckpt", default="000600.pt", help="Checkpoint file name inside run_dir/checkpoint or full path.")
    parser.add_argument("--size", type=int, default=1024, help="Generator output size.")
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps per edit.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducibility.")
    parser.add_argument("--prompt_blue", default="anime portrait with blue hair")
    parser.add_argument("--prompt_glasses", default="anime portrait with glasses")
    parser.add_argument("--source_prompt", default="", help="Optional source prompt, defaults to run args source_class.")
    parser.add_argument("--truncation", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for latent optimization.")
    parser.add_argument("--lambda_w", type=float, default=0.02, help="L2 regularization weight for W distance.")
    parser.add_argument("--lambda_img", type=float, default=0.03, help="L2 regularization weight for image distance.")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_checkpoint(run_dir: Path, ckpt_arg: str) -> Path:
    candidate = Path(ckpt_arg)
    if candidate.is_file():
        return candidate.resolve()
    ckpt_path = run_dir / "checkpoint" / ckpt_arg
    if ckpt_path.is_file():
        return ckpt_path.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg}")


def clip_preprocess_tensor(img_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Input image tensor expected in [-1, 1]
    x = (img_tensor + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
    mean = CLIP_MEAN.to(device)
    std = CLIP_STD.to(device)
    x = (x - mean) / std
    return x


class ClipBackend:
    def __init__(self, device: torch.device):
        self.device = device
        self.kind = ""
        self.model = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            import clip  # type: ignore

            model, _ = clip.load("ViT-B/32", device=str(self.device), jit=False)
            model.eval()
            self.kind = "clip"
            self.model = model
            self.tokenizer = clip.tokenize
            return
        except Exception:
            pass

        try:
            import open_clip  # type: ignore

            model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=str(self.device))
            model.eval()
            self.kind = "open_clip"
            self.model = model
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            return
        except Exception as e:
            raise RuntimeError("Neither `clip` nor `open_clip` is available.") from e

    def encode_text(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer([prompt]).to(self.device)  # type: ignore[misc]
        with torch.no_grad():
            txt = self.model.encode_text(tokens)
            txt = txt / txt.norm(dim=-1, keepdim=True)
        return txt

    def image_text_similarity(self, image_tensor: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        image_input = clip_preprocess_tensor(image_tensor, self.device)
        img_feat = self.model.encode_image(image_input)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat * text_features).sum(dim=-1)
        return sim


def pil_from_tensor(img: torch.Tensor) -> Image.Image:
    img = img.detach().cpu().clamp(-1, 1)
    img = ((img + 1.0) * 127.5).to(torch.uint8)
    arr = img.permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def make_three_col_grid(base: Image.Image, blue: Image.Image, glasses: Image.Image, out_path: Path) -> None:
    images = [base.convert("RGB"), blue.convert("RGB"), glasses.convert("RGB")]
    labels = ["base", "blue hair", "glasses"]
    tile_w = min(i.width for i in images)
    tile_h = min(i.height for i in images)
    margin = 16
    label_h = 28
    canvas_w = 3 * tile_w + 4 * margin
    canvas_h = tile_h + 2 * margin + label_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for idx, (im, lbl) in enumerate(zip(images, labels)):
        x = margin + idx * (tile_w + margin)
        y = margin
        canvas.paste(im.resize((tile_w, tile_h), Image.Resampling.LANCZOS), (x, y))
        draw.text((x, y + tile_h + 6), lbl, fill=(25, 25, 25))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def optimize_edit(
    generator: Any,
    mean_latent: torch.Tensor,
    w_base: torch.Tensor,
    base_img: torch.Tensor,
    text_feat: torch.Tensor,
    truncation: float,
    steps: int,
    lr: float,
    lambda_w: float,
    lambda_img: float,
    clip_backend: ClipBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    w_opt = w_base.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([w_opt], lr=lr)

    for _ in range(steps):
        edited_img, _ = generator(
            [w_opt],
            input_is_latent=True,
            truncation=truncation,
            truncation_latent=mean_latent,
            randomize_noise=False,
        )
        sim = clip_backend.image_text_similarity(edited_img, text_feat)
        loss_clip = -sim.mean()
        loss_w = F.mse_loss(w_opt, w_base)
        loss_img = F.mse_loss(edited_img, base_img)
        loss = loss_clip + lambda_w * loss_w + lambda_img * loss_img

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_img, _ = generator(
        [w_opt.detach()],
        input_is_latent=True,
        truncation=truncation,
        truncation_latent=mean_latent,
        randomize_noise=False,
    )
    return w_opt.detach(), final_img.detach()


def load_generator_from_ckpt(ckpt_path: Path, size: int, device: torch.device):
    stylegan_root = REPO_ROOT / "external" / "stylegan-nada"
    if str(stylegan_root) not in sys.path:
        sys.path.insert(0, str(stylegan_root))
    from ZSSGAN.model.sg2_model import Generator  # type: ignore

    generator = Generator(size, 512, 8, channel_multiplier=2).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if "g_ema" not in state:
        raise KeyError("Checkpoint does not contain `g_ema`.")
    generator.load_state_dict(state["g_ema"], strict=True)
    generator.eval()
    with torch.no_grad():
        mean_latent = generator.mean_latent(4096)
    return generator, mean_latent


def build_exact_command(args: argparse.Namespace) -> str:
    parts = ["python", "course1_anime_faces/scripts/synthetic_edit_styleclip.py"]
    for k, v in vars(args).items():
        parts.append(f"--{k}")
        parts.append(str(v))
    return " ".join(shlex.quote(p) for p in parts)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    run_args_path = run_dir / "args.json"
    if not run_args_path.exists():
        raise FileNotFoundError(f"args.json not found in run_dir: {run_dir}")

    run_args = load_json(run_args_path)
    ckpt_path = resolve_checkpoint(run_dir, args.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA is not available. Running on CPU may be slow.")

    set_seed(args.seed)
    clip_backend = ClipBackend(device)

    generator, mean_latent = load_generator_from_ckpt(ckpt_path, args.size, device)
    source_prompt = args.source_prompt.strip() or str(run_args.get("source_class", "photo of a face"))

    z_gen = torch.Generator(device=device)
    z_gen.manual_seed(args.seed)
    z = torch.randn(1, 512, generator=z_gen, device=device)

    with torch.no_grad():
        w_base = generator.style(z)
        base_img, _ = generator(
            [w_base],
            input_is_latent=True,
            truncation=args.truncation,
            truncation_latent=mean_latent,
            randomize_noise=False,
        )

    text_blue = clip_backend.encode_text(args.prompt_blue)
    text_glasses = clip_backend.encode_text(args.prompt_glasses)

    _, img_blue = optimize_edit(
        generator=generator,
        mean_latent=mean_latent,
        w_base=w_base,
        base_img=base_img,
        text_feat=text_blue,
        truncation=args.truncation,
        steps=args.steps,
        lr=args.lr,
        lambda_w=args.lambda_w,
        lambda_img=args.lambda_img,
        clip_backend=clip_backend,
    )

    _, img_glasses = optimize_edit(
        generator=generator,
        mean_latent=mean_latent,
        w_base=w_base,
        base_img=base_img,
        text_feat=text_glasses,
        truncation=args.truncation,
        steps=args.steps,
        lr=args.lr,
        lambda_w=args.lambda_w,
        lambda_img=args.lambda_img,
        clip_backend=clip_backend,
    )

    base_pil = pil_from_tensor(base_img[0])
    blue_pil = pil_from_tensor(img_blue[0])
    glasses_pil = pil_from_tensor(img_glasses[0])

    base_path = out_dir / "base.png"
    blue_path = out_dir / "edit_blue_hair.png"
    glasses_path = out_dir / "edit_glasses.png"
    grid_path = out_dir / "grid.png"
    meta_path = out_dir / "meta.json"

    base_pil.save(base_path)
    blue_pil.save(blue_path)
    glasses_pil.save(glasses_path)
    make_three_col_grid(base_pil, blue_pil, glasses_pil, grid_path)

    meta = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "size": args.size,
        "seed": args.seed,
        "steps": args.steps,
        "truncation": args.truncation,
        "lr": args.lr,
        "lambda_w": args.lambda_w,
        "lambda_img": args.lambda_img,
        "source_prompt": source_prompt,
        "prompt_blue": args.prompt_blue,
        "prompt_glasses": args.prompt_glasses,
        "clip_backend": clip_backend.kind,
        "cli_args": vars(args),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cmd = build_exact_command(args)
    print(f"Output folder: {out_dir}")
    print(f"Exact command: {cmd}")
    print("Success: synthetic edits generated (base, blue hair, glasses, grid, meta.json).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
