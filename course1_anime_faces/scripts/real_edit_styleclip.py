from __future__ import annotations

import argparse
import json
import random
import shlex
import sys
from pathlib import Path
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

DEFAULT_INPUTS_DIR = REPO_ROOT / "course1_anime_faces" / "final" / "edits_real" / "inputs"
DEFAULT_RUN_DIR = REPO_ROOT / "course1_anime_faces" / "results" / "B_run_1024_fast_sanity"
DEFAULT_OUT_DIR = REPO_ROOT / "course1_anime_faces" / "final" / "edits_real"
DEFAULT_FFHQ_CKPT = REPO_ROOT / "course1_anime_faces" / "models" / "stylegan2-ffhq-config-f.pt"

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

BASE_PROMPT = "anime portrait"
PROMPT_BLUE = "anime portrait with blue hair"
PROMPT_GLASSES = "anime portrait with glasses"

INV_LR = 0.05
EDIT_LR = 0.05
TRUNCATION = 0.7
INV_CLIP_W = 1.0
INV_PIX_W = 0.1
EDIT_W_L2 = 0.03
EDIT_IMG_W = 0.03
EDIT_BASE_W = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real image inversion + CLIP-guided edits with StyleGAN2.")
    parser.add_argument("--inputs_dir", default=str(DEFAULT_INPUTS_DIR))
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--ckpt", default="000600.pt")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--inv_steps", type=int, default=320)
    parser.add_argument("--edit_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(run_dir: Path, ckpt_arg: str) -> Path:
    candidate = Path(ckpt_arg)
    if candidate.is_file():
        return candidate.resolve()
    ckpt_path = run_dir / "checkpoint" / ckpt_arg
    if ckpt_path.is_file():
        return ckpt_path.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg}")


def find_inputs(inputs_dir: Path) -> list[Path]:
    if not inputs_dir.exists():
        raise FileNotFoundError(f"inputs_dir not found: {inputs_dir}")
    items = sorted(inputs_dir.glob("*.png"))
    if not items:
        raise FileNotFoundError(f"No .png files found in: {inputs_dir}")
    return items


def center_crop_resize(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    image = image.crop((left, top, left + s, top + s))
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def tensor_from_pil(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def pil_from_tensor(img: torch.Tensor) -> Image.Image:
    img = img.detach().cpu().clamp(-1.0, 1.0)
    img = ((img + 1.0) * 127.5).to(torch.uint8)
    arr = img.permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def clip_norm_image(img: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = (img + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
    mean = CLIP_MEAN.to(device)
    std = CLIP_STD.to(device)
    return (x - mean) / std


def image_256(img: torch.Tensor) -> torch.Tensor:
    x = (img + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    return F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)


class ClipModel:
    def __init__(self, device: torch.device):
        self.device = device
        try:
            import clip  # type: ignore
        except Exception as e:
            raise RuntimeError("Package `clip` is required.") from e
        model, _ = clip.load("ViT-B/32", device=str(device), jit=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.tokenize = clip.tokenize

    def encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenize([text]).to(self.device)
        with torch.no_grad():
            t = self.model.encode_text(tokens)
            t = t / t.norm(dim=-1, keepdim=True)
        return t

    def encode_image(self, image_m1_1: torch.Tensor) -> torch.Tensor:
        x = clip_norm_image(image_m1_1, self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat


def load_generator(size: int, ckpt_path: Path, device: torch.device, strict: bool = True):
    from ZSSGAN.model.sg2_model import Generator  # type: ignore

    g = Generator(size, 512, 8, channel_multiplier=2).to(device)
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "g_ema" in state:
        sd = state["g_ema"]
    elif isinstance(state, dict) and "generator" in state:
        sd = state["generator"]
    else:
        sd = state

    g.load_state_dict(sd, strict=strict)
    g.eval()
    with torch.no_grad():
        w_avg = g.mean_latent(4096)
    return g, w_avg


def render_from_w(generator: Any, w: torch.Tensor, w_avg: torch.Tensor, truncation: float) -> torch.Tensor:
    image, _ = generator(
        [w],
        input_is_latent=True,
        truncation=truncation,
        truncation_latent=w_avg,
        randomize_noise=False,
    )
    return image


def invert_to_w(
    generator: Any,
    clip_model: ClipModel,
    target_img: torch.Tensor,
    w_avg: torch.Tensor,
    inv_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        target_feat = clip_model.encode_image(target_img)
        target_pix = image_256(target_img)

    w = w_avg.detach().clone().requires_grad_(True)
    optim = torch.optim.Adam([w], lr=INV_LR)

    for _ in range(inv_steps):
        pred = render_from_w(generator, w, w_avg, TRUNCATION)
        pred_feat = clip_model.encode_image(pred)
        pred_pix = image_256(pred)

        loss_clip = (1.0 - (pred_feat * target_feat).sum(dim=-1)).mean()
        loss_pix = F.mse_loss(pred_pix, target_pix)
        loss = INV_CLIP_W * loss_clip + INV_PIX_W * loss_pix

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    with torch.no_grad():
        recon = render_from_w(generator, w.detach(), w_avg, TRUNCATION)
    return w.detach(), recon.detach()


def optimize_edit(
    generator: Any,
    clip_model: ClipModel,
    w_start: torch.Tensor,
    base_img: torch.Tensor,
    w_avg: torch.Tensor,
    target_txt: torch.Tensor,
    base_txt: torch.Tensor,
    edit_steps: int,
) -> torch.Tensor:
    w_opt = w_start.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([w_opt], lr=EDIT_LR)

    base_ref = image_256(base_img).detach()

    for _ in range(edit_steps):
        edited = render_from_w(generator, w_opt, w_avg, TRUNCATION)
        img_feat = clip_model.encode_image(edited)
        sim_target = (img_feat * target_txt).sum(dim=-1).mean()
        sim_base = (img_feat * base_txt).sum(dim=-1).mean()

        loss_text = -(sim_target - EDIT_BASE_W * sim_base)
        loss_w = F.mse_loss(w_opt, w_start)
        loss_img = F.mse_loss(image_256(edited), base_ref)
        loss = loss_text + EDIT_W_L2 * loss_w + EDIT_IMG_W * loss_img

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return render_from_w(generator, w_opt.detach(), w_avg, TRUNCATION).detach()


def make_grid(images: list[tuple[str, Image.Image]], out_path: Path) -> None:
    cols = 3
    rows = 2
    margin = 14
    label_h = 24
    tile_w = min(img.width for _, img in images)
    tile_h = min(img.height for _, img in images)

    canvas_w = cols * tile_w + (cols + 1) * margin
    canvas_h = rows * (tile_h + label_h) + (rows + 1) * margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(246, 246, 246))
    draw = ImageDraw.Draw(canvas)

    for i, (label, image) in enumerate(images):
        r = i // cols
        c = i % cols
        x = margin + c * (tile_w + margin)
        y = margin + r * (tile_h + label_h + margin)
        canvas.paste(image.resize((tile_w, tile_h), Image.Resampling.LANCZOS), (x, y))
        draw.text((x, y + tile_h + 4), label, fill=(20, 20, 20))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_exact_command(args: argparse.Namespace) -> str:
    parts = ["python", "course1_anime_faces/scripts/real_edit_styleclip.py"]
    for k, v in vars(args).items():
        parts.extend([f"--{k}", str(v)])
    return " ".join(shlex.quote(p) for p in parts)


def main() -> int:
    args = parse_args()
    inputs_dir = Path(args.inputs_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    ffhq_ckpt = DEFAULT_FFHQ_CKPT.resolve()

    if not ffhq_ckpt.exists():
        raise FileNotFoundError(f"FFHQ checkpoint not found: {ffhq_ckpt}")
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    adapted_ckpt = resolve_checkpoint(run_dir, args.ckpt)
    input_paths = find_inputs(inputs_dir)

    outputs_dir = out_dir / "outputs"
    latents_dir = out_dir / "latents"
    grids_dir = out_dir / "grids"
    meta_dir = out_dir / "meta"
    for p in [outputs_dir, latents_dir, grids_dir, meta_dir]:
        p.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is unavailable. This script is intended for Windows + CUDA.")
    set_seed(args.seed)

    clip_model = ClipModel(device)

    ffhq_gen, ffhq_w_avg = load_generator(args.size, ffhq_ckpt, device, strict=False)
    adapted_gen, adapted_w_avg = load_generator(args.size, adapted_ckpt, device, strict=True)

    base_txt = clip_model.encode_text(BASE_PROMPT)
    blue_txt = clip_model.encode_text(PROMPT_BLUE)
    glasses_txt = clip_model.encode_text(PROMPT_GLASSES)

    exact_cmd = build_exact_command(args)

    for input_path in input_paths:
        name = input_path.stem
        real_pil = center_crop_resize(Image.open(input_path), args.size)
        real_t = tensor_from_pil(real_pil, device)

        w_inv, inv_img = invert_to_w(
            generator=ffhq_gen,
            clip_model=clip_model,
            target_img=real_t,
            w_avg=ffhq_w_avg,
            inv_steps=args.inv_steps,
        )

        with torch.no_grad():
            anime_img = render_from_w(adapted_gen, w_inv, adapted_w_avg, TRUNCATION)
        blue_img = optimize_edit(
            generator=adapted_gen,
            clip_model=clip_model,
            w_start=w_inv,
            base_img=anime_img,
            w_avg=adapted_w_avg,
            target_txt=blue_txt,
            base_txt=base_txt,
            edit_steps=args.edit_steps,
        )
        glasses_img = optimize_edit(
            generator=adapted_gen,
            clip_model=clip_model,
            w_start=w_inv,
            base_img=anime_img,
            w_avg=adapted_w_avg,
            target_txt=glasses_txt,
            base_txt=base_txt,
            edit_steps=args.edit_steps,
        )

        inv_pil = pil_from_tensor(inv_img[0])
        anime_pil = pil_from_tensor(anime_img[0])
        blue_pil = pil_from_tensor(blue_img[0])
        glasses_pil = pil_from_tensor(glasses_img[0])

        inv_path = outputs_dir / f"{name}_inverted.png"
        anime_path = outputs_dir / f"{name}_anime.png"
        blue_path = outputs_dir / f"{name}_blue_hair.png"
        glasses_path = outputs_dir / f"{name}_glasses.png"
        latent_path = latents_dir / f"{name}.pt"
        grid_path = grids_dir / f"{name}_grid.png"
        meta_path = meta_dir / f"{name}.json"

        inv_pil.save(inv_path)
        anime_pil.save(anime_path)
        blue_pil.save(blue_path)
        glasses_pil.save(glasses_path)
        torch.save({"w": w_inv.detach().cpu()}, latent_path)

        grid_items = [
            ("real", real_pil),
            ("inverted(ffhq)", inv_pil),
            ("anime", anime_pil),
            ("blue_hair", blue_pil),
            ("glasses", glasses_pil),
        ]
        make_grid(grid_items, grid_path)

        meta = {
            "input": str(input_path),
            "run_dir": str(run_dir),
            "ckpt": str(adapted_ckpt),
            "ffhq_ckpt": str(ffhq_ckpt),
            "output_files": {
                "inverted": str(inv_path),
                "anime": str(anime_path),
                "blue_hair": str(blue_path),
                "glasses": str(glasses_path),
                "latent": str(latent_path),
                "grid": str(grid_path),
            },
            "settings": {
                "size": args.size,
                "seed": args.seed,
                "inv_steps": args.inv_steps,
                "edit_steps": args.edit_steps,
                "truncation": TRUNCATION,
                "inv_lr": INV_LR,
                "edit_lr": EDIT_LR,
                "inv_clip_weight": INV_CLIP_W,
                "inv_pixel_weight": INV_PIX_W,
                "edit_w_l2_weight": EDIT_W_L2,
                "edit_img_weight": EDIT_IMG_W,
                "edit_base_prompt_weight": EDIT_BASE_W,
                "base_prompt": BASE_PROMPT,
                "edit_prompts": [PROMPT_BLUE, PROMPT_GLASSES],
            },
            "exact_command": exact_cmd,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Processed: {input_path.name}")

    print(f"Done. Outputs: {out_dir}")
    print(f"Exact command: {exact_cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
