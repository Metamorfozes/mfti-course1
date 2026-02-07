# CLIP-Guided Anime Adaptation with StyleGAN-NADA (ZSSGAN)

This project adapts a pretrained FFHQ StyleGAN2 face generator to anime portraits in a zero-shot setting using CLIP guidance. The objective is to transfer anime style without target-domain paired training data, while keeping facial structure coherent.

## Method Overview

The method follows **StyleGAN-NADA / ZSSGAN**: a frozen source generator anchors identity and structure, while a trainable copy is optimized with CLIP-based text-image alignment toward the target concept. This is CLIP-guided zero-shot domain adaptation in latent/style space.

## Final Run Summary

- Selected run: `course1_anime_faces/results/B_run_1024_fast_sanity`
- Final artifact pack: `course1_anime_faces/final/B_run_1024_fast_sanity_best600/`
- Resolution: `1024x1024`
- Iterations: `1200`
- Best checkpoint: `checkpoint/000600.pt`
- Target prompt: `"anime portrait, clean lineart, cel shading, high quality"`
- Detailed report: `course1_anime_faces/final/B_run_1024_fast_sanity_best600/REPORT.md`

## Results

### Best Single Sample

![Best single sample](course1_anime_faces/final/B_run_1024_fast_sanity_best600/figures/best_single.png)

### Progression / Checkpoint Grid

![Best-steps grid](course1_anime_faces/final/B_run_1024_fast_sanity_best600/figures/grid_best_steps.png)

### Selected Best Checkpoint Sample

![Checkpoint 000600 sample](course1_anime_faces/final/B_run_1024_fast_sanity_best600/samples_best/dst_000600.jpg)

## Relevant Repository Structure

```text
.
+- README.md
+- run_env.cmd
+- run_train_anime_mapping.cmd
+- run_train_anime_lastblocks.cmd
L- course1_anime_faces/
   +- results/
   ¦  L- B_run_1024_fast_sanity/
   ¦     +- checkpoint/000600.pt
   ¦     L- sample/
   L- final/
      L- B_run_1024_fast_sanity_best600/
         +- REPORT.md
         +- checkpoint/000600.pt
         +- figures/
         L- samples_best/
```

## How to Reproduce (Minimal)

```bat
run_patch.cmd
run_train_anime_mapping.cmd
run_train_anime_lastblocks.cmd
```

For the final submission figures/checkpoint package, use the prepared artifacts in:
`course1_anime_faces/final/B_run_1024_fast_sanity_best600/`.
