# Project Spec

## Project goal

CLIP-guided domain adaptation of a pretrained StyleGAN2 (FFHQ) towards anime-face style.

## Datasets

- Source: FFHQ (pretrained StyleGAN2, plus optional real face samples for inversion/editing)
- Target: Anime Face Dataset (used for evaluation / reference)

## Minimum deliverables (baseline)

- A reproducible script to generate baseline images from pretrained StyleGAN2 (FFHQ).
- A training script placeholder for CLIP-guided adaptation (StyleGAN-NADA-like).
- An evaluation script placeholder that computes at least CLIP similarity to an "anime face" prompt.
- Result artifacts saved to results/ (image grids + a simple metrics.json).

## Experiments (minimal)

- Fine-tune only mapping network vs fine-tune last synthesis blocks (2 settings).

## Definition of Done

- [ ] Baseline generation script runs end-to-end and produces images.
- [ ] CLIP-guided adaptation placeholder is in place with clear TODOs.
- [ ] Evaluation placeholder computes CLIP similarity to an "anime face" prompt.
- [ ] results/ contains image grids and a metrics.json example.
- [ ] README and docs reflect the current state and how to run each step.
