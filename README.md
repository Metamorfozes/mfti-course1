# CLIP-Guided Anime Adaptation with StyleGAN-NADA (ZSSGAN)

This project adapts a pretrained FFHQ StyleGAN2 face generator to anime portraits in a zero-shot setting using CLIP guidance. The goal is to transfer anime style without target-domain paired training data while preserving facial structure coherence.

## Method Overview

The pipeline follows **StyleGAN-NADA / ZSSGAN**: a frozen source generator anchors identity/structure, and a trainable copy is optimized with CLIP text-image alignment toward the target domain.

## Final Run Summary

- Selected run: [`course1_anime_faces/results/B_run_1024_fast_sanity`](course1_anime_faces/results/B_run_1024_fast_sanity)
- Final artifact pack: [`course1_anime_faces/final/B_run_1024_fast_sanity_best600/`](course1_anime_faces/final/B_run_1024_fast_sanity_best600/)
- Resolution: `1024x1024`
- Iterations: `1200`
- Best checkpoint: [`checkpoint/000600.pt`](course1_anime_faces/final/B_run_1024_fast_sanity_best600/checkpoint/000600.pt)
- Target prompt: `"anime portrait, clean lineart, cel shading, high quality"`

## Submission Links

- Final report: [`course1_anime_faces/final/B_run_1024_fast_sanity_best600/REPORT.md`](course1_anime_faces/final/B_run_1024_fast_sanity_best600/REPORT.md)
- Synthetic edit grid: [`course1_anime_faces/final/edits_synthetic/grid.png`](course1_anime_faces/final/edits_synthetic/grid.png)
- Real edit grid 1: [`course1_anime_faces/final/edits_real/grids/real_01_grid.png`](course1_anime_faces/final/edits_real/grids/real_01_grid.png)
- Real edit grid 2: [`course1_anime_faces/final/edits_real/grids/real_02_grid.png`](course1_anime_faces/final/edits_real/grids/real_02_grid.png)

## Scoring Checklist Coverage

- Training pipeline:
- Generated images editing (synthetic):
- Real image editing:
- Blocks study (mapping vs lastblocks):
- Report + GitHub:

## Results

### Best Single Sample

[![Best single sample](course1_anime_faces/final/B_run_1024_fast_sanity_best600/figures/best_single.png)](course1_anime_faces/final/B_run_1024_fast_sanity_best600/figures/best_single.png)

### Progression / Checkpoint Grid

[![Best-steps grid](course1_anime_faces/final/B_run_1024_fast_sanity_best600/figures/grid_best_steps.png)](course1_anime_faces/final/B_run_1024_fast_sanity_best600/figures/grid_best_steps.png)

## Relevant Repository Structure

```text
.
|-- README.md
|-- run_env.cmd
|-- run_patch.cmd
|-- run_train_anime_mapping.cmd
|-- run_train_anime_lastblocks.cmd
`-- course1_anime_faces/
    |-- models/
    |   `-- stylegan2-ffhq-config-f.pt
    |-- results/
    |   `-- B_run_1024_fast_sanity/
    |       |-- checkpoint/
    |       |   `-- 000600.pt
    |       `-- sample/
    `-- final/
        |-- B_run_1024_fast_sanity_best600/
        |   |-- REPORT.md
        |   |-- checkpoint/
        |   |   `-- 000600.pt
        |   |-- figures/
        |   |   |-- best_single.png
        |   |   `-- grid_best_steps.png
        |   `-- samples_best/
        |-- edits_synthetic/
        |   `-- grid.png
        `-- edits_real/
            |-- inputs/
            `-- grids/
                |-- real_01_grid.png
                `-- real_02_grid.png
```

## How to Reproduce (Minimal, Windows)

If environment variables/dependencies are not initialized in your current shell, run:

```bat
run_env.cmd
```

Then run patching and training scripts:

```bat
run_patch.cmd
run_train_anime_mapping.cmd
run_train_anime_lastblocks.cmd
```

For final submission artifacts, use:
[`course1_anime_faces/final/B_run_1024_fast_sanity_best600/`](course1_anime_faces/final/B_run_1024_fast_sanity_best600/)

## Block Freezing Study (Mapping vs Last Blocks)

The repository contains two launchers for this scoring item: [`run_train_anime_mapping.cmd`](run_train_anime_mapping.cmd) and [`run_train_anime_lastblocks.cmd`](run_train_anime_lastblocks.cmd). In the committed artifacts, strategy-specific output folders expected from those launchers (`course1_anime_faces/results/anime_mapping/`, `course1_anime_faces/results/anime_lastblocks/`) are not present, and saved `args.json` files in existing runs record `phase: null`. Therefore, we report this item conservatively using available qualitative outputs from committed runs: [`course1_anime_faces/results/A_run_1024_iter1200/sample/`](course1_anime_faces/results/A_run_1024_iter1200/sample/), [`course1_anime_faces/results/B_run_1024_fast_sanity/sample/`](course1_anime_faces/results/B_run_1024_fast_sanity/sample/), and the final package figures/samples in [`course1_anime_faces/final/B_run_1024_fast_sanity_best600/`](course1_anime_faces/final/B_run_1024_fast_sanity_best600/), including the side-by-side references documented in the report section [`course1_anime_faces/final/B_run_1024_fast_sanity_best600/REPORT.md`](course1_anime_faces/final/B_run_1024_fast_sanity_best600/REPORT.md).

Final submission decision: we use [`course1_anime_faces/final/B_run_1024_fast_sanity_best600/checkpoint/000600.pt`](course1_anime_faces/final/B_run_1024_fast_sanity_best600/checkpoint/000600.pt), which is the selected best qualitative checkpoint from the committed run artifacts.

