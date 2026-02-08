# CLIP-Guided Domain Adaptation of a Face Generator to Anime Style

## 1. Project Goal

This project performs zero-shot domain adaptation of a pretrained real-face generator to an anime portrait domain. No dedicated target dataset is used for supervised training. CLIP text guidance is used to transfer anime style while preserving facial structure and identity-related characteristics.

## 2. Method Overview

The approach is based on StyleGAN-NADA (ZSSGAN). A frozen source generator serves as a structural and semantic anchor, while a trainable generator is optimized toward the target text concept using CLIP loss.

Optimization is performed over generator parameters, effectively steering outputs in latent/style space toward the target domain. The frozen reference reduces catastrophic drift, and CLIP guidance drives stylization.

## 3. Experimental Setup

- Base generator: StyleGAN2 pretrained on FFHQ.
- Adaptation method: StyleGAN-NADA / ZSSGAN.
- Resolution: 1024x1024.
- Target prompt: "anime portrait, clean lineart, cel shading, high quality".
- Key hyperparameters: `iterations = 1200`, `learning_rate = 0.001`, `lambda_patch = 0.3`, `lambda_manifold = 0.2`.
- Hardware: single consumer GPU (NVIDIA GTX 1660 Super).
- Framework: PyTorch + CUDA.

Final run directory:
- `course1_anime_faces/final/B_run_1024_fast_sanity_best600/`

Selected best checkpoint inside the final package:
- `checkpoint/000600.pt`

Adapted checkpoint used for editing pipelines:
- `course1_anime_faces/results/B_run_1024_fast_sanity/checkpoint/000600.pt`

Base FFHQ checkpoint used for inversion:
- `course1_anime_faces/models/stylegan2-ffhq-config-f.pt`

## 4. Training Process and Checkpoint Selection

Training followed iterative CLIP-guided adaptation with periodic sample export and checkpointing. As typical for CLIP-based optimization, style quality and structural stability vary across iterations.

Checkpoint selection was qualitative. Checkpoint `000600.pt` was selected by visual inspection as the best trade-off between anime stylization and facial coherence, rather than by minimum loss.

## 5. Qualitative Results

Evaluation is qualitative only.

Main artifacts for final run review:
- `course1_anime_faces/final/B_run_1024_fast_sanity_best600/figures/best_single.png`
- `course1_anime_faces/final/B_run_1024_fast_sanity_best600/figures/grid_best_steps.png`
- `course1_anime_faces/final/B_run_1024_fast_sanity_best600/samples_best/`

Observed behavior:
- Clear anime-style transfer (line emphasis and cel-shading-like appearance).
- Core facial layout remains stable in selected samples.
- The step grid supports selecting the region around step 600 as the best visual balance.

## 6. Synthetic Editing (Generated Image)

Synthetic editing (StyleCLIP-like) is applied to one fixed latent initialized from seed `42` using the adapted checkpoint `course1_anime_faces/results/B_run_1024_fast_sanity/checkpoint/000600.pt`.

For each edit prompt, the generator is frozen and only latent `w` is optimized for a short run, with regularization toward the base latent and base image.

Edit prompts:
- `"anime portrait with blue hair"`
- `"anime portrait with glasses"`

Result grid:
- `course1_anime_faces/final/edits_synthetic/grid.png`

## 7. Real Image Editing

Real-image editing is implemented as a three-stage pipeline:

1. Invert each real portrait from `course1_anime_faces/final/edits_real/inputs/*.png` into latent `w` with the frozen FFHQ generator `course1_anime_faces/models/stylegan2-ffhq-config-f.pt`, using CLIP image-feature reconstruction and a small pixel MSE term.
2. Re-render the same `w` with the adapted checkpoint `course1_anime_faces/results/B_run_1024_fast_sanity/checkpoint/000600.pt` to obtain the base anime output.
3. Run short CLIP-guided latent optimization on the adapted generator for prompts `"anime portrait with blue hair"` and `"anime portrait with glasses"`, with regularization to the starting latent and base anime image.

Example real-edit grids:
- `course1_anime_faces/final/edits_real/grids/real_01_grid.png`
- `course1_anime_faces/final/edits_real/grids/real_02_grid.png`

Reproduce (Windows):

```bat
python course1_anime_faces/scripts/real_edit_styleclip.py ^
  --inputs_dir course1_anime_faces/final/edits_real/inputs ^
  --run_dir course1_anime_faces/results/B_run_1024_fast_sanity ^
  --ckpt 000600.pt ^
  --size 1024 ^
  --inv_steps 320 ^
  --edit_steps 100 ^
  --seed 42 ^
  --out_dir course1_anime_faces/final/edits_real
```

## 8. Evaluation Criteria Coverage

This submission covers:
- Training pipeline: zero-shot CLIP-guided adaptation with StyleGAN-NADA.
- Final qualitative results: selected checkpoint and visual progression artifacts.
- Synthetic edit pipeline: latent-only semantic editing from a fixed generated identity.
- Real edit pipeline: inversion, anime rerendering, and semantic latent edits on real inputs.

## Block Freezing Study: Mapping vs Last Blocks

In this project setup (StyleGAN2 + StyleGAN-NADA), the comparison target is which generator parts are trainable:
- **Mapping** strategy: keep mapping network frozen and optimize synthesis blocks (default branch in `SG2Generator.get_training_layers` when `phase` is not set).
- **Last blocks** strategy: intended to optimize only late synthesis blocks via `--phase lastblocks`.

Runnable entrypoints for the two strategies:
- `run_train_anime_mapping.cmd`
- `run_train_anime_lastblocks.cmd`

Artifact-to-strategy mapping from committed runs:
- Expected output dirs from scripts are `course1_anime_faces/results/anime_mapping/` and `course1_anime_faces/results/anime_lastblocks/`, but these directories are not present in the repository snapshot.
- Committed run folders under `course1_anime_faces/results/` have `args.json` with `phase: null`, so strict per-strategy attribution is not recoverable from saved metadata.
- We therefore report the comparison conservatively from available artifacts and code paths.

| Strategy | Trainable parts | Expected effect | Observed behavior in our runs | Notes / failure modes |
|---|---|---|---|---|
| Mapping (`run_train_anime_mapping.cmd`) | In current code path with `phase=null`: most synthesis blocks trainable; mapping and ToRGB frozen (`external/stylegan-nada/ZSSGAN/model/ZSSGAN.py`). | Stable identity/layout, moderate style shift. | Qualitative stylization is visible in committed runs, including `course1_anime_faces/results/B_run_1024_fast_sanity/sample/dst_000600.jpg` and final curated samples in `samples_best/`. | No committed run folder explicitly named/recorded as `anime_mapping`; attribution relies on code + args behavior. |
| Last blocks (`run_train_anime_lastblocks.cmd`) | Intended: only last synthesis blocks trainable. In this snapshot, `lastblocks` is not a dedicated `phase` branch, so it falls back to default behavior unless code is changed. | Potentially stronger late-stage texture/style changes with less global drift. | No committed run has `phase="lastblocks"` in `args.json`, so direct artifact-level observation for a distinct last-blocks regime is unavailable. | Main failure mode is provenance: missing strategy-tagged output dirs/metadata (`anime_lastblocks`, `phase=lastblocks`). |

Side-by-side visual references from committed outputs (qualitative context):

| Committed run A (`A_run_1024_iter1200`) | Committed run B (`B_run_1024_fast_sanity`) |
|---|---|
| ![](../../results/A_run_1024_iter1200/sample/dst_000600.jpg) | ![](../../results/B_run_1024_fast_sanity/sample/dst_000600.jpg) |
| ![](../../results/A_run_1024_iter1200/sample/dst_001000.jpg) | ![](../../results/B_run_1024_fast_sanity/sample/dst_001000.jpg) |

Conclusion for submission: we selected the final package `course1_anime_faces/final/B_run_1024_fast_sanity_best600/` with checkpoint `checkpoint/000600.pt` (source run `course1_anime_faces/results/B_run_1024_fast_sanity/`). Based on committed metadata (`phase=null`) and current code path, this corresponds to the default mapping-frozen style of training rather than a separately verifiable `lastblocks` run.
## 10. Limitations and Conclusion

Limitations are consistent with zero-shot CLIP-guided GAN adaptation:
- CLIP bias may not fully match desired artistic quality.
- Stronger stylization can reduce identity preservation.
- No target-domain training set limits controllability and robustness.
- Local artifacts may appear (texture noise, edge distortions, inconsistencies).

Overall, the project demonstrates practical zero-shot retargeting of a pretrained FFHQ face generator to anime portraits at 1024 resolution. For this run, `checkpoint/000600.pt` provides the best observed visual balance under qualitative evaluation.

