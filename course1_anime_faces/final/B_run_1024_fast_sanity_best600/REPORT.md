# CLIP-Guided Domain Adaptation of Face Generator to Anime Style

## 1. Project Goal

This project targets zero-shot domain adaptation of a pretrained face generator from the real-face domain to an anime portrait domain. No dedicated target image dataset is used for supervised training. Instead, CLIP provides semantic guidance from text, and the objective is to transfer anime style while preserving stable facial structure and identity-related characteristics.

## 2. Method Overview

The approach is based on StyleGAN-NADA (ZSSGAN). A frozen source generator acts as a semantic and structural anchor, while a trainable generator is optimized toward the target text concept. CLIP loss is used as the main supervision signal to enforce alignment between generated images and the target description.

Optimization is performed over generator parameters, which effectively steers outputs in latent/style space toward the target domain. The frozen reference helps reduce catastrophic drift, while CLIP guidance drives stylization.

## 3. Experimental Setup

- Base generator: StyleGAN2 pretrained on FFHQ.
- Adaptation method: StyleGAN-NADA / ZSSGAN.
- Resolution: 1024x1024.
- Target prompt: "anime portrait, clean lineart, cel shading, high quality".
- Key hyperparameters: iterations = 1200, learning rate = 0.001, `lambda_patch = 0.3`, `lambda_manifold = 0.2`.
- Hardware: single consumer GPU (NVIDIA GTX 1660 Super).
- Framework: PyTorch + CUDA.

Final selected run directory: `final/B_run_1024_fast_sanity_best600/`.
Selected checkpoint: `checkpoint/000600.pt`.

## 4. Training Process

Training followed iterative CLIP-guided adaptation with periodic sample export and checkpointing. As expected for CLIP-based optimization, training behavior was not fully stable across iterations: style strength can improve while local artifacts or identity drift can appear later.

Checkpoint selection was therefore qualitative. Checkpoint `000600.pt` was chosen by visual inspection as the best trade-off between stylization and facial coherence, rather than by minimal loss value.

## 5. Results

Evaluation is qualitative only.

Referenced artifacts:
- `figures/best_single.png`
- `figures/grid_best_steps.png`
- `samples_best/`

The selected outputs show clear anime-style transfer (line emphasis and cel-shading-like appearance) while keeping core facial layout and partial identity consistency from the source domain. The step-wise grid supports the choice of the region around step 600 as the most balanced result.

## 6. Analysis and Limitations

Main limitations are consistent with zero-shot CLIP-guided GAN adaptation:

- CLIP bias: optimization follows CLIP priors, which may not fully match desired artistic quality.
- Style-identity trade-off: stronger stylization can reduce identity preservation.
- Zero-shot limitation: absence of target-domain training data limits controllability and robustness.
- Artifacts: occasional texture noise, edge distortions, and local inconsistencies may appear.

These effects motivate manual checkpoint inspection and conservative model selection.

## 7. Conclusion

The project demonstrates that StyleGAN-NADA can adapt a pretrained FFHQ face generator to anime portrait style in a zero-shot setting at 1024 resolution. Qualitative results indicate meaningful style transfer with partial identity preservation, and checkpoint `000600.pt` provides the best observed balance for this run. The method is practical for rapid domain retargeting when target datasets are unavailable, with the caveat that visual quality control remains necessary.

## 8. Editing Generated Images (Synthetic)

To demonstrate synthetic editing (StyleCLIP-like), we apply CLIP-guided latent optimization on a single fixed latent code generated from seed `42` using the adapted checkpoint `course1_anime_faces/results/B_run_1024_fast_sanity/checkpoint/000600.pt`. For each target prompt, the generator is kept frozen and only the latent `w` is optimized for a small number of steps, with regularization that keeps the result close to the base image and base latent. The edit prompts are `"anime portrait with blue hair"` and `"anime portrait with glasses"`. This produces reproducible edits from the same starting identity while changing only the requested semantic attribute. Result grid path: `course1_anime_faces/final/edits_synthetic/grid.png`.

## 9. Real Image Editing

Real-image editing is implemented as a three-stage pipeline:
1. Invert each real portrait from `course1_anime_faces/final/edits_real/inputs/*.png` into latent `w` with the frozen FFHQ StyleGAN2 generator (`course1_anime_faces/models/stylegan2-ffhq-config-f.pt`) using CLIP image-feature reconstruction plus a small pixel MSE term.
2. Re-render the same `w` with the adapted anime generator checkpoint `course1_anime_faces/results/B_run_1024_fast_sanity/checkpoint/000600.pt` to get the base anime output.
3. Perform short CLIP-guided latent optimization on the adapted generator for two prompts: `"anime portrait with blue hair"` and `"anime portrait with glasses"`, while regularizing to the starting latent and base anime image.

Example grids:
- `../edits_real/grids/real_01_grid.png`
- `../edits_real/grids/real_02_grid.png`

Reproduce:

```bat
python course1_anime_faces/scripts/real_edit_styleclip.py --inputs_dir course1_anime_faces/final/edits_real/inputs --run_dir course1_anime_faces/results/B_run_1024_fast_sanity --ckpt 000600.pt --size 1024 --inv_steps 320 --edit_steps 100 --seed 42 --out_dir course1_anime_faces/final/edits_real
```
