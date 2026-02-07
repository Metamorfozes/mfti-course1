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
