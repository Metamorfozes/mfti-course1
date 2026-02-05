# course1_anime_faces

Minimal project skeleton for an educational ML project.

Task formulation: adapt a pretrained StyleGAN2 (FFHQ) to the anime face domain using CLIP guidance.

This is an educational project for MFTI Course 1.

## Windows Quickstart
1. Run un_patch.cmd to patch CUDA extensions and clear the Torch extensions cache.
2. Run un_help.cmd to confirm the training script is available (prints 	rain.py help).
3. Run one of the training launchers: un_train_anime_mapping.cmd or un_train_anime_lastblocks.cmd. Edit the args inside as needed.

Results are saved under course1_anime_faces\\results\\... (samples in sample\\, checkpoints in checkpoint\\).

