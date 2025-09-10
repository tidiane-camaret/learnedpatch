import numpy as np
import sys
#sys.path.append("/work/dlclarge2/ndirt-SegFM3D/learnedpatch")  # Adjust the path as necessary to import medsegbench
from learnedpatch.src.patcher import PatchSelectionEnv, dice_score

# Dummy image and mask (replace with real data as needed)
image = np.zeros((256, 256))
mask = np.zeros((256, 256))
mask[50:150, 50:150] = 1  # Example: square region

patch_size = (64, 64)
max_steps = 10

env = PatchSelectionEnv(image, mask, patch_size, max_steps)
obs = env.reset()

for step in range(max_steps):
    # Randomly select a valid patch location
    r = np.random.randint(0, image.shape[0] - patch_size[0] + 1)
    c = np.random.randint(0, image.shape[1] - patch_size[1] + 1)
    action = (r, c)
    obs, reward, done, info = env.step(action)
    print(f"Step {step+1}: Reward={reward:.4f}")
    if done:
        break

final_dice = dice_score(env.current_mask, mask)
print(f"Final Dice score: {final_dice:.4f}") 