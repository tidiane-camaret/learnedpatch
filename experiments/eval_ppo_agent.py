import numpy as np

from stable_baselines3 import PPO
import sys
sys.path.append("/work/dlclarge2/ndirt-SegFM3D/learnedpatch")  # Adjust the path as necessary to import medsegbench

from src.patchselectionenv import GymPatchSelectionEnv, dice_score


"""
# Dummy data
image = np.zeros((256, 256))
mask = np.zeros((256, 256))
mask[50:150, 50:150] = 1

"""

from medsegbench import Promise12MSBench
import torch.utils.data as data
import torchvision.transforms as transforms

data_transform = transforms.Compose([
    transforms.ToTensor()
])
val_dataset = Promise12MSBench(split="val", transform=data_transform, target_transform=data_transform, download=True)
image_list = [val_dataset[i][0] for i in range(len(val_dataset))]
mask_list = [val_dataset[i][1] for i in range(len(val_dataset))]
patch_size = (64, 64)
max_steps = 10

env = GymPatchSelectionEnv(image_list, mask_list, patch_size, max_steps)

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10)

# Evaluate
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}")

final_dice = dice_score(env.env.current_mask, env.env.mask)
print(f"Final Dice score: {final_dice:.4f}")
