import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
sys.path.append("/work/dlclarge2/ndirt-SegFM3D/learnedpatch")  # Adjust the path as necessary to import medsegbench

from src.patchselectionenv import GymPatchSelectionEnv, dice_score

from medsegbench import Promise12MSBench
import torch.utils.data as data
import torch
import torchvision.transforms.v2 as transforms

data_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    #transforms.RandomCrop(256)
])

crop_transform = transforms.RandomCrop(280)
train_dataset = Promise12MSBench(split="train", transform=data_transform, target_transform=data_transform, download=True)

image_list = [train_dataset[i][0] for i in range(len(train_dataset))]
mask_list = [train_dataset[i][1] for i in range(len(train_dataset))]
patch_size = (32, 32)
max_steps = 10

env = GymPatchSelectionEnv(image_list, mask_list, patch_size, max_steps)

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="results/ppo_learnedpatch")  
# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000, callback=checkpoint_callback)

# Evaluate
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}")

final_dice = dice_score(env.env.current_mask, env.env.mask)
print(f"Final Dice score: {final_dice:.4f}")

#model.save("results/ppo_learnedpatch.pkl")