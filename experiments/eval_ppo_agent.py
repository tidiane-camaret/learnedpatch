import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
import sys
sys.path.append("/work/dlclarge2/ndirt-SegFM3D/learnedpatch")  # Adjust the path as necessary to import medsegbench

from src.patcher import PatchSelectionEnv, dice_score

class GymPatchSelectionEnv(gym.Env):
    def __init__(self, image, mask, patch_size, max_steps):
        super().__init__()
        self.env = PatchSelectionEnv(image, mask, patch_size, max_steps)
        self.image = image
        self.mask = mask
        self.patch_size = patch_size
        self.max_steps = max_steps

        # Action: (row, col) for top-left of patch
        self.action_space = spaces.MultiDiscrete([
            image.shape[0] - patch_size[0] + 1,
            image.shape[1] - patch_size[1] + 1
        ])
        # Observation: stack image, current_mask, covered as channels
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, image.shape[0], image.shape[1]),
            dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()
        return self._obs_to_array(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(tuple(action))
        return self._obs_to_array(obs), reward, done, info

    def _obs_to_array(self, obs):
        # Stack image, current_mask, covered as channels
        return np.stack([
            obs["image"].astype(np.float32),
            obs["current_mask"].astype(np.float32),
            obs["covered"].astype(np.float32)
        ], axis=0)

# Dummy data
image = np.zeros((256, 256))
mask = np.zeros((256, 256))
mask[50:150, 50:150] = 1
patch_size = (64, 64)
max_steps = 10

env = GymPatchSelectionEnv(image, mask, patch_size, max_steps)

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Evaluate
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}")

final_dice = dice_score(env.env.current_mask, mask)
print(f"Final Dice score: {final_dice:.4f}")
