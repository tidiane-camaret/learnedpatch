import random
import numpy as np
import gym
from gym import spaces
import torch

def dice_score(pred, target, eps=1e-6):
    intersection = torch.sum((pred > 0) & (target > 0))
    union = torch.sum(pred > 0) + torch.sum(target > 0)
    return (2. * intersection + eps) / (union + eps)

class GymPatchSelectionEnv(gym.Env):
    def __init__(self, image_list, mask_list, patch_size, max_steps):
        super().__init__()
        self.env = PatchSelectionEnv(image_list, mask_list, patch_size, max_steps)
        self.image = image_list[0][0]
        self.mask = mask_list[0][0]
        self.patch_size = patch_size
        self.max_steps = max_steps

        print(self.image.shape)

        # Action: (row, col) for top-left of patch
        self.action_space = spaces.MultiDiscrete([
            self.image.shape[0] - patch_size[0] + 1,
            self.image.shape[1] - patch_size[1] + 1
        ])
        # Observation: stack image, current_mask, covered as channels
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, self.image.shape[0], self.image.shape[1]),
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
        return torch.stack([
            obs["image"],
            obs["current_mask"],
            obs["covered"]
        ], dim=0)
    
class PatchSelectionEnv:
    def __init__(self, image_list, mask_list, patch_size, max_steps):
        # image: input image (e.g., numpy array)
        # mask: ground truth segmentation mask
        # patch_size: (height, width) of each patch
        # max_steps: maximum number of patches per episode
        self.image_list = image_list
        self.mask_list = mask_list
        self.patch_size = patch_size
        self.max_steps = max_steps

        self.reset()

    def reset(self):
        # Initialize state: empty predicted mask, step counter, etc.
        self.index = random.randint(0, len(self.image_list) - 1)
        self.image = self.image_list[self.index][0]
        self.mask = self.mask_list[self.index][0]
        self.current_mask = torch.zeros_like(self.mask)
        self.steps = 0
        self.covered = torch.zeros_like(self.mask, dtype=torch.bool)
        # Optionally: history of selected patches
        return self._get_observation()

    def _get_observation(self):
        # Return the current observation (e.g., image, current_mask, covered)
        return {
            "image": self.image,
            "current_mask": self.current_mask,
            "covered": self.covered
        }

    def step(self, action):
        # action: (row, col) coordinates for the top-left of the patch
        r, c = action
        # Extract patch region
        patch_mask = self.mask[r:r+self.patch_size[0], c:c+self.patch_size[1]]
        # Update predicted mask (for now, just copy ground truth)
        self.current_mask[r:r+self.patch_size[0], c:c+self.patch_size[1]] = patch_mask
        self.covered[r:r+self.patch_size[0], c:c+self.patch_size[1]] = True

        # Compute reward (e.g., incremental Dice score)
        reward = dice_score(self.current_mask, self.mask)

        self.steps += 1
        done = (self.steps >= self.max_steps) or torch.all(self.covered)

        return self._get_observation(), reward, done, {}

    def render(self):
        # Optional: visualize current state
        pass