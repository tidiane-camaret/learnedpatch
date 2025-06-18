import random
import numpy as np

def dice_score(pred, target, eps=1e-6):
    intersection = np.sum((pred > 0) & (target > 0))
    union = np.sum(pred > 0) + np.sum(target > 0)
    return (2. * intersection + eps) / (union + eps)

class Patcher:
    def __init__(self, image_width, image_height, patch_size):
        self.patch_size = patch_size

    def random(self, input_image):
        image_shape = input_image.shape
        assert len(image_shape) == len(self.patch_size), "Input image shape must match patch size dimensions"
        coords = [random.randint(0, image_shape[i] - self.patch_size[i]) for i in range(len(self.patch_size))]
        return [coords[i:i + 2] for i in range(0, len(coords), 2)]

class PatchSelectionEnv:
    def __init__(self, image, mask, patch_size, max_steps):
        # image: input image (e.g., numpy array)
        # mask: ground truth segmentation mask
        # patch_size: (height, width) of each patch
        # max_steps: maximum number of patches per episode
        self.image = image
        self.mask = mask
        self.patch_size = patch_size
        self.max_steps = max_steps

        self.reset()

    def reset(self):
        # Initialize state: empty predicted mask, step counter, etc.
        self.current_mask = np.zeros_like(self.mask)
        self.steps = 0
        self.covered = np.zeros_like(self.mask, dtype=bool)
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
        done = (self.steps >= self.max_steps) or np.all(self.covered)

        return self._get_observation(), reward, done, {}

    def render(self):
        # Optional: visualize current state
        pass