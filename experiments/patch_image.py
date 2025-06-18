import sys
sys.path.append("/work/dlclarge2/ndirt-SegFM3D/learnedpatch/src")  # Adjust the path as necessary to import medsegbench
from patcher import Patcher
from medsegbench import Promise12MSBench
import torch.utils.data as data
import torchvision.transforms as transforms

data_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_dataset = Promise12MSBench(split="val", transform=data_transform, target_transform=data_transform, download=True)

val_loader = data.DataLoader(
    val_dataset,
    batch_size=1,  # Adjust batch size as needed
    shuffle=False,
    num_workers=4,  # Adjust number of workers based on your system
    pin_memory=True,
)

# get the first batch
for batch in val_loader:
    images, targets = batch
    break

print(f"Image shape: {images.shape}, Target shape: {targets.shape}")

