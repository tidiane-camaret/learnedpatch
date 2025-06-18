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


### plot first image and label
import matplotlib.pyplot as plt
def plot_image_and_label(image, label):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image.permute(1, 2, 0))  # Change to HWC format for plotting
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(label.squeeze(), cmap='gray')  # Assuming label is a single channel
    ax[1].set_title("Label")
    ax[1].axis("off")

    plt.savefig("results/image_and_label.png")
    
# Example usage: plot the first image and label from the validation set
for i, (image, label) in enumerate(val_loader):
    if i == 0:  # Plot only the first image and label
        plot_image_and_label(image[0], label[0])
        break