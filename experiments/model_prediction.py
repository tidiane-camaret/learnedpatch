from medsegbench import Promise12MSBench
import segmentation_models_pytorch as smp
import torch.utils.data as data
import torch
import torchvision.transforms as transforms

def main():
    
    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #train_dataset = Promise12MSBench(split="train", download=True)
    val_dataset = Promise12MSBench(split="val", transform=data_transform, target_transform=data_transform, download=True)

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=1,  # Adjust batch size as needed
        shuffle=False,
        num_workers=4,  # Adjust number of workers based on your system
        pin_memory=True,
    )
    ENCODER_NAME = "resnet18"  # You can change this to any other encoder supported by segmentation_models_pytorch
    n_channels = 1  # Change this if your input images have a different number of channels
    
    model = smp.Unet(
        encoder_name=ENCODER_NAME,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=n_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    ).to(device)


    for i, batch in enumerate(val_loader):

        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits_mask = model(inputs)
            
            # Lets compute metrics for some threshold
            # first convert mask values to probabilities, then 
            # apply thresholding
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()

            # We will compute IoU metric by two ways
            #   1. dataset-wise
            #   2. image-wise
            # but for now we just compute true positive, false positive, false negative and
            # true negative 'pixels' for each image and class
            # these values will be aggregated in the end of an epoch
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), labels.long(), mode="binary")
            # print the results
            print(f"Image {i+1}:")
            print(f"True Positives: {tp.item()}")   
            print(f"False Positives: {fp.item()}")


if __name__ == "__main__":
    main()
