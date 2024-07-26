from code.baseline_dataset import BaselineDataset
from code.collate import pad_collate

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.segmentation import MeanIoU
from tqdm import tqdm


def main(
    data_folder: str,
    mini_dataset: bool = False,
    num_channels: int = 10,
    num_classes: int = 20,
    init_features: int = 32,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_epochs: int = 10,
    device: str = "cpu",
    scheduler: dict = None,
):
    # Dataset and Dataloader
    dt = BaselineDataset(data_folder, mini_dataset)
    train_loader = torch.utils.data.DataLoader(
        dt, batch_size=batch_size, collate_fn=pad_collate, shuffle=True
    )

    # Model
    unet = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=num_channels,
        out_channels=num_classes,
        init_features=init_features,
        pretrained=False,
    ).to(device)
    pytorch_total_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print("UNet Model ready.")
    print(unet)
    print("Trainable parameters:", pytorch_total_params)

    # Training criterions
    criterion = nn.CrossEntropyLoss().to(device)
    mean_iou = MeanIoU(num_classes).to(device)
    optimizer = optim.Adam(unet.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, **training_args["scheduler"])

    epoch_losses = []
    epoch_ious = []

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}.")
        unet.train()
        running_loss = 0.0
        running_iou = 0.0
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()

            outputs = unet(x["S2"][:, 0, :, :, :].to(device))

            loss = criterion(outputs, y.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            running_loss += loss.item()
            iou = mean_iou(torch.argmax(outputs, dim=1), y.to(device))
            running_iou += iou.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_iou = running_iou / len(train_loader)
        epoch_losses.append(epoch_loss)
        epoch_ious.append(epoch_iou)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.5f}, IoU: {epoch_iou:.5f}"
        )
        for group in optimizer.param_groups:
            print(f"Learning rate: {group['lr']}")

    plot_metrics(epoch_losses, epoch_ious, num_epochs)


def plot_metrics(epoch_losses, epoch_ious, num_epochs):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker="o", label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), epoch_ious, marker="o", label="Mean IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.title("Mean IoU")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Fill these file paths with the locations on your machine.
    training_args = {
        "data_folder": "/Users/lstefanu/Documents/mines/pastis-benchmark/PASTIS/",
        "mini_dataset": True,
        "num_channels": 10,
        "num_classes": 20,
        "init_features": 32,
        "batch_size": 32,
        "lr": 1e-3,
        "num_epochs": 1000,
        "device": "mps",  # cpu, gpu, mps (metal chips on Mac)
        "scheduler": {"mode": "min", "factor": 0.1, "patience": 50},
    }
    main(**training_args)
