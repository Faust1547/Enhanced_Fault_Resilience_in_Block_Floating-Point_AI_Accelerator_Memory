import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import googlenet


# ===== path settings =====
DATA_ROOT = "D:/Anaconda/PythonCode/data" 
SAVE_PATH = os.path.join(DATA_ROOT, "googlenet_cifar10_ckpt_best.pth")

# ===== training settings =====
EPOCHS = 20
BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
NUM_WORKERS = 0                      
LR = 0.1                             
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DROPOUT = 0.2                      
NUM_CLASSES = 10

# ===== CIFAR-10 transform =====
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

train_tfm = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def make_loaders(device):
    os.makedirs(DATA_ROOT, exist_ok=True)

    trainset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        transform=train_tfm,
        download=False,              
    )
    testset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        transform=test_tfm,
        download=False,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    testloader = DataLoader(
        testset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    return trainloader, testloader


def build_model():
    model = googlenet(
        weights=None,
        aux_logits=False,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        init_weights=True,
    )


    model.conv1 = type(model.conv1)(
        3, 64, kernel_size=3, stride=1, padding=1
    )
    model.maxpool1 = nn.Identity()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    trainloader, testloader = make_loaders(device)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=len(trainloader),
        cycle_momentum=False,        
    )

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in trainloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()         

            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        acc = evaluate(model, testloader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
            f"Loss: {running_loss / seen:.4f} "
            f"Test Acc: {acc * 100:.2f}% "
            f"LR: {current_lr:.6f}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "acc": best_acc,
                    "epoch": epoch + 1,
                    "arch": "googlenet_cifar10",
                },
                SAVE_PATH,
            )
            print(f"Saved best checkpoint: {SAVE_PATH}, acc={best_acc * 100:.2f}%")

    print("Training finished.")
    print(f"Best Acc: {best_acc * 100:.2f}%")
    print(f"Best checkpoint saved at: {SAVE_PATH}")


if __name__ == "__main__":
    main()
