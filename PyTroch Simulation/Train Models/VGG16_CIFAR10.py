import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader

# ===== path settings =====
DATA_ROOT = "D:/Anaconda/PythonCode/data"
SAVE_PATH = os.path.join(DATA_ROOT, "vgg16_cifar10_ckpt_best.pth")

# ===== training settings =====
EPOCHS = 100          # CPU is slow; try 10 first if you only want to test the flow
BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
NUM_WORKERS = 0      # Windows + Jupyter: 0 is usually the safest
LR = 0.01            # safer for fine-tuning ImageNet-pretrained VGG16 than 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# ===== device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===== CIFAR-10 transform =====
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

train_tfm = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

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

# ===== model: ImageNet-pretrained VGG16 fine-tuned for CIFAR-10 =====
model = vgg16(weights=VGG16_Weights.DEFAULT)
model.classifier[6] = nn.Linear(4096, 10)
model = model.to(device)

# ===== loss / optimizer / scheduler =====
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 40],
    gamma=0.2,
)

# ===== evaluation =====
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total

# ===== training =====
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for x, y in trainloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    acc = evaluate(model, testloader)

    print(
        f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
        f"Loss: {running_loss / len(trainloader):.4f} "
        f"Test Acc: {acc * 100:.2f}% "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    if acc > best_acc:
        best_acc = acc
        torch.save(
            {
                "model": model.state_dict(),
                "acc": best_acc,
                "epoch": epoch + 1,
                "arch": "vgg16_cifar10",
            },
            SAVE_PATH,
        )
        print(f"Saved best checkpoint: {SAVE_PATH}, acc={best_acc * 100:.2f}%")

print("Training finished.")
print(f"Best Acc: {best_acc * 100:.2f}%")
print(f"Best checkpoint saved at: {SAVE_PATH}")
