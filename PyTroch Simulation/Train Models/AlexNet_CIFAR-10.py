# Train AlexNet + CIFAR-10 <訓練模型>
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===== Path Settings =====
DATA_ROOT = "D:/Anaconda/PythonCode/data"
SAVE_PATH = os.path.join(DATA_ROOT, "alexnet_cifar10_ckpt_best.pth")

# ===== Training Settings =====
EPOCHS = 300
BATCH_SIZE = 128
NUM_WORKERS = 0
LR = 0.01

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class AlexNetCIFAR10(nn.Module):
    """
    AlexNet-style network adjusted for CIFAR-10 32x32 images.
    Do not load torchvision AlexNet pretrained weights into this model.
    Use this same architecture for training and fault injection.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 32 -> 16

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 16 -> 8

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 8 -> 4
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_alexnet_cifar10(num_classes=10):
    return AlexNetCIFAR10(num_classes=num_classes)


# ===== CIFAR-10 Transform =====
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2023, 0.1994, 0.2010)

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

trainset = datasets.CIFAR10(root=DATA_ROOT, train=True, transform=train_tfm, download=False)
testset = datasets.CIFAR10(root=DATA_ROOT, train=False, transform=test_tfm, download=False)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)

# ===== Model / Loss / Optimizer =====
model = build_alexnet_cifar10(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.2)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for x, y in trainloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    acc = evaluate(model, testloader)

    print(f"Epoch [{epoch+1:03d}/{EPOCHS}] Loss: {running_loss/len(trainloader):.4f} Test Acc: {acc*100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save({
            "model": model.state_dict(),
            "acc": best_acc,
            "epoch": epoch + 1,
            "arch": "alexnet_cifar10",
        }, SAVE_PATH)
        print(f"Saved best checkpoint: {SAVE_PATH}, acc={best_acc*100:.2f}%")

print("Training finished.")
print(f"Best Acc: {best_acc*100:.2f}%")
