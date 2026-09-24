import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from msfp_converter import MSFPConverter
from tqdm import tqdm

# ===== Path Settings =====
DATA_ROOT = "D:/Anaconda/PythonCode/data"
CKPT_PATH = os.path.join(DATA_ROOT, "alexnet_cifar10_ckpt_best.pth")


class MSFPQuantizeLayer(nn.Module):
    def __init__(self, msfp_converter):
        super().__init__()
        self.msfp = msfp_converter

    def forward(self, x):
        return self.msfp.quantize(x)


class AlexNetCIFAR10(nn.Module):
    """
    AlexNet-style network adjusted for CIFAR-10 32x32 images.
    This architecture must match AlexNet_CIFAR10_train.py.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
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


def build_alexnet_cifar10(num_classes=10) -> nn.Module:
    return AlexNetCIFAR10(num_classes=num_classes)


def fast_even_sample_counts(group_sizes, total_samples, seed=None):
    if total_samples == group_sizes.sum():
        return group_sizes.clone()

    num_groups = group_sizes.size(0)
    if seed is not None:
        torch.manual_seed(seed)

    base = total_samples // num_groups
    counts = torch.full_like(group_sizes, base)
    counts = torch.min(counts, group_sizes)

    remaining = total_samples - counts.sum()
    if remaining > 0:
        room = group_sizes - counts
        eligible_idx = (room > 0).nonzero(as_tuple=True)[0]
        if len(eligible_idx) > 0:
            weights = room[eligible_idx].float()
            probs = weights / weights.sum()
            sampled = torch.multinomial(probs, remaining, replacement=True)
            update = torch.bincount(sampled, minlength=len(eligible_idx))
            counts[eligible_idx] += update
            counts = torch.min(counts, group_sizes)

    return counts


def compute_counts_per_conv_from_ber(model, msfp, ber: float, seed: int):
    """
    Allocate total faulty bits according to MSFP bit count of every Conv2d layer.
    Return counts_per_conv with length = number of Conv2d layers.
    """
    conv_sizes = torch.tensor(
        [m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv2d)],
        dtype=torch.int64,
    )

    box = msfp.box_size
    m_bits = msfp.mantissa_bits
    num_tiles = (conv_sizes + box - 1) // box
    bits_per_layer = num_tiles * 8 + conv_sizes * (m_bits + 1)
    total_bits = bits_per_layer.sum()

    num_samples = torch.round(total_bits.float() * float(ber)).to(torch.int64)
    counts = fast_even_sample_counts(bits_per_layer.to(torch.int64), num_samples, seed=seed)
    return counts.to(torch.int64)


def insert_quant_before_conv(module: nn.Module, quant_layer_):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            wrapped = nn.Sequential(
                quant_layer_(),
                child,
            )
            setattr(module, name, wrapped)
        else:
            insert_quant_before_conv(child, quant_layer_)


def load_checkpoint_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return state


def build_testloader(data_root=DATA_ROOT, batch_size=256, num_workers=0):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    testset = datasets.CIFAR10(root=data_root, train=False, transform=tfm, download=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )


@torch.no_grad()
def eval_top1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    it = tqdm(loader, desc="Inference", unit="batch", leave=True) if tqdm is not None else loader
    for x, y in it:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        if tqdm is not None:
            it.set_postfix(acc=f"{(correct/total):.4f}")

    return correct / total


@torch.no_grad()
def evaluate_clean_model(data_root=DATA_ROOT, ckpt_path=CKPT_PATH, batch_size=256, num_workers=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testloader = build_testloader(data_root=data_root, batch_size=batch_size, num_workers=num_workers)
    state = load_checkpoint_state(ckpt_path)

    model = build_alexnet_cifar10(num_classes=10)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    acc = eval_top1(model, testloader, device)
    print(f"Clean AlexNet-CIFAR10 Acc: {acc * 100:.2f}%")
    return acc


def main(
    mant_bits,
    box_size,
    iter,
    ber,
    index,
    seed=None,
    data_root=DATA_ROOT,
    ckpt_path=CKPT_PATH,
    batch_size=256,
    num_workers=0,
):
    top1_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computed counts_per_conv from BER:", ber)
    print("Bit position:", index)
    print("Device:", device)

    msfp = MSFPConverter(mantissa_bits=mant_bits, box_size=box_size)
    testloader = build_testloader(data_root=data_root, batch_size=batch_size, num_workers=num_workers)

    model = build_alexnet_cifar10(num_classes=10)
    state = load_checkpoint_state(ckpt_path)
    model.load_state_dict(state, strict=True)

    for _ in range(iter):
        counts_per_conv = compute_counts_per_conv_from_ber(model, msfp, ber=ber, seed=seed)

        num_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        if counts_per_conv.numel() != num_conv:
            raise ValueError(f"counts_per_conv len={counts_per_conv.numel()} != num_conv={num_conv}")

        quant_layer = lambda: MSFPQuantizeLayer(msfp)
        model_quantized = build_alexnet_cifar10(num_classes=10)
        model_quantized.load_state_dict(state, strict=True)
        insert_quant_before_conv(model_quantized, quant_layer)
        model_quantized.to(device).eval()

        conv_layers = [layer for layer in model_quantized.modules() if isinstance(layer, nn.Conv2d)]
        counts_list = counts_per_conv.detach().cpu().tolist() if torch.is_tensor(counts_per_conv) else list(counts_per_conv)

        for layer_idx, layer in enumerate(conv_layers):
            c = int(counts_list[layer_idx])
            w = layer.weight.data
            quantized_param = msfp.quantize_with_fault_injection(
                data=w,
                index=index,
                fault_num=c,
                seed=seed,
            )
            layer.weight.data.copy_(quantized_param.to(device))

        acc = eval_top1(model_quantized, testloader, device)
        top1_list.append(acc)

    print("\nFinish !")
    print(f"\n=== Bit {index} Top-1 accuracy in BER {ber} all results ===")
    for acc in top1_list:
        print(acc)

    return top1_list


def run_bit_sweep_to_csv(
    ber_list,
    bit_points=None,
    csv_name="AlexNet_CIFAR10_bit_sweep.csv",
    mant_bits=7,
    box_size=16,
    repeats=1,
    base_seed=0,
    data_root=DATA_ROOT,
    ckpt_path=CKPT_PATH,
    batch_size=256,
    num_workers=0,
):
    """
    Sweep multiple BER points and bit positions, then save mean accuracy to CSV.

    With mant_bits=7:
      Bit 0~6   : mantissa
      Bit 7~14  : exponent
      Bit 15    : sign
    """
    if bit_points is None:
        bit_points = [14, 13, 12, 11, 10, 9, 8, 7]

    rows = []
    for ber in ber_list:
        print("\n" + "=" * 70)
        print(f"BER = {ber}")
        print("=" * 70)
        row = {"BER": ber}

        for bit in bit_points:
            print("\n" + "-" * 30 + f" Bit {bit} " + "-" * 30)
            acc_list = []
            for r in range(repeats):
                seed = None if base_seed is None else base_seed + r
                result = main(
                    mant_bits=mant_bits,
                    box_size=box_size,
                    iter=1,
                    ber=ber,
                    index=bit,
                    seed=seed,
                    data_root=data_root,
                    ckpt_path=ckpt_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                acc_list.extend(result)

            row[f"Bit {bit}"] = float(np.mean(acc_list)) * 100.0

        rows.append(row)
        df_partial = pd.DataFrame(rows)
        df_partial.to_csv(csv_name, index=False)
        print(f"\n[Saved partial CSV] {csv_name}")
        print(df_partial)

    df = pd.DataFrame(rows)
    df.to_csv(csv_name, index=False)
    print(f"\n[Done] CSV saved to: {csv_name}")
    print(df)
    return df


if __name__ == "__main__":
    # Sanity check first if needed:
    # evaluate_clean_model(data_root=DATA_ROOT, ckpt_path=CKPT_PATH, batch_size=256, num_workers=0)

    # run_bit_sweep_to_csv(
    #     ber_list=[1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 1.05e-5],
    #     bit_points=[14, 13, 12, 11, 10, 9, 8, 7],
    #     csv_name="AlexNet_CIFAR10_exponent_bit_sweep.csv",
    #     mant_bits=7,
    #     box_size=16,
    #     repeats=1,
    #     base_seed=0,
    #     data_root=DATA_ROOT,
    #     ckpt_path=CKPT_PATH,
    #     batch_size=256,
    #     num_workers=0,
    # )

    run_bit_sweep_to_csv(
        ber_list=[1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 1.05e-5],
        bit_points=[15, 6, 5, 4, 3, 2, 1, 0],
        csv_name="AlexNet_CIFAR10_Matissa_bit_sweep.csv",
        mant_bits=7,
        box_size=16,
        repeats=30,
        base_seed=0,
        data_root=DATA_ROOT,
        ckpt_path=CKPT_PATH,
        batch_size=256,
        num_workers=0,
    )
