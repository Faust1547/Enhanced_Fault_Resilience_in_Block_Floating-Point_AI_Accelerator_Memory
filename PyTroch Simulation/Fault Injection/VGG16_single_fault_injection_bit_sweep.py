import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import itertools
import os
import pandas as pd
import numpy as np

from msfp_converter import MSFPConverter
from tqdm import tqdm
from collections import OrderedDict
from collections import defaultdict
from typing import List, Dict, Tuple, Any


# ===== Path Settings =====
DATA_ROOT = "D:/Anaconda/PythonCode/data"
CKPT_PATH = os.path.join(DATA_ROOT, "vgg16_cifar10_ckpt_best.pth")

#########################################################################################
class MSFPActivationWrapper(nn.Module):
    def __init__(self, module, msfp_converter):
        super().__init__()
        self.module = module                  # 原本的 Conv2d 或 Linear
        self.msfp_converter = msfp_converter  # 你寫的 MSFPConverter 實例

    def forward(self, x):
        # activation 先做 MSFP 量化
        x = self.msfp_converter.quantize(x)
        return self.module(x)
    
class MSFPQuantizeLayer(nn.Module):
    def __init__(self, msfp_converter):
        super().__init__()
        self.msfp = msfp_converter

    def forward(self, x):
        return self.msfp.quantize(x)
    
msfp = MSFPConverter(mantissa_bits=7, box_size=16, verbose=False)
#########################################################################################
# ---------- Model: VGG16 for CIFAR-10 ----------
def build_vgg16_cifar10(num_classes=10) -> nn.Module:
    # Must match the training script:
    #   model = vgg16(weights=VGG16_Weights.DEFAULT)
    #   model.classifier[6] = nn.Linear(4096, 10)
    # Here weights=None is correct because we load your trained .pth below.
    m = vgg16(weights=None)
    m.classifier[6] = nn.Linear(4096, num_classes)
    return m


@torch.no_grad()
def predict_one(model: nn.Module, x: torch.Tensor, device: torch.device):
    model.eval()
    logits = model(x.unsqueeze(0).to(device))
    probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    pred = int(probs.argmax().item())
    conf = float(probs[pred].item())
    return pred, conf, probs


def denorm_for_show(x, mean, std):
    x = x.clone()
    for c in range(3):
        x[c] = x[c] * std[c] + mean[c]
    x = torch.clamp(x, 0, 1)
    x = (x * 255.0).byte()
    return x.permute(1, 2, 0).numpy()
#########################################################################################
# ---------- Sampling helper (your original) ----------
def fast_even_sample_counts(group_sizes, total_samples, seed=None):
    """
    Proportionally distribute faults according to each Conv layer's MSFP bit count.

    This avoids the old equal-per-layer allocation, which can severely
    over-inject small early layers (especially the first Conv layer) and cause
    an artificial full-layer bit flip / accuracy rebound at higher BER.
    """
    group_sizes = group_sizes.to(torch.int64)
    total_samples = int(total_samples)

    total_capacity = int(group_sizes.sum().item())

    if total_samples <= 0:
        return torch.zeros_like(group_sizes)

    if total_samples >= total_capacity:
        return group_sizes.clone()

    # Ideal proportional allocation:
    # count_i ~= total_samples * group_sizes[i] / sum(group_sizes)
    ideal = group_sizes.to(torch.float64) * (
        float(total_samples) / float(total_capacity)
    )
    counts = torch.floor(ideal).to(torch.int64)

    # Largest-remainder method so the final sum exactly matches total_samples.
    remaining = total_samples - int(counts.sum().item())

    if remaining > 0:
        fractional = ideal - counts.to(torch.float64)

        # Reproducible tie-breaking only.
        if seed is not None:
            g = torch.Generator(device=group_sizes.device)
            g.manual_seed(seed)
            tie_noise = torch.rand(
                fractional.shape,
                generator=g,
                device=group_sizes.device,
                dtype=torch.float64,
            ) * 1e-12
            fractional = fractional + tie_noise

        order = torch.argsort(fractional, descending=True)

        for idx in order.tolist():
            if remaining == 0:
                break
            if counts[idx] < group_sizes[idx]:
                counts[idx] += 1
                remaining -= 1

    return counts


def compute_counts_per_conv_from_ber(model, msfp, ber: float, seed: int):
    """
    依照「每個 conv layer 的 MSFP bit 數」分配 total faulty bits。
    回傳 1D tensor counts_per_conv，長度 = conv 層數。
    """
    conv_sizes = torch.tensor([m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv2d)],
                              dtype=torch.int64)

    box = msfp.box_size
    m_bits = msfp.mantissa_bits

    # num_tiles = ceil(N / box)
    num_tiles = (conv_sizes + box - 1) // box

    # per-layer total bits = exp_bits + mantissa_bits + sign_bits
    bits_per_layer = num_tiles * 8 + conv_sizes * (m_bits + 1)
    total_bits = bits_per_layer.sum()

    num_samples = torch.round(total_bits.float() * float(ber)).to(torch.int64)
    counts = fast_even_sample_counts(bits_per_layer.to(torch.int64), num_samples, seed=seed)

    return counts.to(torch.int64)
#########################################################################################
#MSFP Quantization

# 插入量化層到 Conv2d 前
def insert_quant_before_conv(module: nn.Module, quant_layer_):
    for name, child in list(module.named_children()):
        # 如果碰到 Conv2d，就把它換成 [quant,conv]
        if isinstance(child, nn.Conv2d):
            wrapped = nn.Sequential(
                quant_layer_(),
                child,
            )
            setattr(module, name, wrapped)
        else:
            # 否則繼續深入子結構遍歷
            insert_quant_before_conv(child, quant_layer_)

#########################################################################################
def eval_top1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0

    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc="Inference", unit="batch", leave=True)

    for x, y in it:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

        # 更新進度條資訊（即時顯示 accuracy）
        if tqdm is not None:
            it.set_postfix(acc=f"{(correct/total):.4f}")

    return correct / total
#########################################################################################
def main(mant_bits, box_size, iter, ber, index, seed=None, data_root=DATA_ROOT, ckpt_path=CKPT_PATH, batch_size=256, num_workers=0):

    top1_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computed counts_per_conv from BER:", ber)
    print("Bit position:", index)
    print("Device:", device)
    # MSFP converter
    msfp = MSFPConverter(mantissa_bits=mant_bits, box_size=box_size)

    # CIFAR-10 mean/std（跟你訓練一致）
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    testset = datasets.CIFAR10(root=data_root, train=False, transform=tfm, download=False)
    classes = testset.classes

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),])

    # testset = datasets.CIFAR10(
    #     root="E:/python code/data",
    #     train=False, transform=tfm)

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False)

    # Load model
    model = build_vgg16_cifar10(num_classes=10)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)


    for i in range(iter):
        # Get counts per conv
        counts_per_conv = compute_counts_per_conv_from_ber(model, msfp, ber=ber, seed=seed)

        # sanity check
        num_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        if counts_per_conv.numel() != num_conv:
            raise ValueError(f"counts_per_conv len={counts_per_conv.numel()} != num_conv={num_conv}")
        
        # Load model
        """插入量化層到 Conv2d 前"""
        quant_layer = lambda: MSFPQuantizeLayer(msfp)
        model_quantized = build_vgg16_cifar10(num_classes=10)
        model_quantized.load_state_dict(state, strict=True)
        insert_quant_before_conv(model_quantized, quant_layer)
        model_quantized.to(device).eval()

        conv_layers = [layer for layer in model_quantized.modules() if isinstance(layer, nn.Conv2d)]

        if torch.is_tensor(counts_per_conv):
            counts_list = counts_per_conv.detach().cpu().tolist()
        else:
            counts_list = list(counts_per_conv)

        for i, layer in enumerate(conv_layers):
            c = int(counts_list[i])
            w = layer.weight.data

            # Keep runs reproducible, but use a different deterministic
            # random stream for each Conv layer.
            layer_seed = None if seed is None else seed + i * 100003

            quantized_param = msfp.quantize_with_fault_injection(
                data=w,
                index=index,
                fault_num=c,
                seed=layer_seed
            )

            layer.weight.data.copy_(quantized_param.to(device))
        acc = eval_top1(model_quantized, testloader, device)
        top1_list.append(acc)

    print("\nFinish !")
    print(f"\n=== Bit {index} Top-1 accuracy in BER {ber} 全部迭代結果 ===")
    for i in top1_list:
        print(i)

    return top1_list


def run_exponent_bit_sweep_to_csv(
    ber_list,
    bit_points=None,
    csv_name="VGG16_CIFAR10_bit_sweep.csv",
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
    Sweep multiple BER points and multiple bit positions, then save results to CSV.

    With mant_bits=7:
      Bit 0~6   : mantissa
      Bit 7~14  : exponent
      Bit 15    : sign
    """
    #################################################################################### 改測試BIT ######################################################
    if bit_points is None:
        #bit_points = [15, 6, 5, 4, 3, 2, 1, 0]
        bit_points = [14, 13, 12, 11, 10, 9, 8, 7]
    ####################################################################################################################################################
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

            # 儲存百分比 accuracy；CSV 只保留 Bit 欄位，方便原本的 ber_plot.py 直接畫圖
            row[f"Bit {bit}"] = float(np.mean(acc_list)) * 100.0

        rows.append(row)

        # 每跑完一個 BER 就先存一次，避免中途停止時資料全丟
        df_partial = pd.DataFrame(rows)
        df_partial.to_csv(csv_name, index=False)
        print(f"\n[Saved partial CSV] {csv_name}")
        print(df_partial)

    df = pd.DataFrame(rows)
    df.to_csv(csv_name, index=False)
    print(f"\n[Done] CSV saved to: {csv_name}")
    print(df)
    return df



@torch.no_grad()
def evaluate_clean_model(data_root=DATA_ROOT, ckpt_path=CKPT_PATH, batch_size=256, num_workers=0):
    """Evaluate the clean VGG16-CIFAR10 checkpoint without fault injection."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    testset = datasets.CIFAR10(root=data_root, train=False, transform=tfm, download=False)
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    model = build_vgg16_cifar10(num_classes=10)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    acc = eval_top1(model, testloader, device)
    print(f"Clean VGG16-CIFAR10 Acc: {acc * 100:.2f}%")
    return acc


if __name__ == "__main__":

    # run_exponent_bit_sweep_to_csv(
    #     ber_list=[1e-7, 1e-6, 1e-5, 1e-4],
    #     bit_points=[15, 6, 5, 4, 3, 2, 1, 0],
    #     csv_name="VGG16_CIFAR10_bit_sweep.csv",
    #     mant_bits=7,
    #     box_size=16,
    #     repeats=1,
    #     base_seed=0,
    #     data_root=DATA_ROOT,
    #     ckpt_path=CKPT_PATH,
    #     batch_size=256,
    #     num_workers=0,
    #     )

    run_exponent_bit_sweep_to_csv(
        ber_list = [
            5e-4,
        
            1e-3,
            3e-3,
            5e-3,
            7e-3,
            
            1e-2,
            3e-2,
            5e-2,
            7e-2,
            
            1e-1,
            3e-1
        ],
        bit_points=[15,6,5,4,3,2,1,0],
        csv_name="VGG16_CIFAR10_bit_sweep_new_SM.csv",
        mant_bits=7,
        box_size=16,
        repeats=25,
        base_seed=0,
        data_root=DATA_ROOT,
        ckpt_path=CKPT_PATH,
        batch_size=256,
        num_workers=0,
        )
    
    # run_exponent_bit_sweep_to_csv(
    #     ber_list = [1e-4, 1e-3, 1e-2, 1e-1],
    #     bit_points=[15],
    #     csv_name="VGG16_CIFAR10_bit_sweep.csv",
    #     mant_bits=7,
    #     box_size=16,
    #     repeats=30,
    #     base_seed=0,
    #     data_root=DATA_ROOT,
    #     ckpt_path=CKPT_PATH,
    #     batch_size=256,
    #     num_workers=0,
    #     )
