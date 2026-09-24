import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from msfp_converter import MSFPConverter
from torchvision.models import googlenet

DATA_ROOT = "D:/Anaconda/PythonCode/data"
CKPT_PATH = os.path.join(DATA_ROOT, "googlenet_cifar10_ckpt_best.pth")

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

def build_googlenet_cifar10(num_classes=10):
    model = googlenet(
        weights=None,
        aux_logits=False,      
        num_classes=num_classes,
        dropout=0.2,          
        init_weights=False,
    )
    model.conv1 = type(model.conv1)(
        3, 64, kernel_size=3, stride=1, padding=1
    )
    model.maxpool1 = nn.Identity()
    return model


class MSFPQuantizeLayer(nn.Module):
    def __init__(self, msfp_converter):
        super().__init__()
        self.msfp = msfp_converter

    def forward(self, x):
        return self.msfp.quantize(x)


def insert_quant_before_conv(module: nn.Module, quant_layer_factory):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, nn.Sequential(quant_layer_factory(), child))
        else:
            insert_quant_before_conv(child, quant_layer_factory)


def get_conv_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def get_conv_weight_sizes(model):
    return torch.tensor(
        [m.weight.numel() for m in get_conv_layers(model)],
        dtype=torch.int64,
    )


def fast_even_sample_counts(group_sizes, total_samples, seed=None):
    """最大餘數法：按照各層 MSFP bit 數比例分配，且分配總和等於要求數量。"""
    group_sizes = torch.as_tensor(group_sizes, dtype=torch.int64)
    total_samples = int(total_samples)
    total_capacity = int(group_sizes.sum().item())

    if total_samples <= 0:
        return torch.zeros_like(group_sizes)
    if total_samples >= total_capacity:
        return group_sizes.clone()

    ideal = group_sizes.to(torch.float64) * (total_samples / total_capacity)
    counts = torch.floor(ideal).to(torch.int64)
    remaining = total_samples - int(counts.sum().item())

    if remaining:
        fractions = ideal - counts.to(torch.float64)
        if seed is not None:
            generator = torch.Generator(device=group_sizes.device)
            generator.manual_seed(int(seed))
            fractions += torch.rand(
                fractions.shape,
                generator=generator,
                device=group_sizes.device,
                dtype=torch.float64,
            ) * 1e-12

        for i in torch.argsort(fractions, descending=True).tolist():
            if remaining == 0:
                break
            if counts[i] < group_sizes[i]:
                counts[i] += 1
                remaining -= 1

    assert int(counts.sum().item()) == total_samples
    return counts


def compute_counts_per_conv_from_ber(model, msfp, ber, seed=None):
    return compute_counts_from_sizes(
        get_conv_weight_sizes(model),
        mant_bits=msfp.mantissa_bits,
        box_size=msfp.box_size,
        ber=ber,
        seed=seed,
    )


def compute_counts_from_sizes(conv_sizes, mant_bits, box_size, ber, seed=None):
    if not 0 <= float(ber) <= 1:
        raise ValueError("ber 應在 0 到 1 之間")
    if not 1 <= int(box_size):
        raise ValueError("box_size 應大於 0")
    if not 1 <= int(mant_bits):
        raise ValueError("mant_bits 應大於 0")

    conv_sizes = torch.as_tensor(conv_sizes, dtype=torch.int64)
    num_tiles = (conv_sizes + box_size - 1) // box_size
    bits_per_layer = num_tiles * 8 + conv_sizes * (mant_bits + 1)
    total_bits = int(bits_per_layer.sum().item())
    # float64 避免大型模型的 total_bits 在 float32 下失去低位精度。
    total_faults = int(torch.round(torch.tensor(total_bits, dtype=torch.float64) * ber).item())
    return fast_even_sample_counts(bits_per_layer, total_faults, seed=seed)


def warn_if_target_capacity_exceeded(conv_sizes, counts, bit_index, mant_bits, box_size):
    if mant_bits <= bit_index < mant_bits + 8:
        target_capacity = (conv_sizes + box_size - 1) // box_size
    else:
        target_capacity = conv_sizes
    over = counts > target_capacity
    if bool(over.any()):
        warnings.warn(
            f"Bit {bit_index} 的要求故障數有 {int(over.sum().item())} 層超過此 bit 的實際位置數；"
            "此程式保留原 VGG16 的 MSFP 總 bit BER 分配公式，不會偷偷截斷。"
            "若 msfp_converter 不支援重複抽樣，高 BER 可能發生錯誤，"
            "請縮小 BER 或明確決定新的注入規則。",
            stacklevel=2,
        )


def load_checkpoint_state(ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"找不到 checkpoint: {ckpt_path}\n"
            "請先執行對應模型的 CIFAR-10 訓練程式，或修改 CKPT_PATH。"
        )
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:  # 舊版 PyTorch
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "arch" in ckpt:
        if ckpt["arch"].lower() != "googlenet_cifar10":
            raise ValueError(
                f"checkpoint arch={ckpt['arch']!r}，預期 'googlenet_cifar10'；"
                "請確認 CKPT_PATH 是否指向正確模型。"
            )
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return state


def make_testloader(data_root, batch_size, num_workers, device):
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    testset = datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=test_tfm,
        download=False,
    )
    return DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )


@torch.no_grad()
def eval_top1(model, loader, device, show_progress=True):
    model.eval()
    correct, total = 0, 0
    iterator = tqdm(loader, desc="Inference", unit="batch", leave=False, disable=not show_progress)
    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.numel()
        if show_progress:
            iterator.set_postfix(acc=f"{correct / total:.4f}")
    return correct / total


def inject_conv_weights(model_quantized, counts_per_conv, msfp, bit_index, seed):
    conv_layers = get_conv_layers(model_quantized)
    if len(conv_layers) != len(counts_per_conv):
        raise RuntimeError(
            f"Conv layers={len(conv_layers)}，counts={len(counts_per_conv)} 不一致"
        )

    with torch.no_grad():
        for layer_index, (layer, fault_num) in enumerate(zip(conv_layers, counts_per_conv.tolist())):
            # 每一層用不同但可重現的 random stream；repeat 時也會換 seed。
            layer_seed = None if seed is None else int(seed) + layer_index * 100003
            injected = msfp.quantize_with_fault_injection(
                data=layer.weight.detach().clone(),
                index=int(bit_index),
                fault_num=int(fault_num),
                seed=layer_seed,
            )
            layer.weight.copy_(
                injected.to(device=layer.weight.device, dtype=layer.weight.dtype)
            )


def run_one_fault_trial(state, counts_per_conv, msfp, bit_index, seed, loader, device, show_progress=True):
    model = build_googlenet_cifar10(num_classes=10)
    model.load_state_dict(state, strict=True)
    insert_quant_before_conv(model, lambda: MSFPQuantizeLayer(msfp))

    wrapped_conv_count = len(get_conv_layers(model))
    if wrapped_conv_count != 57:
        raise RuntimeError(f"Conv2d 數量異常：{wrapped_conv_count}，應為 57")

    model.to(device).eval()
    inject_conv_weights(model, counts_per_conv, msfp, bit_index, seed)
    return eval_top1(model, loader, device, show_progress=show_progress)


def main(
    mant_bits, box_size, iter, ber, index, seed=None,
    data_root=DATA_ROOT, ckpt_path=CKPT_PATH, batch_size=256, num_workers=0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_testloader(data_root, batch_size, num_workers, device)
    state = load_checkpoint_state(ckpt_path)

    reference = build_googlenet_cifar10(num_classes=10)
    reference.load_state_dict(state, strict=True)
    conv_sizes = get_conv_weight_sizes(reference)
    if len(conv_sizes) != 57:
        raise RuntimeError(f"預期 57 個 Conv2d，實際 {len(conv_sizes)} 個")
    del reference

    results = []
    for r in range(iter):
        trial_seed = None if seed is None else int(seed) + r
        msfp = MSFPConverter(mantissa_bits=mant_bits, box_size=box_size, verbose=False)
        counts = compute_counts_from_sizes(conv_sizes, mant_bits, box_size, ber, seed=trial_seed)
        if r == 0:
            warn_if_target_capacity_exceeded(conv_sizes, counts, index, mant_bits, box_size)
        print(f"[{r + 1}/{iter}] BER={ber:g} | Bit {index} | faults={int(counts.sum())} | device={device}")
        acc = run_one_fault_trial(state, counts, msfp, index, trial_seed, loader, device)
        results.append(acc)
        print(f"Top-1 accuracy: {acc * 100:.2f}%")
    return results


def run_exponent_bit_sweep_to_csv(
    ber_list,
    bit_points=None,
    csv_name="GoogLeNet_CIFAR10_bit_sweep_new_SM.csv",
    mant_bits=7,
    box_size=16,
    repeats=1,
    base_seed=0,
    data_root=DATA_ROOT,
    ckpt_path=CKPT_PATH,
    batch_size=256,
    num_workers=0,
    show_progress=True,
):
    if bit_points is None:
        bit_points = [15, 6, 5, 4, 3, 2, 1, 0]
    if repeats < 1:
        raise ValueError("repeats 必須 >= 1")
    if not bit_points:
        raise ValueError("bit_points 不能為空")
    max_bit = mant_bits + 8   # 0~mantissa-1: mantissa; next 8: exponent; final: sign
    for bit in bit_points:
        if not 0 <= int(bit) <= max_bit:
            raise ValueError(f"bit_points 不可含 {bit}；目前應介於 0~{max_bit}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_testloader(data_root, batch_size, num_workers, device)
    state = load_checkpoint_state(ckpt_path) 
    reference = build_googlenet_cifar10(num_classes=10)
    reference.load_state_dict(state, strict=True)
    conv_sizes = get_conv_weight_sizes(reference)
    if len(conv_sizes) != 57:
        raise RuntimeError(f"預期 57 個 Conv2d，實際 {len(conv_sizes)} 個")
    del reference

    print(f"Model: GoogLeNet | Conv2d={len(conv_sizes)} | device={device}")
    print(f"mant_bits={mant_bits}, box_size={box_size}, repeats={repeats}")

    rows = []
    for ber in ber_list:
        print("\n" + "=" * 70)
        print(f"BER = {ber}")
        print("=" * 70)
        row = {"BER": ber}

        for bit in bit_points:
            print(f"\n{'-' * 24} Bit {bit} {'-' * 24}")
            accuracies = []
            warned = False

            for r in range(repeats):
                trial_seed = None if base_seed is None else int(base_seed) + r
                msfp = MSFPConverter(
                    mantissa_bits=mant_bits,
                    box_size=box_size,
                    verbose=False,
                )
                counts = compute_counts_from_sizes(
                    conv_sizes, mant_bits, box_size, ber=ber, seed=trial_seed
                )
                if not warned:
                    warn_if_target_capacity_exceeded(
                        conv_sizes, counts, bit, mant_bits, box_size
                    )
                    warned = True

                print(
                    f"BER={ber:g} | Bit {bit} | repeat {r + 1}/{repeats} "
                    f"| total faults={int(counts.sum())}"
                )
                acc = run_one_fault_trial(
                    state, counts, msfp, bit, trial_seed, loader, device,
                    show_progress=show_progress,
                )
                accuracies.append(acc)
                print(f"Top-1 accuracy: {acc * 100:.2f}%")

            row[f"Bit {bit}"] = float(np.mean(accuracies)) * 100.0

        rows.append(row)
        # 每一組 BER 完成就存檔，避免中途停止全部資料遺失。
        os.makedirs(os.path.dirname(os.path.abspath(csv_name)), exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_name, index=False)
        print(f"[Saved partial CSV] {csv_name}")

    result = pd.DataFrame(rows)
    result.to_csv(csv_name, index=False)
    print(f"\n[Done] {csv_name}")
    print(result)
    return result


@torch.no_grad()
def evaluate_clean_model(
    data_root=DATA_ROOT, ckpt_path=CKPT_PATH, batch_size=256, num_workers=0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_testloader(data_root, batch_size, num_workers, device)
    state = load_checkpoint_state(ckpt_path)
    model = build_googlenet_cifar10(num_classes=10)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    acc = eval_top1(model, loader, device)
    print(f"Clean GoogLeNet-CIFAR10 Acc: {acc * 100:.2f}%")
    return acc


if __name__ == "__main__":
    run_exponent_bit_sweep_to_csv(
        ber_list=[
            5e-4,
            1e-3, 3e-3, 5e-3, 7e-3,
            1e-2, 3e-2, 5e-2, 7e-2,
            1e-1, 3e-1,
        ],
        bit_points=[15, 6, 5, 4, 3, 2, 1, 0],
        csv_name="GoogLeNet_CIFAR10_bit_sweep_new_SM.csv",
        mant_bits=7,
        box_size=16,
        repeats=25,
        base_seed=0,
        data_root=DATA_ROOT,
        ckpt_path=CKPT_PATH,
        batch_size=256,
        num_workers=0,
    )
