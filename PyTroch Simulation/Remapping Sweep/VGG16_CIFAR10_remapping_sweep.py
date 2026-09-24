import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg16, vgg16_bn

from msfp_converter import MSFPConverter
from ResNet18_for_loop import (
    compute_counts_per_conv_from_ber,
    eval_top1,
    inject_faults_conv_by_counts,
    inject_faults_conv_with_remapping_inter,
    inject_faults_conv_with_remapping_intra,
)


METHODS = {
    "fault": "Fault (Without Remapping)",
    "inter": "Inter-bank-XOR",
    "intra": "Intra-bank",
    "inter_mask": "Inter-bank-XOR + masking",
    "intra_mask": "Intra-bank + masking",
}


def parse_ber_list(text):
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals or any(x < 0 for x in vals):
        raise argparse.ArgumentTypeError("BER list must contain non-negative numbers")
    return vals


def build_vgg(name, num_classes=10):
    if name == "vgg16":
        model = vgg16(weights=None)
    elif name == "vgg16_bn":
        model = vgg16_bn(weights=None)
    else:
        raise ValueError(name)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def clean_state_dict(state):
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def load_checkpoint(model, path):
    state = clean_state_dict(torch.load(path, map_location="cpu"))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint does not match the selected torchvision architecture. "
            "Try --arch vgg16 versus --arch vgg16_bn. If the checkpoint came "
            "from a custom VGG class, replace build_vgg() with that exact class.\n\n"
            f"Original error:\n{exc}"
        ) from exc


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_faulty_model(method, model, msfp, counts, seed, device, args):
    common = dict(
        model=model, msfp=msfp, counts_per_conv=counts, seed=seed,
        device=device, verbose=args.verbose,
    )
    if method == "fault":
        return inject_faults_conv_by_counts(**common, mask=False)[0]
    if method == "inter":
        return inject_faults_conv_with_remapping_inter(
            **common, mode="xor", split=True, mask=False)[0]
    if method == "intra":
        return inject_faults_conv_with_remapping_intra(
            **common, ms_group=args.ms_group, exp_group=args.exp_group,
            mask=False)[0]
    if method == "inter_mask":
        return inject_faults_conv_with_remapping_inter(
            **common, mode="xor", split=True, mask=True)[0]
    if method == "intra_mask":
        return inject_faults_conv_with_remapping_intra(
            **common, ms_group=args.ms_group, exp_group=args.exp_group,
            mask=True)[0]
    raise ValueError(method)


def save_rows(path, rows):
    fields = ["ber", "trial", "seed", "method", "accuracy", "accuracy_percent"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["ber"], row["method"]), []).append(row["accuracy_percent"])
    out = []
    for (ber, method), values in sorted(grouped.items()):
        a = np.asarray(values, dtype=float)
        out.append({
            "ber": ber, "method": method, "n": len(a),
            "mean_accuracy_percent": float(a.mean()),
            "std_accuracy_percent": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        })
    return out


def save_summary(path, rows):
    fields = ["ber", "method", "n", "mean_accuracy_percent", "std_accuracy_percent"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summary, output, title, error_bars=False):
    by_key = {(r["ber"], r["method"]): r for r in summary}
    bers = sorted({r["ber"] for r in summary})
    plt.figure(figsize=(13, 7))
    for method, label in METHODS.items():
        ys = [by_key[(b, method)]["mean_accuracy_percent"] for b in bers]
        stds = [by_key[(b, method)]["std_accuracy_percent"] for b in bers]
        if error_bars:
            plt.errorbar(bers, ys, yerr=stds, marker="o", capsize=2, label=label)
        else:
            plt.plot(bers, ys, marker="o", label=label)
    if all(b > 0 for b in bers):
        plt.xscale("log")
    plt.xlabel("BER (Bit Error Rate)")
    plt.ylabel("Model Accuracy (%)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.55)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "vgg16_remapping_raw.csv"
    summary_path = out_dir / "vgg16_remapping_summary.csv"
    plot_path = out_dir / "vgg16_remapping_curves.png"

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    model = build_vgg(args.arch).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.CIFAR10(root=args.data_root, train=False,
                               download=args.download, transform=tfm)
    loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    msfp = MSFPConverter(mantissa_bits=args.mantissa_bits,
                         box_size=args.box_size, verbose=False)

    rows = []
    total_jobs = len(args.bers) * args.trials * len(METHODS)
    job = 0
    for ber_index, ber in enumerate(args.bers):
        for trial in range(args.trials):
            trial_seed = args.base_seed + ber_index * 100_000 + trial
            seed_everything(trial_seed)
            counts = compute_counts_per_conv_from_ber(
                model, msfp, ber=ber, seed=trial_seed)
            print(f"\nBER={ber:g}, trial={trial + 1}/{args.trials}, "
                  f"fault_bits={int(counts.sum())}")

            # Same counts and seed => all five methods see the same physical faults.
            for method, label in METHODS.items():
                job += 1
                print(f"[{job}/{total_jobs}] {label}")
                tested = make_faulty_model(
                    method, model, msfp, counts, trial_seed, device, args)
                acc = eval_top1(tested, loader, device)
                rows.append({
                    "ber": ber, "trial": trial, "seed": trial_seed,
                    "method": method, "accuracy": acc,
                    "accuracy_percent": acc * 100.0,
                })
                save_rows(raw_path, rows)  # checkpoint after every expensive run
                del tested
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        summary = summarize(rows)
        save_summary(summary_path, summary)

    summary = summarize(rows)
    plot_results(summary, plot_path, args.title, args.error_bars)
    (out_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {raw_path}\nSaved: {summary_path}\nSaved: {plot_path}")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="trained VGG16 checkpoint")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--arch", choices=("vgg16", "vgg16_bn"), default="vgg16_bn")
    p.add_argument("--bers", type=parse_ber_list,
                   default=parse_ber_list("1e-8,1e-7,1e-6,1e-5,2e-5,3e-5,4e-5,"
                                          "5e-5,6e-5,8e-5,1e-4,2e-4,3e-4,4e-4,"
                                          "5e-4,6e-4,8e-4,1e-3"))
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=20260719)
    p.add_argument("--mantissa-bits", type=int, default=7)
    p.add_argument("--box-size", type=int, default=16)
    p.add_argument("--ms-group", type=int, default=1,
                   help="intra-bank MS tiles sharing one CW")
    p.add_argument("--exp-group", type=int, default=64,
                   help="intra-bank exponent tiles sharing one CW")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output-dir", default="./vgg16_remapping_results")
    p.add_argument("--title", default="VGG16 CIFAR-10: Faulty vs Inter-bank vs Intra-bank Remapping")
    p.add_argument("--error-bars", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
