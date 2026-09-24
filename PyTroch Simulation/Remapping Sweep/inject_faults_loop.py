import argparse
import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import itertools
import os

from msfp_converter import MSFPConverter
from tqdm import tqdm
from collections import OrderedDict
from collections import defaultdict
from typing import List, Dict, Tuple, Any
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
# ---------- Model: ResNet18 for CIFAR-10 ----------
def build_resnet18_cifar10(num_classes=10) -> nn.Module:
    m = resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
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
# ---------- Sampling helper (your original) ----------
def fast_even_sample_counts(group_sizes, total_samples, seed=None):
    if total_samples == group_sizes.sum():
        return group_sizes.clone()  # 全抽

    num_groups = group_sizes.size(0)
    if seed is not None:
        torch.manual_seed(seed)

    # Step 1: 平均分配
    base = total_samples // num_groups
    counts = torch.full_like(group_sizes, base)
    counts = torch.min(counts, group_sizes)

    # Step 2: 剩餘的部分用隨機分配
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
            counts = torch.min(counts, group_sizes)  # 再次保護別超過

    return counts

#########################################################################################
# ---------- Fault injection (total counts, no duplicates) ----------
def inject_random_faults_msfp_totalcounts(msfp, data, counts_total, seed=None, include=("man", "exp", "sign"), verbose=False):
    """
    msfp: 你的 MSFPConverter instance
    data: 權重 tensor (任意 shape)
    counts_total: 外部算好的總翻轉 bit 數
    include: 要包含哪些欄位 ("man","exp","sign") 任意子集合

    Fault model (對齊你現有 class):
      - mantissa: per-element 隨機 bit-flip (LSB有效 0..mantissa_bits-1)
      - exponent: per-tile(shared) 隨機 bit-flip，tile 範圍內全部元素一起翻 
      - sign: per-element 翻 0/1
    """

    if seed is not None:
        torch.manual_seed(seed)

    flat_w, w_shape = msfp.flatten_tensor(data.to(torch.float32))
    signs, exps, mants = msfp.float_to_msfp(flat_w)  # signs:int8(0/1), exps:int8, mants:int32  :contentReference[oaicite:3]{index=3}

    N = signs.numel()
    box = msfp.box_size
    m_bits = msfp.mantissa_bits
    num_tiles = (N + box - 1) // box  #向上取整

    # --- 定義「總 bit 空間」大小（依 MSFP 定義）---
    man_bits_total  = (N * m_bits) if ("man"  in include) else 0
    exp_bits_total  = (num_tiles * 8) if ("exp" in include) else 0
    sign_bits_total = (N * 1) if ("sign" in include) else 0

    total_bits = man_bits_total + exp_bits_total + sign_bits_total
    #if total_bits == 0:
    #    raise ValueError("include 至少要包含 man/exp/sign 其中一個")
    if counts_total > total_bits:
        raise ValueError(f"counts_total={counts_total} > total_bits={total_bits} "
                         f"(man={man_bits_total}, exp={exp_bits_total}, sign={sign_bits_total})")

    # --- 一次隨機挑全域 bit index（不重複）---
    picked = torch.randperm(total_bits, device=signs.device)[:counts_total]
    
    # --- mask 準備（一次 XOR 套用）---
    mant_mask = torch.zeros(N, dtype=torch.int32, device=signs.device)   # int32，對齊 mants dtype
    exp_mask  = torch.zeros(N, dtype=torch.uint8, device=signs.device)   # uint8 做 XOR
    sign_mask = torch.zeros(N, dtype=torch.uint8, device=signs.device)   # 0/1

    meta = {"man": [], "exp": [], "sign": []}

    def tile_range(tid: int):
        s = tid * box
        e = min(s + box, N)
        return s, e

    # --- 映射全域 index -> (欄位, 位置, bit) ---
    for g in picked.tolist():
        # 1) mantissa 區：大小 N*m_bits
        if g < man_bits_total:
            x = g
            w_idx = x // m_bits
            b = x % m_bits              # LSB 有效 bit 位置：0..m_bits-1
            mant_mask[w_idx] |= (1 << int(b))
            tile_id = int(w_idx) // box
            offset = int(w_idx) % box
            meta["man"].append((tile_id, offset, int(b)))
            continue

        # 2) exponent 區：大小 num_tiles*8
        if g < man_bits_total + exp_bits_total:
            x = g - man_bits_total
            tile_id = x // 8
            b = x % 8                   # 這裡 b=0 對應 shift=0 (LSB)；如你想 MSB-first 再改
            shift = int(b)
            s, e = tile_range(int(tile_id))
            exp_mask[s:e] |= (1 << shift)     # shared exponent：tile 範圍整段一起翻
            meta["exp"].append((int(tile_id), shift))
            continue

        # 3) sign 區：大小 N
        x = g - man_bits_total - exp_bits_total
        w_idx = x                       # sign 只有 1 bit/element
        sign_mask[w_idx] |= 1
        tile_id = int(w_idx) // box
        offset = int(w_idx) % box
        meta["sign"].append((tile_id, offset, 7))

    # --- 套用 fault ---
    before_s = signs.clone()
    before_e = exps.clone()
    before_m = mants.clone()

    mants = mants ^ mant_mask
    # XOR must operate on the raw 8-bit pattern, but the result must be cast
    # back to the converter's original signed dtype.  Keeping it as uint8
    # turns negative exponents (e.g. -5) into large positive values (251),
    # which makes msfp_to_float overflow the whole layer to Inf/NaN.
    exps  = (exps.to(torch.uint8) ^ exp_mask).to(before_e.dtype)
    signs = (signs.to(torch.uint8) ^ sign_mask).to(before_s.dtype)

    if verbose or msfp.verbose:
        print(f"[FaultInject] N={N}, box={box}, tiles={num_tiles}, m_bits={m_bits}")
        print(f"  total_bits={total_bits} (man={man_bits_total}, exp={exp_bits_total}, sign={sign_bits_total}), counts={counts_total}")
        print(f"  changed elems: man={(before_m!=mants).sum().item()}, exp={(before_e!=exps).sum().item()//box} | ({(before_e!=exps).sum().item()}), sign={(before_s!=signs).sum().item()}")
        #print(meta)  # 太多就先別印

    return signs, exps, mants, meta, w_shape, num_tiles
#########################################################################################
##Remapping (Inter & Intra)
def count_global_bit_errors(meta_layer, nbits=8):
    """
    meta_layer: 例如你貼的 dict: {"man":[(t,off,b),...], "sign":[(t,off,b),...], "exp":[(t,b),...]}
    回傳:
      man_bits, sign_bits, exp_bits, total_bits  (都長度=8)
    """
    man_bits  = [0] * nbits
    sign_bits = [0] * nbits
    exp_bits  = [0] * nbits

    # mantissa: (tile, offset, bit)
    for (_t, _off, b) in meta_layer["man"]:
        man_bits[int(b)] += 1

    # sign: (tile, offset, bit) 你已經存成 bit=7
    for (_t, _off, b) in meta_layer["sign"]:
        sign_bits[int(b)] += 1

    # exponent: (tile, bit)
    for (_t, b) in meta_layer["exp"]:
        exp_bits[int(b)] += 1

    total_bits = [man_bits[i] + sign_bits[i] + exp_bits[i] for i in range(nbits)]
    return man_bits, sign_bits, exp_bits, total_bits
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#inter
def score_perm(counts, scale, perm):
    """
    perm: tuple/list length 8
      perm[old_bit] = new_bit
    """
    s = 0
    for old_b in range(8):
        new_b = perm[old_b]
        s += counts[old_b] * scale[new_b]
    return s

def remap_counts_perm(counts, perm):
    """
    new_counts[new_bit] = sum of counts[old_bit] mapped to it
    """
    new = [0] * 8
    for old_b in range(8):
        new_b = perm[old_b]
        new[new_b] += counts[old_b]
    return new

def best_perm(counts, scale):
    """
    統一輸出格式：
      return best_perm_tuple, best_score, best_new_counts
    """
    best_p, best_s, best_new = None, None, None
    for perm in itertools.permutations(range(8)):  # 8! = 40320
        s = score_perm(counts, scale, perm)
        if best_s is None or s < best_s:
            best_s = s
            best_p = perm
            best_new = remap_counts_perm(counts, perm)
    return best_p, best_s, best_new

def score_swap(counts, scale, i, j):
    """
    只交換 new_bit=i 和 new_bit=j 的 scale 位置
    等價於：old_bit i 的錯誤搬到 new_bit j；old_bit j 的錯誤搬到 new_bit i
    其他 bit 不動
    """
    s = 0
    for b in range(8):
        if b == i:
            s += counts[b] * scale[j]
        elif b == j:
            s += counts[b] * scale[i]
        else:
            s += counts[b] * scale[b]
    return s

def remap_counts_swap(counts, i, j):
    """
    swap 後的新 counts（其實就是 counts 的 i/j 位置互換）
    """
    new = list(counts)
    new[i], new[j] = new[j], new[i]
    return new

def best_swap(counts, scale):
    best_pair = None
    best_score = None
    best_new = None

    for i in range(8):
        for j in range(i+1, 8):
            s = score_swap(counts, scale, i, j)
            if best_score is None or s < best_score:
                best_score = s
                best_pair = (i, j)
                best_new = remap_counts_swap(counts, i, j)

    return best_pair, best_score, best_new

def top_swaps(counts, scale, top_k=10):
    cand = []
    for i in range(8):
        for j in range(i+1, 8):
            s = score_swap(counts, scale, i, j)
            cand.append((s, i, j))
    cand.sort()
    return cand[:top_k]

def score_xor(counts, scale, cw):
    # new_bit = b ^ cw
    return sum(counts[b] * scale[b ^ cw] for b in range(8))

def remap_counts_xor(counts, cw):
    new = [0]*8
    for b in range(8):
        new[b ^ cw] += counts[b]
    return new

def score_rot(counts, scale, k):
    # new_bit = (b + k) % 8
    return sum(counts[b] * scale[(b + k) & 7] for b in range(8))

def remap_counts_rot(counts, k):
    new = [0]*8
    for b in range(8):
        new[(b + k) & 7] += counts[b]
    return new

def best_xor(counts, scale):
    best_cw, best_s, best_new = None, None, None
    for cw in range(8):
        s = score_xor(counts, scale, cw)
        if best_s is None or s < best_s:
            best_s = s
            best_cw = cw
            best_new = remap_counts_xor(counts, cw)
    return best_cw, best_s, best_new

def best_rot(counts, scale):
    best_k, best_s, best_new = None, None, None
    for k in range(8):
        s = score_rot(counts, scale, k)
        if best_s is None or s < best_s:
            best_s = s
            best_k = k
            best_new = remap_counts_rot(counts, k)
    return best_k, best_s, best_new

def baseline_score(counts, scale):
    return sum(scale[b] * counts[b] for b in range(8))
#######################################
def remap_meta_bitlane(msfp, old_meta, mode, param):
    """
    mode:
      - "xor":  param = cw (0..7)
      - "rot":  param = k  (0..7)
      - "swap": param = (i,j)
      - "perm": param = perm (length-8), perm[old_bit] = new_bit

    old_meta:
      man : (tile, off, bit)   bit in 0..m_bits-1
      sign: (tile, off, 7)
      exp : (tile, bit)        bit in 0..7
    """
    m_bits = msfp.mantissa_bits

    def _validate_perm(perm):
        if len(perm) != 8:
            raise ValueError(f"perm length must be 8, got {len(perm)}")
        perm = [int(x) for x in perm]
        if sorted(perm) != list(range(8)):
            raise ValueError(f"perm must be a permutation of 0..7, got {perm}")
        return perm

    perm = None
    if mode == "perm":
        perm = _validate_perm(param)

    def map_bit(b):
        b = int(b) & 7  # 確保在 0..7
        if mode == "xor":
            cw = int(param) & 7
            return b ^ cw
        elif mode == "rot":
            k = int(param) & 7
            return (b + k) & 7
        elif mode == "swap":
            i, j = param
            i, j = int(i) & 7, int(j) & 7
            if b == i: return j
            if b == j: return i
            return b
        elif mode == "perm":
            return perm[b]
        else:
            raise ValueError(f"Unknown mode={mode}, use 'xor'/'rot'/'swap'/'perm'")

    new_meta = {"man": [], "sign": [], "exp": []}

    # ---- man ----
    for (t, off, b) in old_meta.get("man", []):
        nb = map_bit(b)
        if nb == 7:
            new_meta["sign"].append((int(t), int(off), 7))
        else:
            if nb >= m_bits:
                raise ValueError(
                    f"remap produced mantissa bit {nb} >= m_bits={m_bits} "
                    f"(mode={mode}, param={param})"
                )
            new_meta["man"].append((int(t), int(off), int(nb)))

    # ---- sign (原本也是 bit=7，但允許被 perm/xor/rot/swap 送到 0..6) ----
    for (t, off, b) in old_meta.get("sign", []):
        nb = map_bit(b)
        if nb == 7:
            new_meta["sign"].append((int(t), int(off), 7))
        else:
            if nb >= m_bits:
                raise ValueError(
                    f"remap produced mantissa bit {nb} >= m_bits={m_bits} "
                    f"(mode={mode}, param={param})"
                )
            new_meta["man"].append((int(t), int(off), int(nb)))

    # ---- exp ----
    for (t, b) in old_meta.get("exp", []):
        nb = map_bit(b)
        new_meta["exp"].append((int(t), int(nb)))

    return new_meta
####################################################################################################################
def remap_meta_split_via_existing(msfp, old_meta,
                                 ms_mode, ms_param,
                                 exp_mode=None, exp_param=None):
    """
    用你原本的 remap_meta_bitlane() 來做到：
      - man+sign 用 (ms_mode, ms_param)
      - exp 用 (exp_mode, exp_param)；若 exp_mode=None，exp 不動
    """

    # 只保留 man+sign
    old_ms = {
        "man":  old_meta.get("man", []),
        "sign": old_meta.get("sign", []),
        "exp":  []  # 不在這裡處理 exp
    }

    # 只保留 exp
    old_exp = {
        "man":  [],
        "sign": [],
        "exp":  old_meta.get("exp", [])
    }

    # 1) remap mantissa+sign
    new_ms = remap_meta_bitlane(msfp, old_ms, mode=ms_mode, param=ms_param)

    # 2) remap exponent（可選）
    if exp_mode is None:
        new_exp = old_exp  # exp 不動
    else:
        new_exp = remap_meta_bitlane(msfp, old_exp, mode=exp_mode, param=exp_param)

    # 3) 合併回一個 new_meta
    new_meta = {
        "man":  new_ms["man"],
        "sign": new_ms["sign"],
        "exp":  new_exp["exp"],
    }
    return new_meta

def inject_faults_msfp_from_meta(msfp, data, meta, seed=None, verbose=False):
    """
    msfp: MSFPConverter instance
    data: 原始權重 tensor (任意 shape)
    meta: {"man":[(tile,off,bit),...], "sign":[(tile,off,7),...], "exp":[(tile,bit),...]}
          - man bit: 0..m_bits-1
          - sign bit: 固定 7（你存成 7）
          - exp bit: 0..7
    回傳: signs, exps, mants, meta(原樣回傳), w_shape
    """
    if seed is not None:
        torch.manual_seed(seed)

    flat_w, w_shape = msfp.flatten_tensor(data.to(torch.float32))
    signs, exps, mants = msfp.float_to_msfp(flat_w)

    N = signs.numel()
    box = msfp.box_size
    m_bits = msfp.mantissa_bits
    num_tiles = (N + box - 1) // box

    # masks
    mant_mask = torch.zeros(N, dtype=torch.int32, device=signs.device)
    exp_mask  = torch.zeros(N, dtype=torch.uint8, device=signs.device)
    sign_mask = torch.zeros(N, dtype=torch.uint8, device=signs.device)

    def widx(tile, off):
        return tile * box + off

    def tile_range(tid):
        s = tid * box
        e = min(s + box, N)
        return s, e

    # --- mantissa: XOR（注意用 XOR 才能支援「同一 bit 出現偶數次會抵消」）---
    for (t, off, b) in meta.get("man", []):
        if not (0 <= b < m_bits):
            raise ValueError(f"mantissa bit out of range: b={b}, m_bits={m_bits}")
        idx = widx(int(t), int(off))
        if idx < 0 or idx >= N:
            continue
        mant_mask[idx] |= (1 << int(b))

    # --- sign: XOR 翻 0/1 ---
    for (t, off, _b) in meta.get("sign", []):
        idx = widx(int(t), int(off))
        if idx < 0 or idx >= N:
            continue
        sign_mask[idx] |= 1

    # --- exponent: per-tile shared, 整段 XOR ---
    for (t, b) in meta.get("exp", []):
        shift = int(b)
        if not (0 <= shift < 8):
            raise ValueError(f"exponent bit out of range: b={shift}")
        s, e = tile_range(int(t))
        exp_mask[s:e] |= (1 << shift)

    # if len(meta["exp"]) > 0:
    #     t, b = meta["exp"][0]
    #     s, e = tile_range(t)
    #     print(exps[s:e])        

    before_s = signs.clone()
    before_e = exps.clone()
    before_m = mants.clone()

    mants = mants ^ mant_mask
    # Same dtype restoration is required on the remapping reinjection path.
    exps  = (exps.to(torch.uint8) ^ exp_mask).to(before_e.dtype)
    signs = (signs.to(torch.uint8) ^ sign_mask).to(before_s.dtype)

    if verbose or getattr(msfp, "verbose", False):
        # print("\n" + "-"*30 + " Injection Verification " + "-"*30)
        # # 驗證第一個尾數故障點
        # if len(meta.get("man", [])) > 0:
        #     t, off, b = meta["man"][0]
        #     idx = t * msfp.box_size + off
        #     # 觀察該位置在注入前後的位元差異
        #     diff = before_m[idx] ^ mants[idx]
        #     print(f"[Mantissa] Tile {t}, Offset {off}")
        #     print(f"  Target Bit: {b}")
        #     print(f"  Actual XOR Mask: {bin(diff)} (Should be {bin(1 << b)})")

        # # 驗證第一個指數故障點
        # if len(meta.get("exp", [])) > 0:
        #     t, b = meta["exp"][0]
        #     idx = t * msfp.box_size # 指數是 shared，檢查 tile 開頭即可
        #     diff = before_e[idx] ^ exps[idx]
        #     print(f"[Exponent] Tile {t}")
        #     print(f"  Target Bit: {b}")
        #     print(f"  Actual XOR Mask: {bin(diff)} (Should be {bin(1 << b)})")
        # print("-"*84 + "\n")

        
        diff_m = (before_m != mants).sum().item()
        diff_s = (before_s != signs).sum().item()
        diff_e = (before_e != exps).sum().item()
        print(f"[InjectFromMeta] N={N}, box={box}, tiles={num_tiles}, m_bits={m_bits}")
        print(f"  changed elems: man={diff_m}, exp_tiles~={diff_e//box} | ({diff_e}), sign={diff_s}")

        

    return signs, exps, mants, meta, w_shape

@torch.no_grad()
def inject_faults_conv_with_remapping_inter(model, msfp, counts_per_conv, seed=None, device="cuda", verbose=False, mode="", split=True, mask=False):
    """
    counts_per_conv: 1D tensor/list, length = number of Conv2d layers.
    每一層 conv 使用 counts_per_conv[i] 當作 counts_total 進行注入。
    """
    m = copy.deepcopy(model).to(device).eval()
    
    """插入量化層到 Conv2d 前"""
    quant_layer = lambda: MSFPQuantizeLayer(msfp)
    insert_quant_before_conv(m, quant_layer)

    conv_layers = [layer for layer in m.modules() if isinstance(layer, nn.Conv2d)]

    if torch.is_tensor(counts_per_conv):
        counts_list = counts_per_conv.detach().cpu().tolist()
    else:
        counts_list = list(counts_per_conv)

    if len(conv_layers) != len(counts_list):
        raise ValueError(f"Conv layers={len(conv_layers)} != counts len={len(counts_list)}")

    remap_conv_metas = []
    for i, layer in enumerate(conv_layers):
        c = int(counts_list[i])

        if verbose:
            print(f"[Conv {i:02d}] weight={tuple(layer.weight.shape)}, counts={c}")

        w = layer.weight.data  # [out,in,kh,kw]

        if c <= 0:
            flat_w, w_shape = msfp.flatten_tensor(w.detach().to(torch.float32))
            q_s, q_e, q_m = msfp.float_to_msfp(flat_w)
            q_flat = msfp.msfp_to_float(q_s, q_e, q_m)
            layer.weight.data.copy_(msfp.unflatten_tensor(q_flat, w_shape).to(device))
            remap_conv_metas.append({"man": [], "exp": [], "sign": []})
            continue

        signs, exps, mants, meta, w_shape, num_tiles = inject_random_faults_msfp_totalcounts(
            msfp=msfp,
            data=w.detach(),   # converter 若支援 GPU 可拿掉 .cpu()
            counts_total=c,
            seed=None if seed is None else seed + i,
            include=("man", "exp", "sign"),
            verbose=False,
        )

        #remapping相關計算及inter-bank演算法
        #針對remapping結果重新注入fault
        # ms_scale = [1, 1, 2, 3, 4, 8, 10, 6]
        # exp_scale  = [1, 2, 8, 8, 8, 8, 8, 64]

        ms_scale = [1, 1, 1, 1, 4, 8, 16, 16]
        exp_scale  = [1, 16, 32, 4, 4, 4, 4, 64]
        man_bits, sign_bits, exp_bits, total_fault_bits = count_global_bit_errors(meta_layer=meta, nbits=8)
        ms_counts = [man_bits[i] + sign_bits[i] for i in range(8)]
        exp_counts = exp_bits

        if(split == False):
        #一起處理
            if mode == "xor":
                print(f"Perform XOR {i+1} times")
                cw, s_xor, new_xor = best_xor(total_fault_bits, scale=exp_scale)
                remap_meta_xor = remap_meta_bitlane(msfp=msfp, old_meta=meta, mode="xor", param=cw)
                remap_signs, remap_exps, remap_mants, remap_meta, _ = inject_faults_msfp_from_meta(msfp=msfp, data=w.detach(), meta=remap_meta_xor, seed=seed, verbose=verbose)
                print(f"Original count: {total_fault_bits}")
                print(f"[XOR] best cw={cw}, score={s_xor}, new_counts={new_xor}")

            elif mode == "rot":
                print(f"Perform Rotate {i+1} times")
                k,  s_rot, new_rot = best_rot(total_fault_bits, scale=exp_scale)
                remap_meta_rot = remap_meta_bitlane(msfp=msfp, old_meta=meta, mode="rot", param=k)
                remap_signs, remap_exps, remap_mants, remap_meta, _ = inject_faults_msfp_from_meta(msfp=msfp, data=w.detach(), meta=remap_meta_rot, seed=seed, verbose=verbose)
                print(f"Original count: {total_fault_bits}")
                print(f"[ROT] best k={k}, score={s_rot}, new_counts={new_rot}")

            elif mode == "swap":
                print(f"Perform swap {i+1} times")
                pair, s_best, new_counts = best_swap(total_fault_bits, scale=exp_scale)
                remap_meta_swap = remap_meta_bitlane(msfp=msfp, old_meta=meta, mode="swap", param=pair)
                remap_signs, remap_exps, remap_mants, remap_meta, _ = inject_faults_msfp_from_meta(msfp=msfp, data=w.detach(), meta=remap_meta_swap, seed=seed, verbose=verbose)
                print(f"Original count: {total_fault_bits}")
                print("Best swap (i,j):", pair, "score:", s_best)
                print("New counts after swap:", new_counts)

            elif mode == "perm":
                print(f"Perform permutation (8!) {i+1} times")
                best_p, best_s, best_new = best_perm(total_fault_bits, scale=exp_scale)
                remap_meta_permute = remap_meta_bitlane(msfp=msfp, old_meta=meta, mode="perm", param=best_p)
                remap_signs, remap_exps, remap_mants, remap_meta, _ = inject_faults_msfp_from_meta(msfp=msfp, data=w.detach(), meta=remap_meta_permute, seed=seed, verbose=verbose)
                print(f"Original count: {total_fault_bits}")
                print("Best permute: ", best_p, "score: ", best_s)
                print(f"New count: {best_new}")
        
            else:
                print("Not using Remapping")
                remap_signs = signs
                remap_exps = exps
                remap_mants = mants
                remap_meta = meta

        else:
            #print("sign and mantissa will process together, but exponent process seperatly")
            if mode == "xor":
                ms_cw, ms_s_xor, ms_new_xor = best_xor(ms_counts, ms_scale)
                exp_cw, exp_s_xor, exp_new_xor = best_xor(exp_counts, exp_scale)
                remap_meta_xor = remap_meta_split_via_existing(msfp, old_meta=meta, ms_mode="xor", ms_param=ms_cw, exp_mode="xor", exp_param=exp_cw)
                remap_signs, remap_exps, remap_mants, remap_meta, _ = inject_faults_msfp_from_meta(msfp=msfp, data=w.detach(), meta=remap_meta_xor, seed=seed, verbose=verbose)

            elif mode == "rot":
                ms_k,  ms_s_rot, ms_new_rot = best_rot(ms_counts, ms_scale)
                exp_k,  exp_s_rot, exp_new_rot = best_rot(exp_counts, exp_scale)
                remap_meta_rot = remap_meta_split_via_existing(msfp, old_meta=meta, ms_mode="rot", ms_param=ms_k, exp_mode="rot", exp_param=exp_k)
                remap_signs, remap_exps, remap_mants, remap_meta, _ = inject_faults_msfp_from_meta(msfp=msfp, data=w.detach(), meta=remap_meta_rot, seed=seed, verbose=verbose)

            elif mode == "swap":
                ms_pair, ms_s_best, ms_new_counts = best_swap(ms_counts, ms_scale)
                exp_pair, exp_s_best, exp_new_counts = best_swap(exp_counts, exp_scale)
                remap_meta_swap = remap_meta_split_via_existing(msfp, old_meta=meta, ms_mode="swap", ms_param=ms_pair, exp_mode="swap", exp_param=exp_pair)
                remap_signs, remap_exps, remap_mants, remap_meta, _ = inject_faults_msfp_from_meta(msfp=msfp, data=w.detach(), meta=remap_meta_swap, seed=seed, verbose=verbose)
            
            elif mode == "perm":
                ms_best_p, ms_best_s, ms_best_new = best_perm(ms_counts, ms_scale)
                exp_best_p, exp_best_s, exp_best_new = best_perm(exp_counts, exp_scale)
                remap_meta_permute = remap_meta_split_via_existing(msfp, old_meta=meta, ms_mode="perm", ms_param=ms_best_p, exp_mode="perm", exp_param=exp_best_p)
                remap_signs, remap_exps, remap_mants, remap_meta, _ = inject_faults_msfp_from_meta(msfp=msfp, data=w.detach(), meta=remap_meta_permute, seed=seed, verbose=verbose)
               
            else:
                print("Not using Remapping")
                remap_signs = signs
                remap_exps = exps
                remap_mants = mants
                remap_meta = meta

        # ==================== Inter-level Global Trace (ROT 驗證版) ====================
        if verbose and mode in ["xor", "rot", "swap", "perm"]:
            print("\n" + "="*45 + f" Inter-level Global Trace (Layer {i}) " + "="*45)
            
            # 1. 根據 mode 抓取對應的全局參數
            ms_p, exp_p = None, None
            if mode == "xor":
                ms_p, exp_p = ms_cw, exp_cw
            elif mode == "rot":
                ms_p, exp_p = ms_k, exp_k
            elif mode == "swap":
                ms_p, exp_p = ms_pair, exp_pair
            elif mode == "perm":
                ms_p, exp_p = ms_best_p, exp_best_p

            print(f"[Decision] Mode: {mode} | MS 參數: {ms_p} | EXP 參數: {exp_p}")
            
            # 2. 驗證 Mantissa / Sign (MS 區)
            if len(meta.get("man", [])) > 0:
                old_t, old_off, old_b = meta["man"][0]
                new_b = None
                # 從 remap_meta 搜尋座標對齊的點
                for nt, noff, nb in (remap_meta.get("man", []) + remap_meta.get("sign", [])):
                    if nt == old_t and noff == old_off:
                        new_b = nb
                        break
                
                print(f"[MS Trace]  座標({old_t}, {old_off}): 原位元 {old_b} -> 映射後 {new_b}")
                
                # --- ROT 驗證邏輯 ---
                if mode == "rot" and ms_p is not None and new_b is not None:
                    # 公式: (old_b + k) % 8
                    expected_rot = (old_b + ms_p) % 8
                    print(f"   ROT 驗證: ({old_b} + {ms_p}) % 8 = {expected_rot} ({'OK' if new_b == expected_rot else 'FAIL'})")
                # 保留 XOR 驗證供切換使用
                elif mode == "xor" and ms_p is not None and new_b is not None:
                    print(f"   XOR 驗證: {old_b} ^ {ms_p} = {old_b ^ ms_p} ({'OK' if new_b == (old_b ^ ms_p) else 'FAIL'})")

            # 3. 驗證 Exponent (EXP 區)
            if len(meta.get("exp", [])) > 0:
                old_et, old_eb = meta["exp"][0]
                new_eb = None
                for nt, nb in remap_meta.get("exp", []):
                    if nt == old_et:
                        new_eb = nb
                        break
                
                print(f"[EXP Trace] Tile {old_et}: 原位元 {old_eb} -> 映射後 {new_eb}")
                
                if mode == "rot" and exp_p is not None and new_eb is not None:
                    expected_exp_rot = (old_eb + exp_p) % 8
                    print(f"   ROT 驗證: ({old_eb} + {exp_p}) % 8 = {expected_exp_rot} ({'OK' if new_eb == expected_exp_rot else 'FAIL'})")
                elif mode == "xor" and exp_p is not None and new_eb is not None:
                    print(f"   XOR 驗證: {old_eb} ^ {exp_p} = {old_eb ^ exp_p} ({'OK' if new_eb == (old_eb ^ exp_p) else 'FAIL'})")
            
            print("="*105 + "\n")

            old_count = len(meta['man']) + len(meta['sign']) + len(meta['exp'])
            new_count = len(remap_meta['man']) + len(remap_meta['sign']) + len(remap_meta['exp'])

            if old_count == new_count:
                print(f"數量驗證成功：重排前後總點數皆為 {new_count}")
            else:
                print(f"警告：點數不匹配！原始 {old_count} vs 重排後 {new_count}")
        # =========================================================================================
        
        #exp masking
        if mask == True:
            remap_exps = exp_masking(remap_exps)

        # ---- convert back to float (flat) then reshape to weight shape ----
        w_fault_flat = msfp.msfp_to_float(remap_signs, remap_exps, remap_mants)  # 1D float
        w_fault = msfp.unflatten_tensor(w_fault_flat, w_shape)
        #w_fault = w_fault_flat.view(w_shape)

        layer.weight.data.copy_(w_fault.to(device))
        remap_conv_metas.append(remap_meta)

    return m, remap_conv_metas


#########################################################################################
@torch.no_grad()
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

    num_samples = int(round(int(total_bits.item()) * float(ber)))
    if num_samples == 0:
        return torch.zeros_like(bits_per_layer, dtype=torch.int64)

    # Draw a global physical-bit population proportionally to each layer's
    # capacity.  The old implementation first assigned an equal count to every
    # layer, which made small layers experience a much larger local BER.
    if seed is not None:
        torch.manual_seed(seed)
    probabilities = bits_per_layer.to(torch.float64)
    probabilities /= probabilities.sum()
    sampled_layers = torch.multinomial(
        probabilities, num_samples=num_samples, replacement=True)
    counts = torch.bincount(sampled_layers, minlength=len(bits_per_layer))

    if torch.any(counts > bits_per_layer):
        raise RuntimeError("Sampled fault count exceeded a layer's bit capacity")

    return counts.to(torch.int64)


@torch.no_grad()
def inject_faults_conv_by_counts(model, msfp, counts_per_conv, seed=0, device="cuda", verbose=False, mask=False):
    """
    counts_per_conv: 1D tensor/list, length = number of Conv2d layers.
    每一層 conv 使用 counts_per_conv[i] 當作 counts_total 進行注入。
    """
    m = copy.deepcopy(model).to(device).eval()
    
    """插入量化層到 Conv2d 前"""
    quant_layer = lambda: MSFPQuantizeLayer(msfp)
    insert_quant_before_conv(m, quant_layer)

    conv_layers = [layer for layer in m.modules() if isinstance(layer, nn.Conv2d)]

    if torch.is_tensor(counts_per_conv):
        counts_list = counts_per_conv.detach().cpu().tolist()
    else:
        counts_list = list(counts_per_conv)

    if len(conv_layers) != len(counts_list):
        raise ValueError(f"Conv layers={len(conv_layers)} != counts len={len(counts_list)}")

    conv_metas = []
    for i, layer in enumerate(conv_layers):
        c = int(counts_list[i])

        if verbose:
            print(f"[Conv {i:02d}] weight={tuple(layer.weight.shape)}, counts={c}")

        w = layer.weight.data  # [out,in,kh,kw]

        if c <= 0:
            flat_w, w_shape = msfp.flatten_tensor(w.detach().to(torch.float32))
            q_s, q_e, q_m = msfp.float_to_msfp(flat_w)
            q_flat = msfp.msfp_to_float(q_s, q_e, q_m)
            layer.weight.data.copy_(msfp.unflatten_tensor(q_flat, w_shape).to(device))
            conv_metas.append({"man": [], "exp": [], "sign": []})
            continue

        signs, exps, mants, meta, w_shape, num_tiles = inject_random_faults_msfp_totalcounts(
            msfp=msfp,
            data=w.detach(),   # converter 若支援 GPU 可拿掉 .cpu()
            counts_total=c,
            seed=None if seed is None else seed + i,
            include=("man", "exp", "sign"),
            verbose=False,
        )

        #exponent masking
        if mask == True:
            exps = exp_masking(exps)

        # ---- convert back to float (flat) then reshape to weight shape ----
        w_fault_flat = msfp.msfp_to_float(signs, exps, mants)  # 1D float
        w_fault = msfp.unflatten_tensor(w_fault_flat, w_shape)
        #w_fault = w_fault_flat.view(w_shape)

        layer.weight.data.copy_(w_fault.to(device))
        conv_metas.append(meta)

    return m, conv_metas
#########################################################################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#intra
# ------------------------------------------------------------
# helper: 從 counts 找最佳 cw（XOR mapping new = old ^ cw）
# ------------------------------------------------------------
def _best_cw_from_counts(counts: List[int], scale: List[int]) -> Tuple[int, int]:
    best_cw, best_s = 0, None
    # 小查表：scale_xor[cw][b] = scale[b ^ cw]
    scale_xor = [[scale[b ^ cw] for b in range(8)] for cw in range(8)]

    for cw in range(8):
        sx = scale_xor[cw]
        s = 0
        for b in range(8):
            s += counts[b] * sx[b]
        if best_s is None or s < best_s:
            best_s = s
            best_cw = cw
    return best_cw, best_s

# ------------------------------------------------------------
# 1) 先把 meta 轉成 per-tile counts（只做一次 O(#events)）
# ------------------------------------------------------------
def _build_tile_counts_ms(old_meta: Dict[str, list], num_tiles: int) -> List[List[int]]:
    """
    tile_counts_ms[t][b] = 該 tile 中 man/sign 事件在 bit b(0..7) 的數量
      - man: 用 tuple 的 bit (0..m_bits-1)，視為 0..7 的 bit lane
      - sign: 一律算到 bit7
    """
    tc = [[0]*8 for _ in range(num_tiles)]
    for (t, off, b) in old_meta.get("man", []):
        t = int(t); b = int(b) & 7
        if 0 <= t < num_tiles:
            tc[t][b] += 1
    for (t, off, b) in old_meta.get("sign", []):
        t = int(t)
        if 0 <= t < num_tiles:
            tc[t][7] += 1
    return tc


def _build_tile_counts_exp(old_meta: Dict[str, list], num_tiles: int) -> List[List[int]]:
    """
    tile_counts_exp[t][b] = 該 tile 中 exp 事件在 bit b(0..7) 的數量
    """
    tc = [[0]*8 for _ in range(num_tiles)]
    for (t, b) in old_meta.get("exp", []):
        t = int(t); b = int(b) & 7
        if 0 <= t < num_tiles:
            tc[t][b] += 1
    return tc


# ------------------------------------------------------------
# 2) 依 window_size 分段（stride=W），每段選最佳 cw
#    回傳 cw_by_tile（每個 tile 對應一個 cw）+ logs
# ------------------------------------------------------------
def _cw_by_tile_from_tilecounts(tile_counts: List[List[int]],
                                num_tiles: int,
                                window_size: int,
                                scale: List[int]) -> Tuple[List[int], List[Dict[str, Any]]]:
    cw_by_tile = [0] * num_tiles
    logs = []

    for tile_lo in range(0, num_tiles, window_size):
        tile_hi = min(tile_lo + window_size - 1, num_tiles - 1)

        win_counts = [0]*8
        for t in range(tile_lo, tile_hi + 1):
            row = tile_counts[t]
            # sum 8 bins
            for b in range(8):
                win_counts[b] += row[b]

        cw, best_s = _best_cw_from_counts(win_counts, scale)
        # 加這行 Debug
        #print(f"DEBUG: Window {tile_lo}-{tile_hi} assigned CW: {cw}")


        for t in range(tile_lo, tile_hi + 1):
            cw_by_tile[t] = cw

        logs.append({
            "range": (tile_lo, tile_hi),
            "cw": cw,
            "score": best_s,
            "counts": win_counts
        })

    return cw_by_tile, logs


# ------------------------------------------------------------
# 3) 最後「一次遍歷 meta」直接建立 new_meta（old_meta 不動）
# ------------------------------------------------------------
def sweep_windows_ms_xor_fast(old_meta, num_tiles, window_size, scale_ms, m_bits):
    """
    快版 non-inplace：
      - 不 deepcopy
      - 不逐 window 掃 meta
      - 回傳 new_meta, logs
    """
    tile_counts_ms = _build_tile_counts_ms(old_meta, num_tiles)
    cw_by_tile, logs = _cw_by_tile_from_tilecounts(tile_counts_ms, num_tiles, window_size, scale_ms)

    new_man = []
    new_sign = []

    # 要Debug再用
    
    # for (t, off, b) in old_meta.get("man", []):
    #     if t == 9 and off == 6:
    #         print(f"DEBUG: 進入 XOR 前的 b 是 {b}")
    #     nb = b ^ cw_by_tile[t]
    #     if t == 9 and off == 6:
    #         print(f"DEBUG: XOR 運算後的 nb 是 {nb} (使用 cw={cw_by_tile[t]})")

    # man
    for (t, off, b) in old_meta.get("man", []):
        t = int(t); off = int(off); b = int(b)
        cw = cw_by_tile[t] if 0 <= t < num_tiles else 0
        nb = (b & 7) ^ (cw & 7)

        if nb == 7:
            new_sign.append((t, off, 7))
        else:
            if nb >= m_bits:
                raise ValueError(f"mantissa bit {nb} >= m_bits={m_bits} after XOR cw={cw} (tile={t},off={off})")
            new_man.append((t, off, nb))

    # sign（原本 bit=7）
    for (t, off, _b) in old_meta.get("sign", []):
        t = int(t); off = int(off)
        cw = cw_by_tile[t] if 0 <= t < num_tiles else 0
        nb = 7 ^ (cw & 7)

        if nb == 7:
            new_sign.append((t, off, 7))
        else:
            if nb >= m_bits:
                raise ValueError(f"mantissa bit {nb} >= m_bits={m_bits} after XOR cw={cw} from sign (tile={t},off={off})")
            new_man.append((t, off, nb))

    # exp 原樣先放回去（ms sweep 不動 exp）
    new_meta = {
        "man": new_man,
        "sign": new_sign,
        "exp": list(old_meta.get("exp", [])),  # 直接複製 list（不改內容），很便宜
    }
    return new_meta, logs


def sweep_windows_exp_xor_fast(old_meta, num_tiles, window_size, scale_exp):
    """
    快版 non-inplace：
      - 不 deepcopy
      - 回傳 new_meta, logs
    注意：只 remap exp，不動 man/sign
    """
    tile_counts_exp = _build_tile_counts_exp(old_meta, num_tiles)
    cw_by_tile, logs = _cw_by_tile_from_tilecounts(tile_counts_exp, num_tiles, window_size, scale_exp)

    new_exp = []
    for (t, b) in old_meta.get("exp", []):
        t = int(t); b = int(b)
        cw = cw_by_tile[t] if 0 <= t < num_tiles else 0
        nb = (b & 7) ^ (cw & 7)
        new_exp.append((t, nb))

    new_meta = {
        "man": list(old_meta.get("man", [])),
        "sign": list(old_meta.get("sign", [])),
        "exp": new_exp,
    }
    return new_meta, logs

@torch.no_grad()
def inject_faults_conv_with_remapping_intra(model, msfp, counts_per_conv, ms_group, exp_group, seed=None, device="cuda", verbose=False, mask=False):
    """
    counts_per_conv: 1D tensor/list, length = number of Conv2d layers.
    每一層 conv 使用 counts_per_conv[i] 當作 counts_total 進行注入。
    """
    m = copy.deepcopy(model).to(device).eval()
    
    """插入量化層到 Conv2d 前"""
    quant_layer = lambda: MSFPQuantizeLayer(msfp)
    insert_quant_before_conv(m, quant_layer)

    conv_layers = [layer for layer in m.modules() if isinstance(layer, nn.Conv2d)]

    if torch.is_tensor(counts_per_conv):
        counts_list = counts_per_conv.detach().cpu().tolist()
    else:
        counts_list = list(counts_per_conv)

    if len(conv_layers) != len(counts_list):
        raise ValueError(f"Conv layers={len(conv_layers)} != counts len={len(counts_list)}")

    remap_conv_metas = []
    for i, layer in enumerate(conv_layers):
        c = int(counts_list[i])

        if verbose:
            print(f"[Conv {i:02d}] weight={tuple(layer.weight.shape)}, counts={c}")

        w = layer.weight.data  # [out,in,kh,kw]

        if c <= 0:
            flat_w, w_shape = msfp.flatten_tensor(w.detach().to(torch.float32))
            q_s, q_e, q_m = msfp.float_to_msfp(flat_w)
            q_flat = msfp.msfp_to_float(q_s, q_e, q_m)
            layer.weight.data.copy_(msfp.unflatten_tensor(q_flat, w_shape).to(device))
            remap_conv_metas.append({"man": [], "exp": [], "sign": []})
            continue

        signs, exps, mants, meta, w_shape, num_tiles = inject_random_faults_msfp_totalcounts(
            msfp=msfp,
            data=w.detach(),   # converter 若支援 GPU 可拿掉 .cpu()
            counts_total=c,
            seed=None if seed is None else seed + i,
            include=("man", "exp", "sign"),
            verbose=False,
        )

        #remapping相關計算及intra-bank演算法
        #針對remapping結果重新注入fault
        ms_scale = [1, 1, 1, 1, 4, 8, 16, 16]
        exp_scale  = [1, 16, 32, 4, 4, 4, 4, 64]
        # man_bits, sign_bits, exp_bits, total_fault_bits = count_global_bit_errors(meta_layer=meta, nbits=8)
        # ms_counts = [man_bits[i] + sign_bits[i] for i in range(8)]
        # exp_counts = exp_bits
        #window = msfp.box_size // msfp.box_size

        ms_window_size = ms_group
        exp_window_size = exp_group

        meta1, ms_logs = sweep_windows_ms_xor_fast(old_meta=meta, num_tiles=num_tiles, window_size=ms_window_size, scale_ms=ms_scale, m_bits=msfp.mantissa_bits)

        # 再對 exp remap（用 meta1 當輸入）
        remap_meta_intra, exp_logs = sweep_windows_exp_xor_fast(meta1, num_tiles=num_tiles, window_size=exp_window_size, scale_exp=exp_scale)

        # remap_meta_intra, ms_logs = sweep_windows_ms_xor(old_meta=meta, num_tiles=num_tiles, window_size=window, scale_ms=ms_scale, m_bits=msfp.mantissa_bits)
        # remap_meta_intra, exp_logs = sweep_windows_exp_xor(old_meta=meta, num_tiles=num_tiles, window_size=8, scale_exp=exp_scale)
        
        # --- 插入以下 Debug 檢查碼 ---  要Debug再用
        # if i == 0:
        # print("\n" + "="*50)
        # print(f"DEBUG CHECK: Layer {i} Intra-Remapping Process")
            
        #     # 檢查 EXP (應該每 8 個 Tile 換一次 CW)
        # print("--- Exponent Windows (Expected 128 weights per CW) ---")
        # for log_idx in range(min(3, len(exp_logs))): # 印前三個 window
        #     log = exp_logs[log_idx]
        #     t_start, t_end = log['range']
        #     print(f"Window {log_idx}: Tiles {t_start:3d}-{t_end:3d} | CW: {log['cw']}")
            

        # # 我們直接去抓 ms_logs 對應出來的實際 Tile 分佈
        # for idx, log in enumerate(ms_logs[:4]): # 看前 4 個 Window
        #     t_start, t_end = log["range"]
        #     print(log["range"])
        #     num_tiles = t_end - t_start + 1
        #     cw = log['cw']
        #     print(f"Window {idx}: Tile {t_start} 到 {t_end} (共 {num_tiles} 個 Tile) | CW: {log['cw']}")
        # print("="*50 + "\n")
        
        if verbose: 
            print("\n" + "="*35 + " Bit-flip Remapping Trace " + "="*35)
            
            # 追蹤 Mantissa (尾數)
            if len(meta.get("man", [])) > 0:
                old_t, old_off, old_b = meta["man"][0]
                
                # --- 修正處：根據實際使用的 window_size 動態計算 ---
                # 假設你的 mantissa window_size 是固定值或變數，這裡要對齊
                ms_win_size = ms_window_size # 如果你現在是用 2，這裡就設 2
                win_idx = old_t // ms_win_size 
                
                # 安全檢查：確保 index 不會溢出
                if win_idx < len(ms_logs):
                    ms_cw = ms_logs[win_idx]['cw']
                    
                    # 搜尋新位元
                    new_b = None
                    all_new_points = remap_meta_intra["man"] + remap_meta_intra["sign"]
                    for t, off, b in all_new_points:
                        if t == old_t and off == old_off:
                            new_b = b
                            break

                    print(f"[Mantissa Trace] Tile {old_t}, Offset {old_off}:")
                    print(f"  原本隨機位元 (old_b): {old_b}")
                    print(f"  選用 XOR CW    (cw): {ms_cw}  (Binary: {ms_cw:03b})")
                    print(f"  映射後新位元 (new_b): {new_b}  (公式驗證: {old_b} ^ {ms_cw % 8} = {old_b ^ (ms_cw % 8)})")
                else:
                    print(f"[Error] win_idx {win_idx} out of ms_logs range {len(ms_logs)}")

            # 追蹤 Exponent (指數) 同理修正
            if len(meta.get("exp", [])) > 0:
                old_et, old_eb = meta["exp"][0]
                exp_win_size = exp_window_size
                exp_win_idx = old_et // exp_win_size
                
                if exp_win_idx < len(exp_logs):
                    exp_cw = exp_logs[exp_win_idx]['cw']
                    new_eb = None
                    for t, b in remap_meta_intra["exp"]:
                        if t == old_et: # 指數在同一個 tile 是唯一的，直接找 tile ID
                            new_eb = b
                            break

                    print(f"\n[Exponent Trace] Tile {old_et}:")
                    print(f"  原本隨機位元 (old_eb): {old_eb}")
                    print(f"  選用 XOR CW    (cw): {exp_cw}  (Binary: {exp_cw:03b})")
                    print(f"  映射後新位元 (new_eb): {new_eb}  (公式驗證: {old_eb} ^ {exp_cw % 8} = {old_eb ^ (exp_cw % 8)})")
            print("="*96 + "\n")

        remap_signs, remap_exps, remap_mants, remap_meta, _ = inject_faults_msfp_from_meta(msfp=msfp, data=w.detach(), meta=remap_meta_intra, seed=seed, verbose=False)

        #exponent masking
        if mask == True:
            remap_exps = exp_masking(remap_exps)

        # ---- convert back to float (flat) then reshape to weight shape ----
        w_fault_flat = msfp.msfp_to_float(remap_signs, remap_exps, remap_mants)  # 1D float
        w_fault = msfp.unflatten_tensor(w_fault_flat, w_shape)
        #w_fault = w_fault_flat.view(w_shape)

        layer.weight.data.copy_(w_fault.to(device))
        remap_conv_metas.append(remap_meta)

    return m, remap_conv_metas
#########################################################################################
def exp_masking(exp):
    """
    MSFP exponent MSB masking.

    正常 shared exponent 為負 int8，MSB 應為 1。
    如果 MSB fault 將其翻成正數，重新把 MSB 設成 1。

    例如：
        127  (0b01111111) -> -1  (0b11111111)
         96  (0b01100000) -> -32 (0b11100000)
    """
    original_dtype = exp.dtype

    # 暫時轉為 uint8，以 raw 8-bit pattern 操作
    exp_bits = exp.to(torch.uint8)

    # 強制將 MSB 設為 1
    exp_bits = exp_bits | 0x80

    # 回到 converter 原本使用的 signed dtype
    return exp_bits.to(original_dtype)
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
#########################################################################################
def main(mant_bits, box_size, iter, ber, seed=None, verbose=False, mode="", inter_mode="", mask=False):

    top1_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, mode: {mode}, inter_mode: {inter_mode}")
    print("Computed counts_per_conv from BER:", ber)

    # MSFP converter
    msfp = MSFPConverter(mantissa_bits=mant_bits, box_size=box_size)

    # CIFAR-10 mean/std（跟你訓練一致）
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    testset = datasets.CIFAR10(root="E:/python code/data", train=False, transform=tfm)
    classes = testset.classes

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),])

    # testset = datasets.CIFAR10(
    #     root="./data",
    #     train=False, transform=tfm)

    testloader = DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False)

    # Load model
    model = build_resnet18_cifar10(num_classes=10)
    ckpt = torch.load("D:/Anaconda/PythonCode/data/vgg16_cifar10_ckpt_best.pth", map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    for i in range(iter):
        # Get counts per conv
        counts_per_conv = compute_counts_per_conv_from_ber(model, msfp, ber=ber, seed=seed)

        # sanity check
        num_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        if counts_per_conv.numel() != num_conv:
            raise ValueError(f"counts_per_conv len={counts_per_conv.numel()} != num_conv={num_conv}")

    # if args.verbose:
    #     conv_sizes = [m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv2d)]
    #     for i, (sz, c) in enumerate(zip(conv_sizes, counts_per_conv.tolist())):
    #         print(f"  conv{i:02d}: weight_numel={sz}, counts={c}")

        if mode == "fault":
            # Faulty model inference
            #print(f"Iteration {i+1}, Run faulty injection")
            faulty_model, conv_metas = inject_faults_conv_by_counts(
                model=model,
                msfp=msfp,
                counts_per_conv=counts_per_conv,
                seed=seed,
                device=device,
                verbose=verbose,
                mask=mask
            )
            #x0, y0 = testset[0]
            acc = eval_top1(faulty_model, testloader, device)
            #print(f"[Info] Testset size: {len(testset)} | classes: {len(testset.classes)}")
            #print(f"[Info] One sample shape: {tuple(x0.shape)} | label(example0): {y0}")
            #print(f"[Result] Test Top-1 Acc: {acc:.6f}")
            top1_list.append(acc)

        elif mode == "inter":
            print(f"Iteration {i+1}, Run Inter-remapping")
            #Inter-remap inference
            split_model_msfp_remap, split_remap_conv_metas = inject_faults_conv_with_remapping_inter(
                model=model, msfp=msfp, counts_per_conv=counts_per_conv, 
                seed=seed, device=device, verbose=verbose, mode=inter_mode, split=True, mask=mask)
            #x0, y0 = testset[0]
            acc = eval_top1(split_model_msfp_remap, testloader, device)
            #print(f"[Info] Testset size: {len(testset)} | classes: {len(testset.classes)}")
            #print(f"[Info] One sample shape: {tuple(x0.shape)} | label(example0): {y0}")
            #print(f"[Result] Test Top-1 Acc: {acc:.6f}")
            top1_list.append(acc)    

        elif mode == "intra":
            print(f"Iteration {i+1}, Run Intra-remapping")
            #Intra-remap inference
            model_remap_intra, remap_conv_metas_intra = inject_faults_conv_with_remapping_intra(
                model=model, msfp=msfp, counts_per_conv=counts_per_conv, ms_group=1, exp_group=64,
                seed=seed, device=device, verbose=verbose, mask=mask)
            #x0, y0 = testset[0]
            acc = eval_top1(model_remap_intra, testloader, device)
            #print(f"[Info] Testset size: {len(testset)} | classes: {len(testset.classes)}")
            #print(f"[Info] One sample shape: {tuple(x0.shape)} | label(example0): {y0}")
            #print(f"[Result] Test Top-1 Acc: {acc:.6f}")
            top1_list.append(acc)   

        elif mode == "msfp":
            """插入量化層到 Conv2d 前"""
            quant_layer = lambda: MSFPQuantizeLayer(msfp)
            model_quantized = build_resnet18_cifar10(num_classes=10)
            model_quantized.load_state_dict(ckpt["model"], strict=True)
            insert_quant_before_conv(model_quantized, quant_layer)
            model_quantized.to(device).eval()
            acc = eval_top1(model_quantized, testloader, device)
            top1_list.append(acc)

        else:
            #FP32 inference
            #print(f"Iteration {i+1}, No fault injection, run FP32 inference")
            #x0, y0 = testset[0]
            acc = eval_top1(model, testloader, device)
            #print(f"[Info] Testset size: {len(testset)} | classes: {len(testset.classes)}")
            #print(f"[Info] One sample shape: {tuple(x0.shape)} | label(example0): {y0}")
            #print(f"[Result] Test Top-1 Acc: {acc:.6f}")
            top1_list.append(acc)


    print("\nFinish !")
    print(f"\n=== Top-1 accuracy in BER {ber} 全部迭代結果 ===")
    for i in top1_list:
        print(i)

    #print("Top-1 List: ", top1_list)



if __name__ == "__main__":
    
    # main(mant_bits=7, box_size=16, iter=20, ber=3e-4, seed=None, 
    #       verbose=False, mode="intra", inter_mode="", mask=False)

    # main(mant_bits=7, box_size=16, iter=20, ber=4e-4, seed=None, 
    #       verbose=False, mode="intra", inter_mode="", mask=False)

    main(mant_bits=7, box_size=16, iter=1, ber=1e-6, seed=None, 
          verbose=False, mode="inter", inter_mode="rot", mask=False)

    # main(mant_bits=7, box_size=16, iter=20, ber=8e-5, seed=None, 
    #       verbose=False, mode="intra", inter_mode="", mask=False)

    # main(mant_bits=7, box_size=16, iter=20, ber=9e-5, seed=None, 
    #       verbose=False, mode="intra", inter_mode="", mask=False)

    # main(mant_bits=7, box_size=16, iter=20, ber=1e-4, seed=None, 
    #       verbose=False, mode="intra", inter_mode="", mask=False)

    # main(mant_bits=7, box_size=16, iter=20, ber=2e-4, seed=None, 
    #       verbose=False, mode="intra", inter_mode="", mask=False)

    # main(mant_bits=7, box_size=16, iter=20, ber=3e-4, seed=None, 
    #       verbose=False, mode="intra", inter_mode="", mask=False)

    # main(mant_bits=7, box_size=16, iter=20, ber=4e-4, seed=None, 
    #       verbose=False, mode="intra", inter_mode="", mask=False)
    
    # main(mant_bits=7, box_size=16, iter=20, ber=5e-4, seed=None, 
    #       verbose=False, mode="intra", inter_mode="", mask=False)

    # main(mant_bits=7, box_size=16, iter=20, ber=6e-4, seed=None, 
    #       verbose=False, mode="intra", inter_mode="", mask=False)
    
    # main(mant_bits=7, box_size=16, iter=20, ber=7e-4, seed=None, 
    #       verbose=False, mode="intra", inter_mode="", mask=False)

    # main(mant_bits=7, box_size=16, iter=20, ber=8e-4, seed=None, 
    #       verbose=False, mode="intra", inter_mode="", mask=False)

    # main(mant_bits=7, box_size=16, iter=20, ber=1e-4, seed=None, 
    #       verbose=False, mode="inter", inter_mode="rot", mask=False)
    
    # main(mant_bits=7, box_size=16, iter=20, ber=2e-4, seed=None, 
    #       verbose=False, mode="inter", inter_mode="rot", mask=False)