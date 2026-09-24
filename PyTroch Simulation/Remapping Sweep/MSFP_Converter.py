# MSFP Converter
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import struct
import os
import random
#####################################################################################################################
class MSFPConverter:
    def __init__(self, mantissa_bits=3, box_size=16, verbose=False):
        """
        MSFPConverter 初始化
        """
        self.mantissa_bits = mantissa_bits
        self.box_size = box_size 
        self.verbose = verbose
#####################################################################################################################
    def torch_frexp(self, x):
        x_abs = x.abs()
        exponent = torch.floor(torch.log2(x_abs))
        mantissa = x_abs / (2.0 ** exponent)
        #exponent = torch.where(x == 0, torch.tensor(0, dtype=x.dtype, device=x.device), exponent)
        #mantissa = torch.where(x == 0, torch.tensor(0.0, device=x.device), mantissa)
        exponent[x == 0] = 0
        mantissa[x == 0] = 0
    
        return mantissa, exponent.to(dtype=torch.int8)
#####################################################################################################################
    def float_to_msfp(self, data):
        """
        將浮點數 array 轉換成 MSFP 表示法
        回傳: signs, shared_exponents, quantized_mantissas
        """
        torch_frexp = self.torch_frexp

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        else:
            data = data.to(dtype=torch.float32)

        if self.verbose:
            print("Original Weight: ", data)

        #data = np.array(data, dtype=np.float32) #轉換成np array
        #n = data.size
        #signs = np.where(data >= 0, 0, 1)
        #mantissas, exps = np.frexp(np.abs(data))
        n = data.numel()
        device = data.device
        signs = (data < 0).to(dtype=torch.int8, device=device)
        mantissas, exps = torch_frexp(data.abs())

        # === 分 box ===
        num_boxes = (n + self.box_size - 1) // self.box_size
        pad_len = num_boxes * self.box_size - n

        if pad_len > 0:
            mantissas = torch.cat([mantissas, torch.zeros(pad_len, dtype=mantissas.dtype, device=device)])
            exps = torch.cat([exps, torch.zeros(pad_len, dtype=exps.dtype, device=device)])

        mantissas = mantissas.view(num_boxes, self.box_size)
        exps = exps.view(num_boxes, self.box_size)
        # === 計算 shared exponent 與量化 mantissa ===
        e_shared = exps.max(dim=1, keepdim=True).values                  # [B, 1]
        y = mantissas / (2.0 ** (e_shared - exps))                       # [B, S]
        m = torch.round(y * (2.0 ** (self.mantissa_bits - 1.0))).to(torch.int32)
        m = torch.clamp(m, 0, 2 ** self.mantissa_bits - 1)
        # shared_exps = torch.zeros(n, dtype=torch.int32, device=device)
        # quant_mants = torch.zeros(n, dtype=torch.int32, device=device)
        # for start in range(0, n, self.box_size):
        #     end = min(start + self.box_size, n)
        #     box_indices = slice(start, end)

        #     box_exps = exps[box_indices]
        #     box_manti = mantissas[box_indices]

        #     if self.verbose:
        #         print("Original Weight Box signs: ", signs)
        #         print("Original Weight Box exponents: ", box_exps)
        #         print("Original Weight Box mantissas: ", box_manti)

        #     e_shared = box_exps.max()
        #     y = box_manti / (2.0 ** (e_shared - box_exps))
        #     m = torch.round(y * (2.0 ** (self.mantissa_bits - 1.0))).to(torch.int32)
        #     m = torch.clamp(m, 0, 2 ** self.mantissa_bits - 1)

        #     shared_exps[box_indices] = e_shared
        #     quant_mants[box_indices] = m
        # === verbose 模式顯示每個 box 的詳細資料 ===
        if self.verbose:
            for i in range(num_boxes):
                start = i * self.box_size
                end = start + self.box_size
                print(f"[Box {i}] signs     :", signs[start:end])
                print(f"[Box {i}] exponents :", exps[i])
                print(f"[Box {i}] mantissas :", mantissas[i])
                print(f"[Box {i}] e_shared  :", e_shared[i].item())
                print(f"[Box {i}] quantized :", m[i])
        
        # 還原成一維，切回原本長度
        shared_exps = e_shared.expand(-1, self.box_size).reshape(-1)[:n]
        quant_mants = m.reshape(-1)[:n]

        return signs, shared_exps.to(torch.int8), quant_mants
#####################################################################################################################
    def msfp_to_float(self, signs, exponents, mantissas):
        #將 MSFP 轉回浮點數
        mantissas = mantissas.float()
        exponents = exponents.float()
        f_result = (1 - 2 * signs) * (mantissas / (2.0 ** (self.mantissa_bits - 1))) * (2.0 ** exponents)
        return f_result
#####################################################################################################################
    def flip_bit_using_xor(self, signs, exponents, mantissas, fault_num, index, seed=None):
        """
        進行 bit-flip fault injection
        """
        #total_w = len(signs)
        total_w = signs.size(0)
        #total_faults = int(total_w * prob)
        total_exp = torch.div(exponents.size(0), self.box_size, rounding_mode='floor')
        print(total_exp)
        total_faults = fault_num
        # print(f"Total faults: {total_faults}")
        
        if seed is not None:
            torch.manual_seed(seed)

        #fault_position = random.sample(range(total_w), total_faults)
        fault_position = torch.randperm(total_w, device=signs.device)[:total_faults]
        fault_position_exp = torch.randperm(total_exp, device=signs.device)[:total_faults]
        # print("Fault position in: ", fault_position)

        # if self.verbose:
        #     print(f"Total faults: {total_faults}")
        #     print(f"Fault positions: {fault_position.tolist()}")

        if index < self.mantissa_bits:
            mantissas[fault_position] ^= (1 << index)
            print(f"Total faults: {total_faults}")
            print("Fault position in: ", fault_position)
            print(f"Total affected elements: {fault_position.numel()}")

        elif index < self.mantissa_bits + 8:
            shift = index - self.mantissa_bits
            box_size = self.box_size
            #total_tiles = (total_w + box_size - 1) // box_size

            # 找出哪些 tile 被選中
            #tile_ids = torch.div(fault_position, box_size, rounding_mode='floor').unique()

            tile_ids = fault_position_exp

            # 建立 mask，選出所有受影響的 index
            affected_mask = torch.zeros(total_w, dtype=torch.bool, device=exponents.device)
            for tile_id in tile_ids:
                start = tile_id * box_size
                end = min(start + box_size, total_w)
                affected_mask[start:end] = True

            # 進行 fault injection（僅翻轉那些被 mask 選中的位置）
            exponents_to_flip = exponents[affected_mask].to(torch.uint8)
            flipped = torch.bitwise_xor(exponents_to_flip, 1 << shift)
            exponents = exponents.clone()
            exponents[affected_mask] = flipped.to(torch.int8)
            e_fault = affected_mask.sum().item() // box_size

            print(f"Total affected exponents: {e_fault}")
            print(f"Total affected elements: {affected_mask.sum().item()}")
            if self.verbose:
                print(f"Tile IDs affected: {tile_ids.tolist()}")
                print(f"Total affected exponents: {e_fault}")
                print(f"Total affected elements: {affected_mask.sum().item()}")
            # exponents_to_flip = exponents[fault_position].to(torch.uint8)
            # exponents_flipped = torch.bitwise_xor(exponents_to_flip, 1 << shift)
            # exponents = exponents.clone()
            # exponents[fault_position] = exponents_flipped.to(torch.int8)

        else:
            signs[fault_position] ^= 1
            print(f"Total faults: {total_faults}")
            print("Fault position in: ", fault_position)
            print(f"Total affected elements: {fault_position.numel()}")
        # for i in fault_position:
        #     i = i.item()
        #     if index <= (self.mantissa_bits - 1):
        #         mantissas[i] ^= (1 << index)
        #     elif index <= (self.mantissa_bits + 7):
        #         shift = index - self.mantissa_bits
        #         #exponents[i] = np.uint8(exponents[i]) ^ (1 << (index - self.mantissa_bits))
        #         exponents[i] = torch.bitwise_xor(exponents[i].to(torch.uint8), 1 << shift)
        #     else:
        #         signs[i] ^= 1
        return signs, exponents, mantissas
#####################################################################################################################
    def flatten_tensor(self, data):
        """
        tile-based channel flatten (支援 1D, 2D, 3D, 4D tensor)
        """
        dim = len(data.shape)
        ori_shape = data.shape

        if dim == 1:
            return data, ori_shape
        elif dim == 2:
            #flat_data = data.numpy().flatten()
            #data = data.permute(1, 0)
            flat_data = data.flatten()
            return flat_data, ori_shape
        
        elif dim == 3:
            C, H, W = data.shape
            data_t = data.permute(1, 2, 0).reshape(H * W, C)
            flat_tiles = [data_t[:, i:i + self.box_size].flatten()
                          for i in range(0, C, self.box_size)]
            # for start_c in range(0, C, self.box_size):
            #     end_c = min(start_c + self.box_size, C)
            #     tile = data_reshaped[:, start_c:end_c]
            #     flat_tiles.append(tile.flatten())
            return torch.cat(flat_tiles).to(data.device), ori_shape
            # for h in range(H):
            #     for w in range(W):
            #         for start_c in range(0, C, self.box_size):
            #             end_c = min(start_c + self.box_size, C)
            #             tile_vals = data[start_c:end_c, h, w]
            #             # flat_result.extend(tile_vals.tolist())
            #             flat_result.append(tile_vals.flatten())
        elif dim == 4:
            N, C, H, W = data.shape
            data_t = data.permute(0, 2, 3, 1).reshape(N * H * W, C)
            flat_tiles = [data_t[:,i:i + self.box_size].flatten()
                          for i in range(0, C, self.box_size)]
            # for start_c in range(0, C, self.box_size):
            #     end_c = min(start_c + self.box_size, C)
            #     tile = data_reshaped[:, start_c:end_c]
            #     flat_tiles.append(tile.flatten())
            return torch.cat(flat_tiles).to(data.device), ori_shape
        
        else:
            raise ValueError(f"Unsupported tensor dimension: {dim}")
            # for n in range(N):
            #     for h in range(H):
            #         for w in range(W):
            #             for start_c in range(0, C, self.box_size):
            #                 end_c = min(start_c + self.box_size, C)
            #                 tile_vals = data[n, start_c:end_c, h, w]
            #                 #flat_result.extend(tile_vals.tolist())
            #                 flat_result.append(tile_vals.flatten())

#####################################################################################################################
    def unflatten_tensor(self, flat_array, shape):
        """
        tile-based flatten 反向還原
        """
        # print("shape type:", type(shape))
        # print("shape content:", shape)
        # print("shape length:", len(shape))

        dim = len(shape)

        if len(shape) == 1:
            #flat_array = torch.tensor(flat_array, dtype=torch.float32)
            return flat_array
        
        elif len(shape) == 2:
            H, W = shape
            #flat_array = flat_array.view(W,H).T
            flat_array = flat_array.reshape(shape)
            return flat_array
        
        elif len(shape) == 3:
            C, H, W = shape
            num_pixels = H * W
            box = self.box_size
            num_tiles = (C + box - 1) // box  # 向上取整
            # 每 tile 長度為 pixel 數量乘以 tile 大小（或剩餘 channel）
            tile_sizes = [min(box, C - i) for i in range(0, C, box)]
            split_sizes = [num_pixels * size for size in tile_sizes]
            # 將 flat_array 分成 num_tiles 個 tensor，每個 shape 為 (num_pixels, box_size)
            tiles = torch.split(flat_array, split_sizes, dim=0)
            tile_tensors = [tile.view(num_pixels, -1) for tile in tiles]

            # 把所有 tile 在 channel 方向上合併 => (num_pixels, C)
            data_t = torch.cat(tile_tensors, dim=1)

            # 還原成 (H, W, C) -> (C, H, W)
            data = data_t.view(H, W, C).permute(2, 0, 1)
            return data
            # output = torch.zeros(C, H, W, dtype=flat_array.dtype, device=flat_array.device)
            # for start_c in range(0, C, box):
            #     end_c = min(start_c + box, C)
            #     size = (end_c - start_c) * H * W
            #     output[start_c:end_c] = flat_array[idx:idx+size].view(end_c - start_c, H, W)
            #     idx += size
            # return output
            # for h in range(H):
            #     for w in range(W):
            #         for start_c in range(0, C, self.box_size):
            #             end_c = min(start_c + self.box_size, C)
            #             size = end_c - start_c
            #             output[start_c:end_c, h, w] = flat_array[idx:idx+size]
            #             idx += size
        elif len(shape) == 4:
            N, C, H, W = shape
            num_pixels = N * H * W
            box_size = self.box_size
            num_tiles = (C + box_size - 1) // box_size  # 向上取整

            # 每 tile 長度為 pixel 數量乘以 tile 大小（或剩餘 channel）
            tile_sizes = [min(box_size, C - i) for i in range(0, C, box_size)]
            split_sizes = [num_pixels * size for size in tile_sizes]

            # 將 flat_array 分段還原
            tiles = torch.split(flat_array, split_sizes, dim=0)
            tile_tensors = [tile.view(num_pixels, -1) for tile in tiles]

            # 合併所有 tile 為完整 channel 資訊
            data_t = torch.cat(tile_tensors, dim=1)  # shape: (N*H*W, C)

            # 還原 shape: (N, H, W, C) → permute → (N, C, H, W)
            data = data_t.view(N, H, W, C).permute(0, 3, 1, 2)

            return data
            # output = torch.zeros(N, C, H, W, dtype=flat_array.dtype, device=flat_array.device)
            # for start_c in range(0, C, box):
            #     end_c = min(start_c + box, C)
            #     size = N * (end_c - start_c) * H * W
            #     output[:, start_c:end_c] = flat_array[idx:idx+size].view(N, end_c - start_c, H, W)
            #     idx += size
            # return output
        
        else:
            raise ValueError(f"Unsupported dimension: {dim}")
            # output = np.zeros((N, C, H, W), dtype=flat_array.dtype)
            # output = torch.zeros((N, C, H, W), dtype=flat_array.dtype)
            # for n in range(N):
            #     for h in range(H):
            #         for w in range(W):
            #             for start_c in range(0, C, self.box_size):
            #                 end_c = min(start_c + self.box_size, C)
            #                 size = end_c - start_c
            #                 output[n, start_c:end_c, h, w] = flat_array[idx:idx+size]
            #                 idx += size
            #return output
#####################################################################################################################
    def quantize(self, data):
        flatten_tensor = self.flatten_tensor
        float_to_msfp = self.float_to_msfp
        msfp_to_float = self.msfp_to_float
        unflatten_tensor = self.unflatten_tensor
        data = data.to(torch.float32)
        device = data.device
        flatten_data, data_shape = flatten_tensor(data)
        sign, shared_exp, mantissas = float_to_msfp(flatten_data)
        reconstructed = msfp_to_float(sign, shared_exp, mantissas)
        quantized_data = unflatten_tensor(reconstructed, data_shape)
        
        return quantized_data.to(device)
#####################################################################################################################
    def quantize_with_fault_injection(self, data, index, fault_num, seed):
        #flip_bit_using_xor(self, signs, exponents, mantissas, index, prob, seed=None):
        flatten_tensor = self.flatten_tensor
        float_to_msfp = self.float_to_msfp
        flip_bit_using_xor = self.flip_bit_using_xor
        msfp_to_float = self.msfp_to_float
        unflatten_tensor = self.unflatten_tensor
        data = data.to(torch.float32)
        device = data.device

        flatten_data, data_shape = flatten_tensor(data)
        sign, shared_exp, mantissas = float_to_msfp(flatten_data)
        sign, shared_exp, mantissas = flip_bit_using_xor(signs=sign, exponents=shared_exp, mantissas=mantissas, fault_num=fault_num, index=index, seed=seed)
        reconstructed = msfp_to_float(sign, shared_exp, mantissas)
        quantized_data = unflatten_tensor(reconstructed, data_shape)
        
        return quantized_data.to(device)
    

msfp = MSFPConverter(mantissa_bits=3, box_size=16, verbose=False)