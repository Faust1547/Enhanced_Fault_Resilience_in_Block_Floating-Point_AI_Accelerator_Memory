# 軟體端（Python）

## 執行流程

1. Train Models
2. Run Fault Injection
3. Generate Error Score
4. Run Remapping Sweep
---

## 執行方式
### 1. Train Models
使用 CIFAR-10 資料集訓練 AlexNet、VGG16、GoogLeNet 與 ResNet18 模型，產生後續 Fault Injection 與 Remapping Sweep 所需的 .pth 模型權重。
```bash
python  "PyTorch Simulation/Train Models/VGG16_CIFAR10.py"
```
### 2. Run Fault Injection
使用 MSFP 格式進行模型權重轉換，並針對不同 Bit Position 與 BER 執行錯誤注入實驗，記錄模型準確率的變化，產生後續位元重要性分析所需的 Bit Sweep 資料。
```bash
 run_exponent_bit_sweep_to_csv(                # Exponent 或 Sign & Mantissa Bit 都可以使用
     ber_list = [1e-4, 1e-3, 1e-2, 1e-1],      # 要測試的 BER
     bit_points=[15],                          # 要測試的 Bit
     csv_name="VGG16_CIFAR10_bit_sweep.csv",   # 輸出檔案名稱
     mant_bits=7,                              # Mantissa Bit 長度
     box_size=16,                              # Bounding Box Size
     repeats=30,                               # 重複實驗次數，最終取平均值
     base_seed=0,                              
     data_root=DATA_ROOT,                      # 訓練集資料來源
     ckpt_path=CKPT_PATH,                      # 訓練完成權重資料來源
     batch_size=256,                           
     num_workers=0,
     )
```
執行檔案須包含 MSFP_Conveter.py。
```bash
python "PyTorch Simulation/Fault Injection/VGG16_single_fault_injection_bit_sweep.py" 
```

### 3. Generate Error Score
根據 Fault Injection 產生的 Bit Sweep 資料，計算各位元對模型準確率的影響，並透過分組與數值量化產生 Error Score，供後續重新映射評估使用。
```bash
python "PyTorch Simulation/Error Score Generator/Generate_ES_Candidates.py" \
  --"PyTorch Simulation/Error Score Generator/VGG16_CIFAR10_bit_sweep_EXP.csv" \ # 讀取 Bit Sweep 的資料路徑
  --baseline 92.74 \                                                             # 基準模型準確率
  --raw-method log-auc \                                                         # raw_importance 計算方式
  --es-min 2 \                                                                   # 最小 ES
  --es-max 32 \                                                                  # 最大 ES
  --group-levels "32,16,8,4,2" \                                                 # ES 區間
  --group-tolerance 0.07 \                                                       # ES 區間級距
  --output-prefix D:/Anaconda/PythonCode/Error_Score/VGG16_Exp.csv               # 輸出檔案路徑
```

### 4. Run Remapping Sweep
在不同 BER 條件下進行重新映射實驗，比較未重新映射、Inter-bank 與 Intra-bank 等模式對模型準確率的影響，評估各種重新映射方法的容錯效果。
```bash
python PyTorch Simulation/Remapping Sweep/VGG16_CIFAR10_remapping_sweep.py \
    --checkpoint "D:/Anaconda/PythonCode/data/vgg16_cifar10_ckpt_best.pth" \ # 讀取權重資料路徑
    --arch vgg16 \                                                           # 使用 VGG16 模型 (因為有綁定模型架構，故其他模型需另外改寫 Remapping Sweep 才可使用)
    --data-root "D:/Anaconda/PythonCode/data" \                              # 訓練集資料路徑
    --bers 1e-8,1e-7,1e-6,1e-5,5e-5,1e-4,5e-4,7e-4,9e-4,1e-3,3e-3,6e-3,8e-3,1e-2,2e-2,3e-2,4e-2,5e-2 \ # 測試的 BER
    --trials 20 \                                                            # 重複實驗次數，最終取平均值
    --workers 0 \                        
    --output-dir "D:/Anaconda/PythonCode/results/VGG16_remapping_result"     # 輸出檔案路徑
```

## Train Models

用於訓練基於 CIFAR-10 資料集的 AlexNet 、 VGG16 、 GoogLeNet 與 ResNet18，
並產生 Fault Injection 與 Remapping Sweep 所需的 `.pth` 權重檔。

- `AlexNet_CIFAR-10.py`
- `GoogLeNet_CIFAR10.py`
- `ResNet18_CIFAR10.py`
- `VGG16_CIFAR10.py`

## Fault Injection

包含 AlexNet 、 VGG16 、 GoogLeNet 與 ResNet18 四種模型的錯誤注入模擬程式，執行時須將 `MSFP_Converter.py` 一同執行

- `MSFP_Converter.py`
- `Alexnet_single_fault_injection_bit_sweep.py`
- `GoogLeNet_single_fault_injection_bit_sweep.py`
- `ResNet18_single_fault_injection_bit_sweep.py`
- `VGG16_single_fault_injection_bit_sweep.py`

## Error Score Generator

執行時須配合 Fault Injection 輸出的 Bit Sweep `.csv` 檔案。這邊提供 VGG16 模型 Bit Sweep 之資料。

- `Generate_ES_Candidates.py`
- `VGG16_CIFAR10_bit_sweep_EXP.csv`

## Remapping Sweep

`VGG16_CIFAR10_remapping_sweep.py` 為主要實驗重新映射效果之程式，執行時須將 `MSFP_Converter.py` 一同執行，這邊提供 VGG16 模型使用之程式。

- `MSFP_Converter.py`
- `VGG16_CIFAR10_remapping_sweep.py`
