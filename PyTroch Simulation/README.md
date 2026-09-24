# 軟體端（Python）－ 蒙地卡羅模擬實驗

## 執行流程

1. Train Models
2. Run Monte Carlo Simulation
3. Generate Constraint Data

---

## 執行方式
### 1. Train Models
以 CIFAR-10 訓練 AlexNet 或 VGG16，產生後續模擬所需的 `.pth` 模型權重。
```bash
python  "PyTorch Simulation/Train Models/VGG16_CIFAR-10.py"
```
### 2. Monte Carlo Simulation
執行蒙地卡羅錯誤注入與 Remapping 模擬，實驗參數由對應的 `.json` 設定檔指定。
```bash
python "PyTorch Simulation/Monte Carlo Simulation/VGG16/vgg16_global_mode_accuracy_experiment.py" \
  --config "PyTorch Simulation/Monte Carlo Simulation/VGG16/vgg16_global_mode_accuracy_config.json"
```
### 3. Generate Constraint Data
根據 Monte Carlo Simulation 產生的 BER constraint regions，進一步轉換為 RTL 端可使用的 Fault-Count Thresholds。
```bash
python "PyTorch Simulation/Generate Constraint Data/generate_constraint_dat.py" \
  --storage-audit "PyTorch Simulation/Generate Constraint Data/VGG16/storage_audit.json" \
  --regions-csv "PyTorch Simulation/Generate Constraint Data/VGG16/ber_constraint_regions.csv" \
  --target-retention 0.98 \ # 目標準確率
  --selection-policy selected_pooled_wilson_monotonic \
  --storage-bit-source valid \
  --rounding floor \
  --width 24 \
  --output-dat VGG16_cifar10_r980_constraint.dat \
  --output-metadata VGG16_cifar10_r980_constraint_metadata.json
```
---
## Train Models

用於訓練基於 CIFAR-10 資料集的 AlexNet 與 VGG16 模型，
並產生 Monte Carlo Simulation 所需的 `.pth` 權重檔。

- `AlexNet_CIFAR-10.py`
- `VGG16_CIFAR-10.py`

## Monte Carlo Simulation

`.py` 為主要模擬程式，`.json` 為實驗參數設定檔。

### AlexNet
- `alexnet_global_mode_accuracy_experiment.py`
- `alexnet_global_mode_accuracy_config.json`

### VGG16
- `vgg16_global_mode_accuracy_experiment.py`
- `vgg16_global_mode_accuracy_config.json`

## Generate Constraint Data

根據模擬結果整理 BER Constraint Regions，
並轉換為 RTL 可使用的 Fault-Count Thresholds。

- `generate_constraint_dat.py`
### AlexNet
- `ber_constraint_regions.csv`
- `storage_audit.json`

### VGG16
- `ber_constraint_regions.csv`
- `storage_audit.json`

## 實驗參數
<img width="750" height="450" alt="image" src="https://github.com/user-attachments/assets/e806a47c-1061-4432-a2a0-2cb54ffef383" />
<img width="750" height="300" alt="image" src="https://github.com/user-attachments/assets/b81cc70f-2607-44ce-87bc-f6a970238cb1" />
