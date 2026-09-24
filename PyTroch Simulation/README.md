# 軟體端（Python）－ 蒙地卡羅模擬實驗

## 執行流程

1. Train Models
2. Run Fault Injection
3. Generate Error Score
4. Run Remapping Sweep
---

## 執行方式
### 1. Train Models
以 CIFAR-10 訓練 AlexNet 、 VGG16 、 GoogLeNet 與 ResNet18，產生後續模擬所需的 `.pth` 模型權重。
```bash
python  "PyTorch Simulation/Train Models/VGG16_CIFAR10.py"
```
### 2. Run Fault Injection
執行錯誤注入模擬以進行位元重要性分析，實驗參數由程式最底下 run_exponent_bit_sweep_to_csv 區塊設定。
```bash
 run_exponent_bit_sweep_to_csv(                # Exponent 或 Sign & Mantissa Bit 都可以使用
     ber_list = [1e-4, 1e-3, 1e-2, 1e-1],      # 要測試的 BER
     bit_points=[15],                          # 要測試的 Bit
     csv_name="VGG16_CIFAR10_bit_sweep.csv",   # 產出檔案名稱
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
根據 Fault Injection 產生的 Bit Sweep 資料計算出各位元對應的 Error Score，並採用分組方式決定數值防止位元級距過大。
```bash
python PyTorch Simulation/Error Score Generator/Generate_ES_Candidates.py 
  PyTorch Simulation/Error Score Generator/VGG16_CIFAR10_bit_sweep_new_SM.csv # 讀取的 Bit Sweep 資料路徑
  --baseline 92.74                                                            # 基準模型準確率
  --raw-method log-auc                                                        # raw_importance 計算方式
  --es-min 2                                                                  # 最小 ES
  --es-max 32                                                                 # 最大 ES
  --group-levels "32,16,8,4,2"                                                # ES 區間
  --group-tolerance 0.07                                                      # ES 區間級距
  --output-prefix D:/Anaconda/PythonCode/Error_Score/AlexNet_E.csv            # 產出檔案路徑
```
### 4. Run Remapping Sweep
根據 Fault Injection 產生的 Bit Sweep 資料計算出各位元對應的 Error Score，並採用分組方式決定數值防止位元級距過大。
```bash
python PyTorch Simulation/Error Score Generator/Generate_ES_Candidates.py 
  PyTorch Simulation/Error Score Generator/VGG16_CIFAR10_bit_sweep_new_SM.csv # 讀取的 Bit Sweep 資料路徑
  --baseline 92.74                                                            # 基準模型準確率
  --raw-method log-auc                                                        # raw_importance 計算方式
  --es-min 2                                                                  # 最小 ES
  --es-max 32                                                                 # 最大 ES
  --group-levels "32,16,8,4,2"                                                # ES 區間
  --group-tolerance 0.07                                                      # ES 區間級距
  --output-prefix D:/Anaconda/PythonCode/Error_Score/AlexNet_E.csv            # 產出檔案路徑
```



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
