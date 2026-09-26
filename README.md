# Intra-Bank and Inter-Bank Weight Remapping for Enhanced Fault Resilience in Block Floating-Point AI Accelerator Memory

## Overview
本專題針對以 Microsoft Floating Point (MSFP) 儲存 DNN 權重之 Flash Memory，提出權重重新映射韌性增強架構。透過 Fault Injection 分析不同 Bit Position 對模型準確率的影響，建立 Bit Significance 與 Error Score (ES)，再以 Inter-bank / Intra-bank Remapping 調整權重與故障 Memory Cell 的對應關係，並依 ES 選擇適合的 Remap Mode，以提升 DNN 權重儲存之故障容忍能力。

## PyTroch Simulation
主要由 Python 完成，使用 PyTroch 組件進行 DNN 模型訓練，包含完整模型訓練程式、錯誤注入模擬實驗、Error Score 產生器與重新映射模式對於準確率改善之驗證程式。詳細操作流程與指令皆紀錄於該資料夾之 README 文件。

## Verilog
主要由 Verilog 完成，包含完整硬體RTL、Testbench。詳細模組說明與層級介紹皆紀錄於該資料夾之 README 文件。

## Results
包含程式執行結果與實體設計之時序、面積、功耗紀錄，以及晶片實現結果與 Partition 表示。

## Appendix
收錄了不同DNN模型與訓練資料集之位元重要性分析結果與不同DNN模型之重新映射準確率改善結果。
