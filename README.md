# Monte Carlo-Based Memory Remapping Selection

## Overview
基於蒙特卡羅方法的容錯深度神經網路權重記憶重映射模式選擇。探討是否能透過大量 Monte Carlo Simulation 分析不同 Bit Error Rate (BER) 與 Fault Injection 條件下，各種 Remapping Modes 的適用情況，再根據統計結果建立 Remapping Selection Rule，使系統能依據觀測到的 Fault Count 選擇適合的映射模式，並維持預先設定的 DNN Inference Accuracy Target。

## Software Flow
主要由 Python 完成，使用 PyTroch 組件進行 DNN 模型訓練，包含完整模型訓練程式、蒙地卡羅模擬實驗程式與參數、Fault-count Threshold 產生程式與參數。詳細操作流程與指令皆紀錄於該資料夾之 README 文件。

## RTL Architecture
主要由 Verilog 完成，包含完整硬體RTL、Testbench。詳細模組說明與層級介紹皆紀錄於該資料夾之 README 文件。

## Results
包含程式執行結果與實體設計之時序、面積、功耗紀錄，以及晶片實現結果與 Partition 表示。
