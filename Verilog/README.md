# 硬體端（Verilog）

## Top Module

`TOP.v`

## Testbench

`TOP_tb.v`


## Module Hierarchy

```text
TOP.v
├── Address_Remapper.v
├── Data_Shifter.v
├── FICAM.v
├── Control_Unit.v
│   └── ES_Score_Engine.v
├── Transposer.v
└── SRAM_SP_ADV_rtl.v
    └── SRAM_SP_ADV.v
```
| Module | Description |
|---|---|
| `TOP.v` | 最上層模組，負責整個系統的模組溝通與資料交互 |
| `Control_Unit.v` | 負責控制系統運作狀態與評估最佳 Error Score 之重新映射模式選擇 |
| `ES_Score_Engine.v` | 透過平行運算輔助 Control_Unit 加速計算 Error Score |
| `FICAM.v` | 負責存放故障資訊與重新映射資訊，包括故障位址、故障模型、映射方式與映射參數 |
| `Address_Remapper.v` | 透過 XOR 運算實現 Intra-bank remapping 之層內位址交換 |
| `Data_Shifter.v` | 透過 Barrel shifter 實現 Inter-bank remapping 之層間位址交換  |
| `Transposer.v` | 在進行 Inter-bank remapping 時負責將資料進行矩陣轉置 |
| `SRAM_SP_ADV_rtl.v` | RTL wrapper for SRAM |
| `SRAM_SP_ADV.v` | SRAM IP |
