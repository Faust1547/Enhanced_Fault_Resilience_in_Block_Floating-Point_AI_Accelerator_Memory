# 硬體端（Verilog）

## Top Module

`TOP.v`

## Testbench

`tb_TOP.v`


## Module Hierarchy

```text
TOP.v
├── Address_Remapper.v
├── Data_Shifter.v
├── FICAM.v
├── MC_Control_Unit_PBG.v
├── Remap_Info_Memory.v
├── Transposer.v
└── SRAM_SP_ADV_Command_Pipeline.v
    └── SRAM_SP_ADV_rtl.v
        └── SRAM_SP_ADV.v
```
| Module | Description |
|---|---|
| `TOP.v` | 最上層模組，負責整個系統的模組溝通與資料交互 |
| `MC_Control_Unit_PBG.v` | 負責控制系統運作狀態與根據 Fault-count 評估使用的重新映射方式 |
| `FICAM.v` | 負責存放故障資訊，包括故障位址與故障模型 |
| `Address_Remapper.v` | 透過 XOR 運算實現 Intra-bank remapping 之層內位址交換 |
| `Data_Shifter.v` | 透過 Barrel shifter 實現 Inter-bank remapping 之層間位址交換  |
| `Transposer.v` | 在進行 Inter-bank remapping 時負責將資料進行矩陣轉置 |
| `Remap_Info_Memory.v` | 負責存放重新映射資訊，包括映射方式與映射參數 |
| `SRAM_SP_ADV_Command_Pipeline.v` | SRAM 控制器 |
| `SRAM_SP_ADV_rtl.v` | RTL wrapper for SRAM |
| `SRAM_SP_ADV.v` | SRAM IP |
