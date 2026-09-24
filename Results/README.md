## Hierarchy
``` text
Results
├── README.md
├── RTL Simulation Result
│   └── Post-sim_RTL_Result.png
└── VLSI Implement
    ├── Area
    ├── Chip
    ├── LVS
    ├── Power
    └── Timing
```
## RTL Simulation
透過 Testbench 驗證 AES-128 加密與解密功能，比對預期結果與實際輸出，並展示六組測試案例之 Post-simulation 通過結果。

## VLSI Implement
| Specification | TSMC 90 nm 1P9M | TSMC N16 ADFP |
|---|---|---|
| Frequency | 200 MHz | 1.25 GHz |
| Timing Closure | Setup / Hold Met | Setup / Hold Met|
| Dynamic Power | 17.7559 mW | 10.6 mW |
| Core Area | 557,343.647 μm² | 10,777.54 μm² |
| Chip Area | 1,044,749.730 μm² | 21,025.00 μm² |
| LVS | Correct | Correct |
