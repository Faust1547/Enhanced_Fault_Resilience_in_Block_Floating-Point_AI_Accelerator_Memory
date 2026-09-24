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
透過 Testbench 驗證完整系統流程正確運行且正確選擇重新映射模式，並且能夠正確將資料依據映射模式存入 SRAM 中，而後再送入 Flash meomory。

## VLSI Implement
| Specification | TSMC 90 nm 1P9M | 
|---|---|
| Frequency | 200 MHz | 
| Page Buffer Size | 16 KB | 
| Timing Closure | Setup / Hold Met |
| Dynamic Power | 29.9801 mW |
| Cell Leakage Power | 1.1363 mW |
| Core Area | 879,628.279 μm² |
| Chip Area | 1,371,395.020 μm² |
| LVS | Correct |
