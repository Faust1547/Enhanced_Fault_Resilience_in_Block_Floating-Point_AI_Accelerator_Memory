## Hierarchy
``` text
Results
├── README.md
├── Post-sim Result
│   └── README.md
└── VLSI Implement
    ├── Area
    ├── Chip
    ├── LVS
    ├── Power
    └── Timing
```
## RTL Simulation
透過 Testbench 驗證系統控制流程及重新映射模式選擇功能，並測試映射後資料的 SRAM 寫入控制與後續資料傳輸流程。模擬結果顯示，相關測試案例均通過預期檢查。

## Physical Implementation
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
