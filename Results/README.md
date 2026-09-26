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
## Post-sim Result
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

## Hardware Overhead
### 整體 HO 計算
將 FICAM、Address Remappaer、Transposer 視作實現架構之外加硬體，而將 Flash Memory 及 Page Buffer 視作原有硬體，故可得到此 Hardware Overhead 表示式：

$$ HO = \frac{TC_{FICAM} + TC_{AR} + TC_{TRAN}}{TC_{FM} + TC_{PB}} \times 100\% $$

映射模式選擇控制單元非固定硬性之設計，而可以依設計架構與功能性要求進行彈性調整，故計算額外硬體消耗不考慮此模組。

### 不同輸入錯誤資訊數量之 HO
以128 GB之快閃記憶體為例，並帶入本專題實作硬體參數即可算出在不同數量 NE 輸入對應之 HO 結果如下：

| NE | HO(%) | 
|---|---|
| 16 |0.000278| 
|64 | 0.001109| 
|128| 0.002219 |
|512 | 0.008877 |
|1024|0.017754|

可見即便 NE 為 1024 筆故障資訊，其硬體帶來的額外開銷亦小於 1%，故本專題實現架構之 HO 幾乎可被忽略不計。

### 個別模組 Hardware Overhead 說明
根據個別模組之電晶體數目 (TC) 計算出整體Hardware Overhead。

* Flash Memory 包含  $$N_B$$ 個Block，其中每個 Block 包含 $$N_p$$  個 Page，而每個 Page 需要 M×N 顆電晶體，因此可表示為：

$$ TC_{FM}= N_B×N_p×M×N $$

* Page Buffer 使用 SRAM 構成，可將其單一 SRAM 單元定義為6顆電晶體，而 1 個 SRAM 需要 M×N 顆電晶體。因此可表示為：

$$ TC_{PB}=6×M×N$$

* Transposer 是由暫存器構成的矩陣，可將其單一暫存器單元定義為 16 顆電晶體，而單一矩陣長度為 Word Length (WL)，故需要 $$WL^2$$ 之矩陣大小暫存器。因此可表示為：

$$ TC_{TRAN}=16 × WL^2$$

* Address Remappaer 是由暫存器與XOR邏輯閘組成之模組，可將其單一暫存器單元定義為 16 顆電晶體、單一XOR邏輯閘定義為 6 顆電晶體，而輸入邏輯位址可表示為 Word Length (WL) 的 $log_2 $，故僅需要 $log_2 WL $長度之暫存器與 XOR 邏輯閘。因此可表示為：

$$ TC_{AR}= log_2 WL×(16+6)$$

* FICAM由多個CAM單元組成，可將其單一CAM單元定義為9顆電晶體，單一CAM單元可保存一組故障資訊，一共輸入 NE 組故障資訊，其中包含1 bit有效位元、PPN位址之長度為 $log_2 (N_B×N_P )$、Bank位址之長度為 $$log_2、N_{Bank}、RG$$ 位址之長度為 $log_2 N_{RG}$、故障資訊包含故障位元與故障類型，總長度為 $log_2 WL×N_{TU}$、重新映射模式之長度為 $log_2 N_{RM}$、重新映射資訊之長度為 $log_2 N_{RV}$。因此可表示為：

$$ TC_{FICAM} = NE \times 9 \times [1 + \log_2(N_B \times N_P) + \log_2 N_{Bank} + \log_2 N_{RG} + \log_2 WL \times N_{TU} + \log_2 N_{RM} + \log_2 N_{RV}] $$
