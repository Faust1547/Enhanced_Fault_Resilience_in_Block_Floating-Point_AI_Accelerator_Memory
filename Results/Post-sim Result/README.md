# Post-sim Result Screenshot
<img width="1272" height="285" alt="Post-sim_RTL_Result" src="https://github.com/user-attachments/assets/3eea3c62-15e2-4b6c-b422-616a17d89d06" />

# Simulation Log
``` text
[PASS] synthesized 32-entry TOP: LOAD -> Eval_Start -> page mode = INTER
[PASS] normal write handshake completed (SRAM internal check disabled)

Simulation complete via $finish(1) at time 246 NS + 0
./TOP_syn_tb.v:217 $finish;
```
# 層內權重重新映射 (Intra-bank Remapping) 資料處理之電路模擬
Intra-bank Remapping 之 輸入資料與 SRAM 儲存資料對照表

|DataIndex|Control Word|	Initial Data	|Logical Address|	Shifted Data	|Physical Address|
|---| ---|	---	|---|	---	|---|
|0	|3|	0x81	|0x0400	|0xA1|	0x0403|
|1	|3	|0x42|	0x0401|	0x62|	0x0402|
|2	|3	|0x24|	0x0402|	0xA4	|0x0401|
|3	|3	|0x18	|0x0403	|0x68|	0x0400|
|4	|3	|0xF0|	0x0404|	0x98	|0x0407|
|5	|3|	0x0F|0x0405|	0x54	|0x0406|
|6	|3|	0xAA	|0x0406	|0x92|	0x0405|
|7|	3	|0x55	|0x0407	|0x51|	0x0404|

### 外部資料與位址輸入
<img width="844" height="61" alt="image" src="https://github.com/user-attachments/assets/b38b7b64-dff8-4530-bbad-be9d4d365a74" />

### Control Word數值
<img width="843" height="30" alt="image" src="https://github.com/user-attachments/assets/1242e644-a552-41bb-9132-1c3119935c6a" />

### SRAM 存放資料結果
<img width="844" height="99" alt="image" src="https://github.com/user-attachments/assets/38c7ea7f-a57d-41d2-9c1d-a6d61505f20f" />



# 層間權重重新映射 (Inter-bank Remapping) 資料處理之電路模擬
Inter-bank Remapping 之 輸入資料與 SRAM 儲存資料對照表

|DataIndex|Shift Mode|	Initial Data	|Logical Address|	Shifted Data	|Physical Address|
|---| ---|	---	|---|	---	|---|
|0	|3|	0xD2	|0x0300	|0x5A|	0x0300
|1	|3	|0x81|	0x0301|	0x30|	0x0301
|2	|3	|0x3C|	0x0302|	0x87	|0x0302
|3	|3	|0xA5	|0x0303	|0xB4|	0x0303
|4	|3	|0x01|	0x0304|	0x20	|0x0304
|5	|3|	0x80|0x0305|	0x10	|0x0305|
|6	|3|	0xF0	|0x0306	|0x1E|	0x0306
|7|	3	|0x5A	|0x0307	|0x4B|	0x0307

### 外部資料與位址輸入
<img width="844" height="73" alt="image" src="https://github.com/user-attachments/assets/57355362-6d97-4506-9769-b33e9d3f0131" />

### Shift Mode 數值
<img width="528" height="37" alt="image" src="https://github.com/user-attachments/assets/bb541a5b-496d-4237-b475-b97f6ccd9310" />

### SRAM 存放資料結果
<img width="844" height="90" alt="image" src="https://github.com/user-attachments/assets/74a5ac70-3166-4393-ad0e-31042c1130dc" />

