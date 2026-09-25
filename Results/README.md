# EFP Baseline Solver

這是一版可先跑通的 ICCAD 2026 Problem E baseline solver。重點是先產出格式完整的 `.cfg`，再逐步調整 floorplan / routing 來優化分數。

## Build

```bash
g++ -std=c++17 -O2 -pipe -static -s src/main.cpp -o efp_solver
```

若平台不允許 static link：

```bash
g++ -std=c++17 -O2 src/main.cpp -o efp_solver
```

## Run

```bash
./efp_solver case00.csv output.cfg
```

## 已實作

- CSV-like testcase parser
- EDGE / MACRO / SOFT shape 初始化
- EDGE block 貼邊 / corner placement
- Greedy placement for macro and soft blocks
- Vertical-strip channel generation
- Rectangle adjacency graph routing
- Shortest-path route through CHANNEL / SOFT feedthrough
- Feedthrough area one-shot expansion
- CFG writer
- Basic outline / overlap checker

## 後續建議優化

1. 加入 simulated annealing：move / swap / resize soft block。
2. 對 high-demand nets 做 path splitting。
3. 加入 channel usage estimator，避免 overflow。
4. Feedthrough area update 後做 legalization，而不是只 clamp。
5. Routing edge 選擇改用 guiding point / HPWL 精算。

## 注意

這份 baseline 的目標是提供可維護、可擴充的架構；官方 parser / evaluator 仍可能對格式細節非常嚴格，請用官方或自製 checker 先驗證。
