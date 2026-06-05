# 命名對照表 (Naming Conventions)

**Last updated:** 2026-06-01

本專案的兩個 Re=10 unsteady 訓練變體已重新定調。**唯一的實質差別是 L-BFGS 在哪裡執行**(優化器數學與 state 在 CPU/SciPy vs 全程常駐 GPU);模型、資料、Adam 階段、loss 定義都相同。

| 正式名稱 | 識別字 | 取代的舊名 | 既有結果資料夾 | 報告舊欄位 | 本質 |
|---|---|---|---|---|---|
| **reproduce** | `reproduce` | strict reproduce | `results/phase1_reproduction/strict_reproduce_scipy/` | "Strict" | SciPy L-BFGS-B,忠實重現 Rao 原始協定(每個 eval 有 host↔device 來回) |
| **GPU-resident** | `gpu_resident` | vanilla / vanilla_pytorch | `results/phase1_reproduction/vanilla_pytorch/` | "Vanilla" | `torch.optim.LBFGS`,整個優化迴圈與 state 常駐 GPU,無 per-eval CPU 搬運 |
| ~~vanilla~~ | — | — | — | — | **已退役**,不再使用(避免與舊 `vanilla_pytorch` 語意衝突) |

第三個對照基準(非本專案訓練產出):
- **Rao TF** — 原始 Rao 等人的 TensorFlow checkpoint(`ref/PINN-laminar-flow/PINN_unsteady/uvNN.pickle`),報告欄位同名 "Rao TF"。

## 注意

- **既有 `results/` 資料夾、程式碼、報告文字尚未實體改名** —— 目前一律以本表為準做語意對應。舊路徑 `strict_reproduce_scipy` 即 reproduce、`vanilla_pytorch` 即 GPU-resident。
- 重新命名只是定調 baseline 與加速版的稱呼,**不代表已驗證效能差異**。「GPU-resident 比較快」這類宣稱仍需固定 eval 數的控制實驗才能寫進報告。
- 受影響但尚未更新的引用點(待之後遷移):`report/sections/results_phase1.tex` 的 Vanilla/Strict 欄位、`report/matrix.tex` 的 "Vanilla (PyTorch)"、benchmark CSV 的 exp-name。
