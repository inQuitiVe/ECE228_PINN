# 現狀陳述與階段檢討：Re=10 復現與 loss-balancing 嘗試

**日期：** 2026-05-28  
**專案：** ECE228 PINN unsteady cylinder flow  
**範圍：** Re=10 baseline reproduction、Wang 2020 gradient-norm loss balancing、後續轉向 inverse / collocation

---

## 1. 簡短結論

目前專案已經完成 Re=10 unsteady cylinder flow 的 baseline run，但若以「嚴格復現論文」作為標準，這一階段應該誠實地判定為**復現不完全成功**。最好的 run 是 `strict_reproduce_5090`，速度場 L2_u 尚可，但 L2_v 偏高，pressure 數值與 pressure probe 曲線都明顯不理想：

| Run | L2_u | L2_v | L2_p | Verdict |
|---|---:|---:|---:|---|
| Phase 1 `strict_reproduce_5090` | 5.30% | 19.86% | 10.79% | best baseline |
| Phase 1 vanilla / scaffold fixed-weight runs | 約 6% | 約 20-23% | - | 可作為穩定 baseline |

後續嘗試用 gradient-norm adaptive loss weighting 改善 PINN 訓練，但結果明顯失敗：

| Run | Method | L2_u | L2_v | L2_p | Verdict |
|---|---|---:|---:|---:|---|
| C `grad_norm_5090` | buggy L2-norm grad ratio | 21.72% | 57.86% | 29.20% | 約 3 倍更差 |
| R1 `strict_grad_norm` clipped | Wang formula + clip max 1e3 | 99.83% | 94.48% | 100.16% | collapse |
| R3 `strict_grad_norm` uncapped | Wang formula, no practical clip | 約 100% | 約 100% | 約 100% | trivial-solution collapse |

因此目前判斷是：

> 我們目前只有一個可用 baseline，還不能宣稱完整復現 Rao 2020。速度場 qualitative behavior 接近，但 pressure probe 曲線和 L2_p 都不夠好。後續 loss-balancing 嘗試也沒有改善，反而把模型推向錯誤的 constraint trade-off 或 trivial solution。這不是傳統資料 overfitting，而是 PINN 加權約束目標的 degeneracy / constraint imbalance。

---

## 2. Baseline 復現目前狀態

Phase 1 的目標是復現 Rao / Sun / Liu 2020 的 unsteady mixed-form PINN。這部分目前只能稱為 baseline reproduction attempt，而不是完整成功復現。

速度場方面，`strict_reproduce_5090` 的 L2_u 接近原始 TensorFlow checkpoint，L2_v 也落在同一個量級，因此可以暫時作為後續方法比較的 baseline。不過，pressure 相關結果明顯不足：

- L2_p = 10.79%，仍然偏高。
- pressure probe 曲線和論文圖中的時間訊號差距大。
- probe pressure 的 phase / amplitude / waveform 都沒有達到可信復現。
- 因此不能只靠 L2_u 接近就宣稱復現成功。

目前最可信 baseline：

- `strict_reproduce_5090`
- SciPy L-BFGS-B
- TF-style initialization
- Adam 5k + 約 68k L-BFGS evaluations
- L2_u = 5.30%
- L2_v = 19.86%
- L2_p = 10.79%

重要限制與失敗點：

- 原 paper 對 unsteady case 主要給 qualitative plots，沒有明確報告全時域 L2。
- 我們的 reference 是 Chorin / IBM solver，不是 Fluent；pressure 尤其容易有數值方法差異。
- 即使考慮 reference solver 差異，pressure probe 曲線仍然太差，不能視為成功復現。
- 因此 Phase 1 應該被定義為「建立可比較 baseline」，而不是宣稱完全重現 paper 的所有物理量。

### 2.1 嚴格復現判定

若採用比較寬鬆的標準：

```text
速度場大致合理，L2_u 接近 TF checkpoint，可以作為 baseline。
```

則 Phase 1 可視為 baseline 已完成。

但若採用嚴格復現標準：

```text
速度場、pressure field、pressure probe time series 都要和論文一致。
```

則目前應判定為復現失敗或至少是不完全復現。尤其 pressure probe 曲線是論文 unsteady case 的重要 evidence，目前我們沒有復現出可信曲線。這點應該在報告中明確揭露。

---

## 3. Loss-balancing 實驗為什麼失敗

### 3.1 原始 `grad_norm` 版本的問題

最早的 `GradNormBalancer` 實作並不是 Wang 2020 Algorithm 1 的公式。

實作使用的是類似：

```text
lambda_hat_i = max_j ||grad L_j||_2 / ||grad L_i||_2
```

但 Wang 2020 的公式是：

```text
lambda_hat_i = max_theta(|grad L_r|) / mean_theta(|grad L_i|)
```

差異很大：

- Wang 的 numerator 是 PDE residual loss `L_r` 的最大梯度元素，不是所有 loss term 裡最大的 L2 norm。
- Wang 的 denominator 是參數梯度絕對值的 mean，不是 L2 norm。
- mean(|grad|) 對 gradient sparsity 很敏感；這是論文演算法的核心行為之一。

所以 C run (`grad_norm_5090`) 的失敗主要是實作公式和論文不一致。它讓某些 boundary / inlet 權重沒有被正確放大，結果 field L2 顯著變差。

### 3.2 `strict_grad_norm` clipped 版本的問題

後來修正為接近 Wang 2020 的公式：

```text
lambda_hat_i = max_theta(|grad L_f|) / mean_theta(|grad L_i|)
lambda_i = (1 - alpha) lambda_i + alpha lambda_hat_i
```

並使用 `alpha = 0.9`。這比較接近原論文。

但是 R1 使用 `clip_max = 1e3`。實驗結果顯示所有非 PDE loss weights 大量飽和：

```text
loss_ic     max clip hit 約 89%
loss_wall   max clip hit 約 76%
loss_inlet  max clip hit 約 74%
loss_outlet max clip hit 約 92%
```

這表示實際需要的 `lambda_hat` 遠大於 1000。換句話說，這個 run 其實不再是 Wang algorithm，而是「大部分時間把所有 boundary / IC / outlet lambda 固定在 1000 附近」。

結果模型不是學到正確物理場，而是被極端 penalty 牽引到錯誤 basin。field L2 幾乎 100%，可視為 collapse。

### 3.3 `strict_grad_norm` uncapped 版本的問題

去掉 practical clip 後，理論上更接近論文。但結果更清楚地暴露了這個 case 的 failure mode：

- `loss_f` 可以降到非常低，例如 `1e-7` 等級。
- 但是 field L2 仍然約 100%。
- 所有 snapshots 都接近 trivial solution。

這代表模型找到了一個幾乎滿足 PDE residual 的解，但不是我們要的流場。最典型的情況是近似 zero-flow / low-flow solution：

```text
u, v 接近 0
PDE residual 很小
但 inlet-driven cylinder flow 完全不對
pressure 也沒有正確物理結構
```

這個問題的核心是：Wang gradient-norm rule 只看 gradient magnitude，不看 constraint 的物理必要性。

在 uncapped run 中，演算法可能認為 inlet term 的 gradient 已經很「大」或很「有效」，因此降低 `lambda_inlet`。但對這個問題來說，inlet profile 是決定非零流場的關鍵 Dirichlet boundary condition。一旦 inlet 被低估，模型就可以滑向 homogeneous PDE 的 trivial solution。

---

## 4. 這算不算 overfit？

可以說它「有類似 overfit 的現象」，但更精準的說法不是傳統 ML overfitting。

傳統 overfit 是：

```text
training data fit 得很好，但 test data/generalization 很差
```

這裡比較像：

```text
weighted training objective 被優化得很好，但這個 objective 本身沒有可靠地代表我們要的物理解
```

因此更合適的描述是：

- constraint imbalance
- degenerate constraint satisfaction
- trivial-solution collapse
- overfitting to the weighted PINN objective

可以在報告中這樣寫：

> The adaptive loss weighting methods exhibited constraint overfitting: they optimized the weighted PINN objective while failing to recover the physical velocity-pressure field.

中文：

> 這不是傳統資料過擬合，而是對加權約束目標的過擬合。模型滿足了某些被演算法偏好的 soft constraints，卻沒有恢復真實的速度與壓力場。

---

## 5. 為什麼 pressure 特別容易壞

在目前 mixed-form PINN 裡，pressure 並沒有像 velocity inlet 一樣被大量直接監督。pressure 主要透過以下方式被間接約束：

- PDE residual
- stress relations
- outlet pressure condition
- streamfunction-induced velocity field

如果 velocity field 已經錯了，pressure 幾乎必然會跟著錯。尤其 trivial / near-zero velocity field 仍可能讓部分 residual 很小，因此 training loss 看起來合理，但 pressure 不會有真實 cylinder flow 的 stagnation / wake 結構。

所以 pressure L2 很差不是單獨的 pressure bug，而是整體流場錯 basin 的結果。

---

## 6. 檢討：這一階段學到什麼

### 6.1 Training loss 不能當唯一指標

R3 最重要的教訓是：

```text
loss_f 極低不代表 field 正確。
```

在 PINN 中，PDE residual 小只代表模型找到了某個 PDE-compatible solution，不代表它滿足正確 boundary-driven physical solution。

未來每個訓練 run 都應該至少檢查：

- field L2_u / L2_v / L2_p
- inlet / wall residual
- probe pressure history
- visualization at representative times
- 是否出現 near-zero velocity field

### 6.2 Soft boundary penalties 是目前 formulation 的弱點

這次 gradient-norm 失敗不是偶然。它顯示 soft penalty BC 對這個 problem 很脆弱：

- inlet 是 time-dependent parabolic profile，是決定非零流場的主要 forcing。
- 如果 inlet 權重被低估，模型會走向 trivial solution。
- 如果 boundary 權重被過度放大，又可能犧牲 interior dynamics。

也就是說，這個問題不是「調一個更好的 lambda」就一定能解。

### 6.3 固定 beta baseline 其實包含重要 prior

原本以為固定 beta = `(wall=5, inlet=5, outlet=1, ic=1)` 只是 heuristic。但實驗顯示它是 load-bearing prior：

```text
它強制保留 inlet / wall 的重要性，避免模型滑向 homogeneous PDE solution。
```

自動 loss weighting 如果只根據 gradient magnitude 來決定權重，可能會移除這個必要 prior。

---

## 7. 為什麼現在可以暫時跳過 loss-balancing

目前已經有足夠證據說明：

1. 原始 grad_norm 實作有公式問題，結果變差。
2. 修正後 clipped strict grad norm 仍然 collapse。
3. uncapped strict grad norm 更明確地走向 trivial solution。
4. 所有結果都無法接近 baseline。

因此繼續在 grad norm 的 `alpha`、`clip_max`、update frequency 上微調，邊際收益很低。比較合理的做法是暫時停止 loss-balancing track，把它作為 negative result 記錄。

這個 negative result 本身有價值：

> Wang-style gradient annealing can fail on boundary-driven unsteady flow problems because gradient-magnitude balancing may underweight physically essential Dirichlet constraints.

---

## 8. 下一步建議：轉向 inverse 和 collocation

目前建議把工作拆成兩條比較可控的路線。

### 8.1 Inverse problem：recover viscosity mu

這條線的優點：

- 是 PINN 經典應用，合理且容易說明。
- 不需要先解決 Re=100 vortex shedding。
- 可以利用現有 Re=10 reference。
- 有明確 metric：`|mu_pred - mu_true| / mu_true`。

建議設計：

- true `mu = 0.005`
- initialize `mu_init = 0.010`，也就是 2x true value
- 加入 sparse sensor velocity data：

```text
loss_sensor = MSE(u_pred(x_s,y_s,t_s), u_ref) + MSE(v_pred(x_s,y_s,t_s), v_ref)
```

總 loss：

```text
loss = loss_physics + beta_bc * loss_bc + lambda_sensor * loss_sensor
```

deliverables：

- mu convergence curve
- final mu error
- sensor count ablation: 5, 10, 20, 50, 100
- noise ablation: 0%, 1%, 5%, 10%

### 8.2 Collocation / sampling study

這條線的目標不是換 loss，而是回答：

```text
PINN 對 collocation distribution 有多敏感？
```

建議比較：

1. baseline LHS sampling
2. more domain collocation points
3. near-cylinder refined collocation
4. inlet / wake-region oversampling
5. time-stratified sampling

評估：

- L2_u / L2_v / L2_p
- near-cylinder error
- early-time vs late-time error
- pressure probe history
- training cost

這比繼續調 gradient-norm 更直接，因為目前問題可能不是單一 loss coefficient，而是 collocation distribution 沒有足夠約束到決定物理解的區域。

---

## 9. 暫時不建議優先做的方向

### 9.1 繼續調 Wang grad norm

不建議。已經試過：

- buggy version
- strict clipped version
- strict uncapped version

都輸給 baseline，而且 strict uncapped 出現明確 trivial collapse。

### 9.2 NTK balancing

暫時不優先。NTK balancing 仍然是 gradient / sensitivity based weighting，可能遇到類似問題：它不一定知道 inlet Dirichlet BC 是物理解的必要 forcing。

### 9.3 Fourier features

Fourier features 對 Re=100 vortex shedding 可能有價值，但目前 Re=10 的 boundary-driven formulation 都還有 collapse 風險。可以之後做，但不是現在解卡關的第一步。

---

## 10. 建議的近期行動清單

1. 把 grad-norm track 標記為 concluded negative result。
2. 保留 B / C / R1 / R3 的 benchmark table，作為報告中的 failure analysis。
3. 新增 inverse branch / script：
   - train with learnable `mu`
   - add sparse sensor loss
   - log `mu(iter)`
4. 新增 collocation experiment config：
   - baseline
   - increased domain points
   - near-cylinder refinement
   - wake / inlet oversampling
5. 每個 run 不只看 training loss，必須跑 `bench_unsteady.py`。

---

## 11. 可放進報告的摘要句

> Although gradient-norm annealing is designed to balance competing PINN loss terms, our boundary-driven unsteady cylinder case exposes a failure mode: gradient statistics alone do not encode which constraints are physically load-bearing. In the uncapped strict implementation, the PDE residual can be minimized to near machine precision while the velocity and pressure fields collapse to a trivial solution, yielding approximately 100% relative field error. This suggests that, for inlet-driven incompressible flows, hard boundary enforcement, sparse data anchoring, or collocation redesign may be more reliable than global scalar loss balancing.
