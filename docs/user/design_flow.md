# Report Slide Design Flow

**Purpose:** 討論並規劃報告投影片前三頁。前兩頁需要後續用 Excalidraw MCP 生成流程圖與架構圖；第三頁以目前實驗結果為主。

---

## Slide 1 — Reference Methodology: Mixed-Form PINN for Cylinder Flow

### Goal

第一頁先交代我們站在哪些論文方法上：PINN 的基本框架、Rao 2020 的 mixed-form incompressible flow setup、以及後續 gradient pathology / causal learning / inverse extension 為什麼自然接上。

### Slide Message

> We reproduce and extend a mixed-form PINN for unsteady incompressible flow around a cylinder. The method represents velocity, pressure, and stress fields with a neural network, then trains by enforcing Navier-Stokes residuals plus boundary and initial constraints.

### Content Blocks

- **Physical problem**
  - 2D incompressible laminar cylinder flow.
  - Inputs: `x, y, t`.
  - Outputs: velocity `u, v`, pressure `p`, stress components.

- **PINN formulation**
  - Neural network predicts fields.
  - Automatic differentiation computes PDE residuals.
  - Loss combines PDE residual, initial condition, wall no-slip, inlet profile, and outlet pressure.

- **Optimization**
  - Adam warm-up.
  - L-BFGS / SciPy L-BFGS-B polishing.
  - Evaluation with field L2 and pressure probes.

- **Reference papers to cite on slide**
  - Raissi et al. 2019: general PINN forward / inverse framework.
  - Rao, Sun, Liu 2020: mixed-form PINN for incompressible laminar flow.
  - Wang et al. 2020: gradient pathologies and loss balancing.
  - Wang, Sankaran, Perdikaris 2024: causal training for time-dependent PINNs.

### Excalidraw Diagram Spec

**Diagram type:** left-to-right methodology pipeline.

**Nodes:**

1. `Physical setup`
   - cylinder domain
   - inlet / wall / outlet / IC
2. `Training points`
   - collocation points
   - boundary points
   - initial points
3. `Mixed-form neural network`
   - input: `(x, y, t)`
   - output: `(u, v, p, stress)`
4. `Autograd residuals`
   - continuity
   - momentum
   - constitutive stress equations
5. `PINN loss`
   - `L_f + beta_wall L_wall + beta_inlet L_inlet + beta_outlet L_outlet + beta_ic L_ic`
6. `Optimizer`
   - Adam
   - L-BFGS
7. `Evaluation`
   - field L2
   - pressure probes
   - field snapshots

**Edges:**

- `Physical setup -> Training points`
- `Training points -> Mixed-form neural network`
- `Mixed-form neural network -> Autograd residuals`
- `Autograd residuals -> PINN loss`
- `PINN loss -> Optimizer`
- `Optimizer -> Mixed-form neural network` as feedback loop
- `Mixed-form neural network -> Evaluation`

**Visual style:**

- Clean academic diagram.
- Use blue for model/PDE pipeline.
- Use orange for loss terms.
- Use green for evaluation.
- Add small citation labels under relevant blocks:
  - `Raissi 2019`
  - `Rao 2020`

---

## Slide 2 — Development Flow: Application Layer + Model Optimization Layer

### Goal

第二頁要呈現我們不是只做單一 reproduction，而是把 baseline 變成一個可擴展研究平台。新的敘事分成兩層：

1. **Application layer:** 把方法推到不同任務與資料設定，包括 inverse problem 與 Re=100 dataset。
2. **Model optimization layer:** 針對 PINN 訓練失敗模式做演算法改進，包括 complex model、loss weight、adaptive collocation、causal learning。

### Slide Message

> After establishing a Re=10 baseline, we separate the project into application extensions and model-optimization interventions. Application experiments test whether the method generalizes, while optimization experiments isolate which training changes actually improve the PINN.

### Research Flow

#### 1. Application / model generalization

**1.1 New model vs MLP**

- Compare vanilla MLP against more expressive architectures such as PirateNet, Fourier-feature MLP, gated MLP, or wider/deeper MLP.
- Evaluate the comparison not only on the original Re=10 setup, but also on other application scenarios:
  - Re=100 dataset as a harder forward problem.
  - Inverse problem as a parameter-identification task.
- Main question: is the architecture improvement robust, or does it only help one benchmark?

#### 2. Model optimization path

**2.1 Vanilla + loss-weight optimization**

- Start from the stable vanilla baseline.
- Search fixed or adaptive loss weights during Adam only.
- Candidate methods:
  - fixed beta baseline `(wall=5, inlet=5, outlet=1, ic=1)`;
  - revised gradient annealing only if boundary collapse is controlled;
  - self-adaptive PINN min-max weights;
  - hard-BC output transform as a fallback.

**2.2 Add adaptive collocation**

- Add residual-guided collocation after the baseline and loss weights have a stable metric.
- Use a hybrid fixed/adaptive collocation budget:
  - early phase: `fixed/adaptive = 90/10`;
  - middle phase: `fixed/adaptive = 80/20`;
  - late phase: `fixed/adaptive = 50/50`.
- The fixed set remains a global anchor to preserve domain coverage and prevent forgetting.
- The adaptive set is mutable and keeps the total collocation count constant.
- Every fixed `K_adapt` Adam iterations:
  1. Evaluate unweighted PDE residuals on current adaptive points and candidate neighbor points.
  2. Select the worst residual locations.
  3. Add one valid neighboring adaptive point around each selected worst location.
  4. Delete the same number of lowest-residual adaptive points.
  5. Keep the total number of collocation points unchanged.
- Reference logic from `docs/reference.md`:
  - RAR: find high-residual regions.
  - R3 and modified RAR with deletion: retain hard points and release solved points.
  - RAD: preserve global supervision through a fixed/background distribution.
  - Causal PINN: avoid adaptive refinement that violates time ordering.

**2.3 Add causal learning**

- Add time-bin causal weighting after loss-weight and collocation variants are defined.
- Candidate sweep:
  - `eps = 10, 30, 100`;
  - Adam monitoring for both weighted loss and unweighted true loss;
  - L-BFGS modes only after causal weights/objective are frozen.

**2.4 Assemble strongest model**

- Pick the best validated mix from 2.1, 2.2, and 2.3.
- Combine the strongest loss weighting, collocation policy, and causal policy into one final model.
- Re-benchmark on:
  - Re=10 reproduction;
  - Re=100 forward dataset;
  - inverse problem extension if implemented.

### Optimization Rule

所有「搜尋」或「自適應」步驟都只放在 Adam 階段：

- loss-weight search;
- residual search for adaptive collocation;
- insertion/deletion of adaptive collocation points;
- causal weight schedule or epsilon search;
- model-selection sweeps.

L-BFGS / SciPy L-BFGS-B 的角色是 final polishing，優化目標必須穩定：

- fixed collocation set;
- fixed loss weights;
- fixed causal weights or fixed causal objective;
- no point insertion/deletion;
- no hyperparameter search.

This avoids turning L-BFGS into a moving-objective optimizer, which would make the quasi-Newton curvature estimate unreliable.

### Layer Breakdown

#### Application layer

**Inverse problem**

**Problem targeted:** demonstrate that the framework can recover unknown physical parameters from sparse observations.

**Candidates:**

- Learn viscosity `mu` as `nn.Parameter`.
- Sparse velocity sensors sampled from reference data.
- Sensor-count ablation: `N_s = 5, 10, 20, 50, 100`.
- Noise robustness ablation.
- Multi-parameter extension: `mu + U_max`.

**Status:**

- Planned extension.
- Should be shown as application-layer future scope unless implemented.

**Re=100 dataset**

**Problem targeted:** test whether the method survives a harder flow regime and more complex temporal/spatial dynamics.

**Candidates:**

- Use the tracked Re=100 reference dataset as a forward-problem stress test.
- Compare vanilla MLP vs complex models under the same training/evaluation protocol.
- Report whether improvement transfers from Re=10 to Re=100.

**Status:**

- Data is available.
- Training/evaluation protocol still needs to be finalized.

#### Model optimization layer

**Complex model**

**Problem targeted:** vanilla MLP spectral bias and weak representation of vortex shedding.

**Candidates:**

- Fourier feature encoding.
- Wang 2020 gated architecture / Method B.
- PirateNet / residual adaptive architecture.
- Wider/deeper MLP baseline.

**Status:**

- Planned for Re=100 stress case.
- Not the first fix for Re=10 pressure issue.

**Loss weighting**

**Problem targeted:** soft constraints compete; PDE residual can dominate or boundary terms can be underweighted.

**Candidates:**

- Fixed beta baseline `(wall=5, inlet=5, outlet=1, ic=1)`.
- GradNorm L2-ratio variant.
- Strict Wang 2020 gradient annealing.
- Self-adaptive PINN min-max weights.
- Hard-BC output transform as fallback.

**Status:**

- Fixed beta remains best usable baseline.
- GradNorm variants failed or collapsed.
- Self-adaptive / hard BC are next logical candidates.

**Adaptive collocation**

**Problem targeted:** residual errors are spatially and temporally localized, especially near cylinder, wake, inlet, and late-time dynamics.

**Candidates:**

- Fixed/adaptive schedule: `90/10 -> 80/20 -> 50/50`.
- Constant-budget residual refinement.
- Add one neighbor near each worst-residual adaptive location.
- Delete the same number of best/lowest-residual adaptive points.
- Respect causal time-front when causal training is enabled.

**Status:**

- Planned.
- Should be evaluated after loss weights define stable metrics.
- Must run only during Adam; L-BFGS receives frozen points.

**Causal learning**

**Problem targeted:** time-dependent PINNs can smear dynamics by learning all times simultaneously.

**Candidates:**

- Causal temporal bins.
- Epsilon sweep: `eps = 10, 30, 100`.
- L-BFGS modes: uniform true loss, frozen causal weights, dynamic weights.
- Monitor both weighted loss and unweighted true loss.

**Status:**

- `src/unsteady_causal.py` exists.
- Initial Adam-10k sweep exists for `causal_eps10`, `causal_eps30`, `causal_eps100`.
- Needs benchmark before claiming improvement.

### Excalidraw Diagram Spec

**Diagram type:** two-layer roadmap with one sequential research path.

**Top-level structure:**

- Left anchor: `Re=10 baseline reproduction`
- Middle split:
  - `Application layer`
  - `Model optimization layer`
- Right anchor: `Final benchmark + report`

**Application layer boxes:**

- `Re=100 stress test`
  - harder forward dataset
  - transfer from Re=10
- `Inverse problem`
  - learn `mu`
  - sparse sensors
  - noise/count ablation

**Model optimization layer boxes:**

- `1.1 Complex model vs MLP`
  - PirateNet
  - Fourier features
  - gated MLP
- `2.1 Loss weighting`
  - fixed beta
  - self-adaptive
  - hard BC
- `2.2 Adaptive collocation`
  - `90/10 -> 80/20 -> 50/50`
  - add worst-neighbor
  - delete lowest residual
  - constant budget
- `2.3 Causal learning`
  - temporal bins
  - eps sweep
  - true-loss monitoring
- `2.4 Strongest model`
  - best mix of 2.1-2.3
  - benchmark on Re=10 / Re=100 / inverse

**Required optimizer gate:**

Add a visible gate near the bottom of the diagram:

```text
Adam: search / adapt / tune
        ->
Freeze objective
        ->
L-BFGS: stable polishing only
```

**Status marks:**

- Green check: completed / usable baseline.
- Red cross: tested and failed.
- Yellow clock: in progress or needs benchmark.
- Gray dotted outline: future extension.

**Specific status labels:**

- `Fixed beta baseline`: green.
- `GradNorm / strict GradNorm`: red.
- `Causal eps sweep`: yellow.
- `Adaptive collocation`, `Re=100`, `inverse`: gray/yellow depending on final narrative.
- `Self-adaptive`, `hard BC`, `PirateNet`: gray/yellow depending on final narrative.

**Visual style:**

- Two horizontal bands:
  - upper band: application layer;
  - lower band: model optimization layer.
- Sequential arrows through the optimization layer:
  - `1.1 -> 2.1 -> 2.2 -> 2.3 -> 2.4`.
- Arrows from `2.4 Strongest model` to application boxes.
- Use compact cards, not dense text.
- The figure should communicate both research coverage and execution order in one glance.

---

## Slide 3 — Current Results: Baseline, Failure Modes, and Early Causal Sweep

### Goal

第三頁秀目前有數字支撐的結果，並誠實區分：

1. 已完成且可信的 baseline。
2. 已確認失敗的 loss-weighting negative results。
3. 還只是初步訓練訊號、尚未 benchmark 的 causal sweep。

### Slide Message

> The baseline reproduces velocity fields reasonably but not pressure probes. Gradient-norm loss balancing failed by underweighting physically essential boundary constraints. Causal training is now the active candidate, but must be benchmarked before claiming improvement.

### Result Table Draft

| Track | Run | Key signal | Interpretation |
|---|---|---:|---|
| Baseline | `strict_reproduce_scipy` | L2_u 5.30%, L2_v 19.86%, L2_p 10.79% | best usable baseline, pressure still weak |
| Baseline | `vanilla_pytorch` | L2_u about 6%, L2_v about 20-23% | stable comparison baseline |
| Loss weighting | `grad_norm_5090` | L2_u 21.72%, L2_v 57.86% | worse than baseline |
| Loss weighting | `strict_grad_norm` clipped | L2 about 100% | collapse |
| Loss weighting | `strict_grad_norm` uncapped | loss_f very low, L2 about 100% | trivial PDE-compatible solution |
| Causal learning | `causal_eps10` | best train loss 0.02427, final min_w 0.889 | closest to causality-satisfied gate, benchmark pending |
| Causal learning | `causal_eps30` | best train loss 0.02418, final min_w 0.667 | stronger causal gating, benchmark pending |
| Causal learning | `causal_eps100` | best train loss 0.01730, final min_w 0.110 | lowest weighted loss, but later-time weights still suppressed |

### Figure Candidates

- **Primary table:** compact result matrix with verdict colors.
- **Plot A:** L2 comparison bar chart for completed runs.
- **Plot B:** causal `min_w` / weight curves for eps sweep.
- **Plot C:** field snapshots or pressure probe comparison for baseline vs failed GradNorm.

### Slide 3 Caution

Do not claim causal learning improves physical accuracy until `bench_unsteady.py` is run on:

- `results/checkpoints/causal_eps10/best.pt`
- `results/checkpoints/causal_eps30/best.pt`
- `results/checkpoints/causal_eps100/best.pt`

Minimum benchmark output needed:

- Mean L2_u, L2_v, L2_p.
- Probe pressure curve.
- Field comparison at `t = 0.3, 0.4, 0.5`.

---

## Excalidraw MCP Generation Plan

### First Excalidraw call: Slide 1 methodology diagram

Use the Slide 1 diagram spec above. The output should be a clean academic pipeline figure titled:

```text
Mixed-Form PINN Methodology for Unsteady Cylinder Flow
```

Target output:

- one Excalidraw scene
- landscape aspect ratio
- enough whitespace for later slide embedding
- no long paragraphs inside boxes

### Second Excalidraw call: Slide 2 development roadmap

Use the Slide 2 diagram spec above. The output should be a roadmap figure titled:

```text
Development Flow: Application Layer and Model Optimization Layer
```

Target output:

- one Excalidraw scene
- two clearly separated horizontal bands
- sequential optimization path `1.1 -> 2.1 -> 2.2 -> 2.3 -> 2.4`
- visible Adam-only search gate before frozen L-BFGS polishing
- status icons/colors
- compact candidate labels

### Follow-up after diagrams

After the two diagrams are generated, use them to finalize:

- exact slide titles
- one-sentence takeaway per slide
- whether Slide 3 should be a table-heavy results slide or a chart-heavy results slide

---

## Immediate TODO

- [ ] Run benchmark on causal eps checkpoints before Slide 3 finalization.
- [ ] Decide exact Adam iteration milestones for adaptive split changes: `90/10 -> 80/20 -> 50/50`.
- [ ] Decide `K_adapt`, number of worst-residual points per update, and neighbor radius/jitter rule.
- [ ] Decide whether Slide 2 should present `self_adap` as "next planned" or down-rank it now that causal learning has started.
- [ ] Generate Slide 1 Excalidraw methodology diagram.
- [ ] Generate Slide 2 Excalidraw development-flow roadmap.
- [ ] Convert the final three-slide design into presentation-ready text and figures.
