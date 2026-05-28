# ECE228-PINN Research Plan

**Working title:** Reproducing and Extending Physics-Informed Neural Networks for 2D Unsteady Cylinder Flow
**Target venue:** ECE228 Course Project (primary); potential journal submission (Physics of Fluids / CMAME) if Phase 4 results are strong
**Last updated:** 2026-05-28 (Phase 4A grad-norm track concluded — all three variants fail at field-L2 level; R3 uncapped revealed trivial-solution collapse; pivoting to `self_adap` or hard-BC output transform)

---

## Scope

**Unsteady case only.** Domain: 2D cylinder, x∈[0,1.1], y∈[0,0.41], cylinder (0.2,0.2) r=0.05,
ρ=1.0, U_max=0.5. Paper: Rao / Sun / Liu (2020), arXiv:2002.10558v2.

Two regimes:
- **Re=10** (μ=0.005, T=0.5 s): laminar transient, no vortex shedding — faithful reproduction of Rao 2020
- **Re=100** (μ=0.0005, T=2.0 s): Kármán vortex shedding (St≈0.30, onset t≈1.4–1.6 s) — stress test and improvement target

**Priority order (updated 2026-05-28):**
1. ~~Reproduce paper's unsteady result at Re=10 (Phase 1)~~ ✅ closed as baseline
2. ~~Build Phase 4A scaffold (`none`, `grad_norm`, `strict_grad_norm`)~~ ✅ done
3. ~~Re=10 runs B (`none`) and C (`grad_norm`)~~ ✅ done — B = scaffold sanity (matches vanilla); C 3× worse (buggy L2-norm formula)
4. ~~R1 `strict_grad_norm` clip=1e3, Adam 40k~~ ✅ done — all BC λ saturated, loss_f stuck, field L2 ≈ 100% (collapse)
5. ~~R3 `strict_grad_norm` clip-free, Adam 40k + 4 snapshots~~ ✅ done — loss_f = 1.21e-7 but **all 4 snapshots collapse to trivial solution (L2 ≈ 100%)**
6. ~~Bench all checkpoints~~ ✅ done — see Phase 4A Results Summary below
7. **NEXT:** Pivot to `self_adap` (4A extension) — addresses BC undertraining by min-max λ training (loss-value-driven, not gradient-magnitude-driven)
8. If `self_adap` also fails → switch to **hard BC via output transform** (eliminate inlet/wall loss terms entirely) or **pivot to 4B/4C** (Fourier features / causal training)
9. Re=100 stress test (Phase 3A) — still deferred, scripts ready
10. Fix validated at Re=100; ablation table (Phase 3B / Phase 4)
11. Inverse problem — recover μ from sparse sensors (Phase 2, optional extension)

---

## Narrative arc

> "We faithfully reproduced the Rao 2020 mixed-form PINN at Re=10, establishing a quantitative baseline (L2_u=5.30%, L2_v=19.86%) close to the original TF checkpoint. Phase 1 revealed clear pathologies (high early-time L2_v, SciPy early convergence, Chorin pressure divergence near tmax). Rather than immediately stress-testing at Re=100, we first attack the loss-balancing problem at Re=10 — where ground truth is available — using gradient-norm annealing (Wang 2020). Only after validating a fix do we escalate to Re=100 Kármán shedding. As an extension, we show the same framework can recover fluid viscosity from sparse sensors."

---

## Phase 1 — Faithful Reproduction *(complete — closed as baseline)*

**Result:** Two runs completed and saved locally.
- `strict_reproduce_5090`: SciPy L-BFGS-B, TF-style init, Adam 5k + ~68k evals → L2_u=5.30%, L2_v=19.86%
- `vanilla`: PyTorch LBFGS, Adam 10k + ~63k evals → L2_u=6.8%, L2_v=20.2%
- Reference TF pickle: L2_u=4.75%, L2_v=18.47%

**Decision:** Phase 1 closed as baseline. L2_v ≤ 10% go/no-go is stricter than what the original TF checkpoint achieves; remaining gap (1.4%) is documented as a baseline limitation, not a failure to reproduce.

**Honest framing (for report writing):**
- Paper reports no L2 number for unsteady — only qualitative figures.
- Our reference is a Chorin IBM solver (not Fluent); pressure magnitudes differ due to Chorin splitting error near tmax (documented as limitation).
- "Reproducing the paper" means: flow field qualitatively correct at t=0.3/0.4/0.5 s; L2_u close to TF pickle.

### Bugs to fix before training (researcher-confirmed, unsteady.py)

| Bug | Location | Paper value | Current value | Fix |
|---|---|---|---|---|
| β weight mismatch | unsteady.py:209 | β=2 uniformly | 5·wall + 5·inlet + 1·outlet + 1·ic | `loss = loss_f + 2.0*(loss_wall + loss_inlet + loss_outlet + loss_ic)` |
| L-BFGS budget | unsteady.py:460 | ≥10k, ftol≈eps | default 500 | `--lbfgs-iters 10000`; set `tolerance_change=1e-12` in LBFGS call |
| p_ic term | unsteady.py:205 | IC: v=0 only | IC: u,v,p=0 | remove p from loss_ic (paper does not enforce p=0 at IC) |

### Collocation budget (bring closer to paper)

| Quantity | Paper | Current | Target |
|---|---|---|---|
| N_g (domain) | 120,000 | ~101,000 | 120,000 (raise LHS from 80k→100k) |
| N_db (BC Dirichlet) | 9,600 | ~14,494 | leave — more BC points is fine |
| N_nb (outlet Neumann) | 3,200 | ~3,321 | ✅ |
| N_I (IC) | 3,500 | ~3,321 | ✅ |

### Adam/L-BFGS schedule

- Adam: 10,000 iters (paper steady value; paper doesn't state unsteady count)
- L-BFGS: 10,000 iters with `tolerance_change=1e-12, tolerance_grad=1e-12, history_size=50`

### Evaluation harness — `bench_unsteady.py` (must build from scratch)

Inputs: checkpoint path, `data/reference/unsteady_reference.mat`

Outputs:
- Per-snapshot L2 relative error for u, v (plot vs time)
- Whole-sequence mean L2_u, L2_v (scalars)
- Per-snapshot L2 for demeaned pressure p (Chorin φ is already in mat, use that)
- Probe pressure time histories at P1=(0.15,0.2), P2=(0.20,0.25), P3=(0.25,0.2)
- Side-by-side field comparison figures at t=0.3, 0.4, 0.5 s (reference vs PINN; matching paper Fig 7)

**Validation criteria (go/no-go for Phase 1 complete):**
- Mean L2_u ≤ 10%, Mean L2_v ≤ 10%
- Fig-7-style fields visually symmetric around cylinder, p maximum at stagnation front
- P1/P2/P3 pressure histories in qualitative agreement with Fig 8 (correct phase and shape)

---

## Phase 2 — Inverse Problem: Recover μ from Sparse Sensors *(optional extension)*

**Goal:** Treat μ as unknown; recover it from N_s noisy velocity measurements using the PINN.

**Reference:** Raissi et al. 2019 JCP §4 (canonical PINN inverse problem).

**Note:** The paper (Rao 2020) does NOT include an inverse problem — this is our extension.

### Implementation (~30 LOC, low risk)

1. `PINNLaminarFlowTransient.__init__`: change `self.mu` from float to `nn.Parameter(torch.tensor(mu_init))` (unsteady.py:46)
2. `build_loss`: add sensor data term `loss_sensor = MSE(u_pred(x_s,y_s,t_s), u_s) + MSE(v_pred(...), v_s)` (unsteady.py:197)
3. Training: add `model.mu` to optimizer param group (unsteady.py:247)
4. Data: sample N_s random fluid points from `unsteady_reference.mat`, add Gaussian noise σ=0.01·U_max

### Design choices (fixed)
- True μ = 0.005; initialize at **2× true** (0.010) — standard protocol
- Sensor noise: σ = 0.01 · U_max = 0.005 (1% noise on velocity)
- Single-parameter recovery: μ only

### Ablations
- Sensor count: N_s ∈ {5, 10, 20, 50, 100}; plot |μ_recovered − μ_true|/μ_true vs N_s
- Noise level: fix N_s=20; σ ∈ {0, 0.01, 0.05, 0.1}; plot recovery error vs σ

### Deliverable
- μ(iter) convergence plot from wrong init to true value
- Table: recovery error % vs N_s and vs σ
- Full-field reconstruction from N_s=20 sensors (side-by-side with forward PINN and reference)

---

## Phase 3 — Re=100: Fail, Then Fix

### Phase 3A — Stress Test: Vanilla DNN Fails at Re=100 *(deferred)*

**Status: DEFERRED** — Phase 4A (loss balancer) will be validated at Re=10 first. Re=100 run starts only after a clear Phase 4A result. Scripts are ready.

**Reference dataset:** `unsteady_reference_re100_t2.mat` generated and validated (51 snapshots, t∈[0,2.0]s, interior max|div|=0.023, shedding onset confirmed at t≈1.4–1.6 s).

**Scripts ready:**
- `scripts/train_re100_vanilla.sh` — `--mu 0.0005 --tmax 2.0 --period 1.0`
- `scripts/bench_re100.sh` — `--t-developed 0.5 --field-times 1.0 1.5 2.0`

### Steps (when resumed)
1. Train vanilla 7×50 MLP at Re=100 (`bash scripts/train_re100_vanilla.sh`)
2. Run `bash scripts/bench_re100.sh` → expect L2_u >> 10%, smeared wake, flat probe pressures
3. Document failure modes: wake smear, absent pressure oscillation, high near-wake L2 (x∈[0.25, 0.6])

### Phase 3B — Fix and Validate: Phase 4 Results Evaluated at Re=100

**Goal:** Apply Phase 4 improvements and verify they recover vortex shedding quantitatively. At least one configuration must pass go/no-go (L2_u ≤ 10%, L2_v ≤ 10%) to constitute a publishable contribution.

### Steps
1. Apply each Phase 4 intervention (see Phase 4 below); re-run bench at Re=100 after each
2. Fill ablation table (Phase 4)
3. Run Schäfer-Turek validation for the best configuration (see below)

### Schäfer-Turek validation (Phase 3B deliverable — not optional for paper)
- Schäfer-Turek 2D-2: Cd_max=3.22–3.24, Cl_max=0.990, St=0.300
- Requires a **steady parabolic inlet** (no sin modulation) — separate training run from Phase 1 setup
- Provides third-party ground truth for Re=100 results; essential if targeting journal submission
- Note in report: our sin-modulated inlet differs from the Schäfer-Turek steady BC; document honestly

---

## Phase 4 — Better Models: Fix the Re=100 Failure

**Goal:** Replace/augment the vanilla MLP to resolve vortex shedding at Re=100. Three primary interventions cover three orthogonal axes (loss weighting, input encoding, temporal ordering). Architecture improvements are optional.

### 4A — Phase 4A Results Summary (Re=10) *(updated 2026-05-28; grad-norm track concluded)*

All four runs (B, C, R1, R3) completed and benchmarked on the Re=10 Chorin reference. Train: seed=1234, identical sampling, entrypoint `src/unsteady_balanced.py`. Bench: forward only on MPS, t-developed=0.1s (paper convention).

| Run | balancer | Adam | LBFGS | Final loss_f | **Mean L2_u** | **Mean L2_v** | Verdict |
|---|---|---|---|---|---|---|---|
| Phase 1 vanilla (PyTorch LBFGS) | fixed β = (5,5,1,1) | 10k | ~63k | 2.6e-4 | **5.94%** | **22.23%** | baseline |
| Phase 1 strict_reproduce (SciPy LBFGS) | fixed β = (5,5,1,1) | 5k | ~68k | 2.1e-4 | **5.30%** | **19.86%** | ⭐ best baseline |
| **B** `none_5090` | fixed β (Phase 4A scaffold) | 10k | 50k | ~3e-4 | 6.48% | 22.90% | ✓ scaffold sanity (matches vanilla) |
| **C** `grad_norm_5090` | buggy L2-norm ratio | 10k | 50k | 3.10e-3 | 21.72% | 57.86% | ❌ ~3× worse |
| **R1** `strict_grad_norm_adam40k_5090` | paper-faithful, clip=1e3 | 40k | 0 | 5.50e-2 | **99.83%** | **94.48%** | ❌ collapse |
| **R3** `strict_grad_norm_uncapped_adam40k_5090` (4 snapshots all tested) | paper-faithful, no clip | 40k | 0 | **1.21e-7** | **~100%** | **~100%** | ❌❌ trivial-solution collapse |

**Key finding 1 — buggy `GradNormBalancer` formula:** Our original implementation (`src/loss_balancers.py:37`) used `λ̂_i = max_j ‖∇L_j‖₂ / ‖∇L_i‖₂` (L2-norm ratio). Wang 2020 Algorithm 1 eq. (40) specifies `λ̂_i = max_θ{|∇L_r|} / mean_θ(|∇L_i|)` — max element of the **residual** gradient (fixed anchor), divided by **mean of |∇L_i|**. Statistics matter: mean-of-abs is sensitive to gradient sparsity. L2 norm averages this out and produces λ̂_inlet ≈ 1, letting inlet dominate training. C's 3× degradation is from this bug.

**Key finding 2 — clip rigidity (R1):** With clip_max=1e3 all BC λ saturated 70-92% of update steps. λ_outlet wanted to be > 30k. Paper does not clip; rigid clipping forces a misfit equilibrium.

**Key finding 3 — trivial-solution collapse (R3):** With paper-faithful formula and no clip, strict_grad_norm drives `loss_f` to **1.21e-7** (2000× lower than Phase 1 vanilla final) BUT collapses to a trivial solution with L2 ≈ 100% on every field. Mechanism: the algorithm sets λ_inlet ≈ 0.01 (inlet gradient is "loud" so deemed unimportant), λ_outlet → 32,449. Without effective inlet weight, the model converges to a near-zero field that satisfies the homogeneous PDE perfectly but doesn't match the physical flow. All 4 Adam snapshots {10k, 20k, 30k, 40k} are equally collapsed (the snapshot-bench-gate methodology, which would normally pick a pre-collapse sweet spot, found that collapse happened before iter 10k).

**Key finding 4 — Wang 2020 has a known failure mode against hard Dirichlet BCs:** The paper's Helmholtz/Klein-Gordon benchmarks have BCs that don't strongly select the solution; here, the inlet `u(0,y,t) = parabolic·sin` is what makes the cylinder flow problem well-posed. Gradient-magnitude balancing systematically underweights this term, producing trivial collapse. This is a paper-worthy observation, not just an engineering inconvenience.

**Key finding 5 — vanilla β = (5,5,1,1) is load-bearing, not a guess:** B's success (6.48% L2_u ≈ vanilla) shows the scaffold works; the failure of all gradient-driven balancers shows the fixed β encodes a necessary prior — that BCs matter MORE than gradient norms suggest. Any auto-balancer that doesn't override this prior will fall into trivial collapse.

**Code fixes landed alongside experiments (2026-05-28):**
- `StrictGradNormBalancer` class added to `src/loss_balancers.py` — paper-faithful formula with parameterized clip + saturation tracking
- `--snapshot-iters` mechanism in `src/unsteady_balanced.py` — non-overwriting snapshot files for benchmark-gated checkpoint selection
- `--balancer-clip-min` / `--balancer-clip-max` CLI flags
- 9 bugs fixed via `/code-review` + `/codex-bug-review` (mu restore on `--load-checkpoint`, optimizer state in final save, best_record tracking, L-BFGS resume warning, lambda_curve survival, NaN-safe colorbar, balancer-update-every validation, train_re100_vanilla.sh exp-name parsing, lambda_history gated on resume)

**Decision (gate resolved 2026-05-28):** Pivot to `self_adap` (McClenny & Braga-Neto 2023). Reason: SA-PINN's min-max formulation is **loss-value-driven** rather than gradient-magnitude-driven — high-loss terms (like our struggling inlet) automatically get λ amplified, which is the opposite of strict_grad_norm's behavior. If SA-PINN also fails, the next move is hard BC via output transform (`u_pred = inlet_profile + x·NN_output`), which eliminates the inlet loss term entirely and removes any possibility of trivial collapse.

### 4A — Minimal Loss-Balancer Scaffold (grad_norm first)

**Motivation:** At Re=10 convergence our `loss_f` dominates ~87% of total loss. BC/IC residuals are underfit because PDE gradients overwhelm them by 10²–10⁵×. Phase 4A replaces the static β with a `--balancer` argparse interface.

**Strategy (updated 2026-05-27):** Start minimal — implement only `none` + `grad_norm`. After a Re=10 validation run, decide whether to expand to `ntk` / `self_adap` / `rba`. This avoids building scaffold for methods that may not be needed.

**Files:**
```
src/loss_balancers.py        ← new: Balancer base + NoneBalancer + GradNormBalancer
src/unsteady_balanced.py     ← new: training entrypoint with --balancer {none, grad_norm}
scripts/train_balanced.sh    ← new
scripts/bench_balanced.sh    ← new
results/phase4a_loss_balancing/{method}/   ← per-method checkpoints/logs/figures/benchmarks
```

**First method — `grad_norm` (Wang/Teng/Perdikaris 2020 Alg.1):**
- `λ̂_i = max|∇L_r| / mean|∇L_i|`, EMA α=0.1, update every K=10 Adam steps
- λ frozen at Adam's final value before L-BFGS starts

**CLI (minimal):**
- `--balancer {none, grad_norm}` (default `none` = β=5/5/1/1)
- `--balancer-alpha 0.1`
- `--balancer-update-every 10`
- `--balancer-freeze-lbfgs` (default on)

**New outputs:** `lambda_history.pkl`, `lambda_curve.png`, `checkpoint["balancer_state"]`.

**Decision gate:** After Re=10 `grad_norm` result — if L2_v improves meaningfully, expand scaffold to include `ntk`, `self_adap`, `rba`. If not, revisit loss design before broad work.

**Extensions (deferred until gate clears):**

| Flag | Method | Reference |
|---|---|---|
| `ntk` | NTK-based balancing | Wang/Yu/Perdikaris 2022 JCP |
| `self_adap` | Self-Adaptive PINN | McClenny & Braga-Neto 2023 |
| `rba` | Residual-Based Attention | Anagnostopoulos 2024 |

### 4B — Fourier Feature Encoding (Tancik et al. 2020; Sallam & Fürth 2023) — ~20 LOC

**Motivation (Re=100 specific):** Kármán shedding introduces a temporal frequency ~1.5 Hz (Strouhal mode, St≈0.30) on top of the 1 Hz inlet forcing; wake shear layers are O(50×) thinner than the domain. Standard MLP networks have documented spectral bias against these multi-scale features. Sallam & Fürth 2023 (FF-PINN, J Eng Maritime Environment) applied Fourier-feature PINNs specifically to the Kármán vortex shedding case and showed vanilla PINN fails to recover shedding even with extended training while FF-PINN succeeds. **Note:** no motivation at Re=10 (smooth, low-frequency solution) — apply at Re=100 only.

```
γ(x, y, t) = [sin(2π B [x, y, t]ᵀ), cos(2π B [x, y, t]ᵀ)]   (B fixed random matrix)
```

- Drop-in: replaces the linear first layer; everything else unchanged

### 4C — Causal Training (Wang/Sankaran/Perdikaris CMAME 2024) — ~30 LOC

**Motivation:** Unsteady PINN training with concurrent time collocation can smear the solution to a steady state because later-time residuals provide gradients that conflict with early-time dynamics. Temporal reweighting enforces a causal learning order:

```
w_i = exp(−ε · Σ_{j < i} L_f(t_j))
```

- Orthogonal to 4A (loss-weighting axis) and 4B (input-encoding axis) — stacks with both
- Pure training-loop change; no model change
- Sweep ε ∈ {1, 5, 10}; report which recovers shedding

### 4D — Architecture Improvements *(optional)*

Two options, in order of complexity:

- **Wang 2020 §2.6 (Method B):** gated multiplicative MLP with U/V transformer projections (~100 LOC). Direct companion to 4A in the same paper. Paper shows Method B alone gives ~7× L2 improvement; combined with 4A gives ~55× on Helmholtz. Can be used instead of or alongside PirateNet.
- **PirateNet (Wang et al. JMLR 2025, arXiv:2402.00326):** 2024 successor to Method B with Random Weight Factorization (RWF) initialization; strongest expected gains but heaviest implementation (~120 LOC). Implement only if 4A+4B+4C do not achieve go/no-go at Re=100.

### Ablation table (Re=100)

| Config | Source | L2_u | L2_v | Wake resolved? |
|---|---|---|---|---|
| Vanilla MLP | Phase 3A | — | — | No |
| + 4A grad_norm | Wang 2020 Alg.1 | TBD | TBD | TBD |
| + 4A ntk | Wang 2022 JCP | TBD | TBD | TBD |
| + 4A self_adap | McClenny 2023 | TBD | TBD | TBD |
| + 4A rba | Anagnostopoulos 2024 | TBD | TBD | TBD |
| + Fourier features (4B) | Sallam 2023 FF-PINN | TBD | TBD | TBD |
| + Causal training (4C) | Wang 2024 | TBD | TBD | TBD |
| Best 4A + 4B + 4C | Combined | TBD | TBD | TBD |
| + Architecture (4D) | Wang 2020 §2.6 / PirateNet | TBD | TBD | TBD |

---

## Major Risks

### Risk 1 — Re=100 vortex shedding requires causal training to converge
**Description:** Even with PirateNet + Fourier, vanilla Adam training of an unsteady PINN at Re=100 may not converge within 10,000 Adam iters.
**Mitigation:** Implement causal training before the Re=100 run. Start at Re=50 as a warm-start checkpoint if Re=100 diverges immediately.
**Status:** Open — Phase 4.

### Risk 2 — Chorin reference at Re=100 accuracy
**Description:** The IBM Chorin solver at dx=dy=0.005 has Re-dependent truncation error.
**Mitigation:** N/A — resolved.
**Status:** Closed — Re=100 T=2.0s reference validated: interior max|div|=0.023 (<0.5), flux balanced, KE finite, Kármán shedding confirmed at t≈1.4–1.6 s. dt=0.001 sufficient.

### Risk 3 — L-BFGS convergence on GPU vs paper's SciPy L-BFGS-B
**Description:** PyTorch LBFGS is not bit-equivalent to SciPy L-BFGS-B. Different line search may give slightly different optima.
**Mitigation:** Use `strong_wolfe` line search, `tolerance_change=1e-12`. If loss plateaus, increase `history_size` to 100. This is expected to give equivalent accuracy in practice.
**Status:** Acceptable divergence from paper — document in report.

### Risk 4 — Pressure comparison ambiguity
**Description:** Our reference stores φ (Chorin pressure correction) not physical pressure. PINN is trained with outlet p=0. The scales may differ slightly.
**Mitigation:** Normalize both to zero-mean before L2 comparison. Use probe relative differences (ΔP between probes) rather than absolute values when comparing to Fig 8.
**Status:** Managed — documented in data/reference/README.md.

### Risk 5 — Single-seed results
**Description:** All training runs to date use one random seed. A reviewer asking for mean ± std will get no answer.
**Mitigation:** After Phase 1 go/no-go passes, run 2 additional seeds (total 3). Report mean ± std for L2_u and L2_v. Repeat for the best Phase 3B configuration.
**Status:** Open — add as task after Phase 1 complete.

---

## Techniques Surveyed

| Technique | Decision | Reason |
|---|---|---|
| Wang 2020 Alg.1 grad-norm annealing | **Phase 4A (`grad_norm`)** | Most direct fix for gradient pathology; replaces hand-tuned β |
| NTK-balancing (Wang/Yu/Perdikaris JCP 2022) | **Phase 4A (`ntk`)** | Eigenvalue-based per-term weighting; same-author sequel to Alg.1 |
| SA-PINN (McClenny & Braga-Neto 2023) | **Phase 4A (`self_adap`)** | Per-term trainable λ (min-max); explores adversarial weighting |
| RBA (Anagnostopoulos 2024) | **Phase 4A (`rba`)** | Per-point residual attention; complementary per-point axis to per-term methods |
| Wang 2020 §2.6 improved arch | Phase 4D (optional) | ~55× gain combined with Alg.1; implement if 4A+4B+4C insufficient |
| Fourier feature encoding | **Phase 4B (primary)** | Sallam 2023 FF-PINN directly validates this for Kármán shedding |
| Causal training (Wang 2024) | **Phase 4C (primary)** | Orthogonal temporal axis; prevents smearing to steady state |
| PirateNet (Wang 2025) | Phase 4D (optional) | Successor to Wang 2020 §2.6; implement only if 4A–4C insufficient |
| RAD adaptive sampling | Optional Phase 4 extension | Low priority vs loss-weighting and architecture improvements |
| KAN-PINN | Stretch only | Install risk on Linux GPU; less established |
| Schäfer-Turek exact validation | **Phase 3B deliverable** | Required for journal submission; moved from deferred |
| Multi-parameter inverse (μ + U_max) | Stretch | Phase 2 single-param recovery must work first |
| Ensemble / Bayesian PINN UQ | Out | Low priority |
| FNO / PINO | Out | Different paradigm; dilutes PINN narrative |
| Separable PINN | Out | Conflicts with cylinder geometry |
