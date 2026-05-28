# ECE228-PINN TODO

**Last updated:** 2026-05-28 (Phase 4A grad-norm track concluded; all variants fail bench; pivoting to `self_adap`)

Priority: top = most immediate. Items marked **[BLOCKED]** cannot start yet. See `docs/plan.md` for full context.

---

## Immediate — Phase 4A pivot to `self_adap`

Grad-norm track concluded — all 3 variants fail at field-L2 level (see `docs/plan.md` 4A Results Summary). Bench results landed 2026-05-28; comparison table:

| Method | Mean L2_u | Mean L2_v | Verdict |
|---|---|---|---|
| Phase 1 strict_reproduce (SciPy LBFGS) | **5.30%** | **19.86%** | ⭐ best baseline |
| Phase 1 vanilla (PyTorch LBFGS) | 5.94% | 22.23% | baseline |
| B `none_5090` (Adam 10k + LBFGS 50k) | 6.48% | 22.90% | scaffold sanity ✓ |
| C `grad_norm_5090` (buggy L2-norm) | 21.72% | 57.86% | ❌ |
| R1 `strict_grad_norm` clip=1e3 | 99.83% | 94.48% | ❌ collapse |
| R3 `strict_grad_norm` no-clip (any of 4 snapshots) | ~100% | ~100% | ❌❌ trivial collapse |

### Next actions

- [ ] **Implement `self_adap` (McClenny & Braga-Neto 2023)** in `src/loss_balancers.py`:
  - λ_i as `nn.Parameter` (reparameterized via softplus to keep > 0)
  - Train λ by **gradient ASCENT** (or equivalently, negate λ's gradient before Adam step)
  - `compute_total = Σ softplus(α_i) · L_i`
  - λ frozen during L-BFGS (consistent with existing balancers)
  - Smoke test on MPS, then optionally re-rent pod for Re=10 run
- [ ] Re=10 validation: compare against B (none) and Phase 1 baselines on L2_u, L2_v
- [ ] If `self_adap` also collapses → implement **hard BC via output transform** in `unsteady.py` (`u_pred = inlet_profile(y,t)·(1-x) + x·NN_out`), eliminating inlet/wall loss entirely
- [ ] Optional sanity check (cheap, on MPS): from `strict_grad_norm_uncapped_adam40k_5090/snapshot_iter10000.pt`, run LBFGS 50k locally to confirm LBFGS cannot rescue trivial-basin convergence

### Phase 4A archive (negative results worth recording in report)

- [x] Bench all checkpoints (B, C, R1, R3 ×4 snapshots, Phase 1 ×2) → `/tmp/bench_logs/SUMMARY.txt`
- [ ] Document the three failure modes (buggy L2-norm formula, clip rigidity, trivial collapse) as a paper-worthy negative result about Wang 2020 Algorithm 1 on hard-Dirichlet-BC problems

---

## Deferred — Report Writing

- [x] Researcher: survey bib keys + result numbers (2026-05-27)
- [x] latex-writer: `report/main.tex` (IEEEtran top-level structure) (2026-05-27)
- [x] latex-writer: `report/sections/methodology.tex` (2026-05-27)
- [x] latex-writer: `report/sections/results_phase1.tex` (2026-05-27)
- [ ] User: review draft and resolve all `\todo{}` markers

---

## Phase 1 — Re=10 Baseline *(complete)*

**Decision (2026-05-27):** Phase 1 is closed as a baseline, not as a full paper-faithful reproduction. The reference implementation itself gives ref `L2_v = 18.47%`; our best run reaches `19.86%`, which is close enough to use as a baseline for loss-improvement work. Remaining pressure/probe mismatch is documented as a limitation of the baseline.

**Saved checkpoints (local):**
- `results/checkpoints/vanilla/` — PyTorch LBFGS, Adam 10k + L-BFGS ~63k evals; L2_u=6.8%, L2_v=20.2%
- `results/checkpoints/strict_reproduce_5090/` — SciPy L-BFGS-B, Adam 5k + ~68k evals (converged); L2_u=5.30%, L2_v=19.86%

- [x] Bug fixes, eval harness, checkpoint isolation (2026-05-25/26)
- [x] vanilla run: Adam 10k + PyTorch LBFGS ~63k evals; L2_u=6.8%, L2_v=20.2% (2026-05-26)
- [x] strict_reproduce run: SciPy L-BFGS-B, TF-style init, 80k LHS; L2_u=5.30%, L2_v=19.86% (2026-05-27)
- [x] Per-snapshot L2 matrix produced for both runs (2026-05-27)
- [x] All checkpoints and logs saved locally (2026-05-27)

---

## Phase 3A — Re=100 Stress Test

**[DEFERRED — skipping for now, decision 2026-05-27]**

Reference data:
- [x] Generate `unsteady_reference_re100_t2.mat` (51 snapshots, t∈[0,2.0]s, Kármán shedding confirmed) (2026-05-26)
- [x] Validate: interior max|div|=0.023, flux OK, KE finite (2026-05-26)
- [x] Scripts ready: `scripts/train_re100_vanilla.sh`, `scripts/bench_re100.sh` (2026-05-27)

Training and evaluation:
- [ ] Train vanilla 7×50 MLP at Re=100 (`bash scripts/train_re100_vanilla.sh`)
- [ ] Run `bash scripts/bench_re100.sh` → expect L2_u >> 10%, smeared wake
- [ ] Document failure modes: wake smear, flat probe pressures, high near-wake L2
- [ ] Save as baseline row in ablation table

---

## Phase 4 — Loss / Training Improvements

**[ACTIVE — strict_grad_norm clip-free in progress, decision 2026-05-28]**

### 4A — Loss-Balancer Scaffold *(scaffold complete, runs B/C/R1 done)*

Scaffold:
- [x] Create `src/loss_balancers.py`: `Balancer` base + `NoneBalancer` + `GradNormBalancer` + `StrictGradNormBalancer` (2026-05-27/28)
- [x] Create `src/unsteady_balanced.py`: training entrypoint with `--balancer {none, grad_norm, strict_grad_norm}` (2026-05-27)
- [x] Create `scripts/train_balanced.sh` and `scripts/bench_balanced.sh` (2026-05-27)
- [x] Persist `balancer_state` in checkpoint; log `lambda_history.pkl` + plot `lambda_curve.png` (2026-05-27)
- [x] Smoke test on MPS for `none`, `grad_norm`, `strict_grad_norm` (2026-05-27/28)
- [x] Fix 9 review-found bugs (mu restore, optimizer state in final save, best_record tracking, L-BFGS resume warning, lambda_curve survival, NaN-safe colorbar, balancer-update-every validation, train_re100_vanilla.sh exp-name parsing, lambda_history gated on resume) (2026-05-28)
- [x] Add `--snapshot-iters` + clip parameterization + saturation tracking (2026-05-28)

Completed runs (RTX 5090):
- [x] **B `none_5090`** — Adam 10k + LBFGS 50k, final total ~3e-4, matches vanilla magnitude (2026-05-27)
- [x] **C `grad_norm_5090`** — buggy L2-norm version: final loss_f=3.10e-3 (12× worse than vanilla); diagnosis: λ_inlet stuck at 1, inlet loss dominates (2026-05-27)
- [x] **R1 `strict_grad_norm_adam40k_5090`** — paper-faithful with clip=1e3, Adam 40k only: all BC λ saturated 70-92%, loss_f=5.5e-2 (13× worse) (2026-05-28)
- [x] All checkpoints + logs + figures pulled to local (results/ untouched Phase 1 preserved) (2026-05-28)

In progress:
- [ ] **R3 `strict_grad_norm_uncapped_adam40k_5090`** — paper-faithful with clip removed (clip_max=1e12), Adam 40k + snapshots {10k, 20k, 30k, 40k}; early signal: loss_f=2.4e-4 at iter 9500 (18× better than vanilla Adam), but inlet stuck at 0.19 because λ_inlet → 0.13 (algorithm de-prioritizes inlet)

Pending after R3:
- [ ] **R2 `strict_grad_norm_lbfgs50k_5090`** — LBFGS 50k from best R3 snapshot by L2 (not training loss)
- [ ] Decision gate: see Decision gate section above

### 4A Extensions — Additional Balancers *(prioritized by R3 outcome — grad-norm dead)*

- [ ] **`self_adap` — Self-Adaptive PINN (McClenny & Braga-Neto 2023): HIGH PRIORITY** — λ_i as `nn.Parameter` trained min-max. Directly addresses "algorithm de-prioritizes hard BC" failure mode by adversarially amplifying **high-loss** terms (opposite direction from grad-norm's gradient-magnitude rule). Predicted to avoid trivial collapse because λ_inlet grows when L_inlet is large.
- [ ] **Hard BC via output transform: BACKUP** — bypass loss-balancing entirely. Modify `net_uv` so `u(0, y, t)` algebraically equals the parabolic inlet, `u(wall) = 0`, etc. (e.g., `u_pred = (1-x)·inlet_profile + x·NN_u`). Eliminates inlet/wall loss terms; only loss_f + loss_ic + loss_outlet remain. Removes any possibility of trivial collapse.
- [ ] `ntk` — NTK-based balancing (Wang/Yu/Perdikaris 2022 JCP): λ_i from `trace(K_ii)`. **Likely same failure mode as grad-norm** — both are gradient/sensitivity-based; expected to also under-weight strongly-constrained BCs. **De-prioritized** unless self_adap fails AND we want a control-group balancer for the report.
- [ ] `rba` — Residual-Based Attention (Anagnostopoulos 2024): per-point weights inside `loss_f` only. **Doesn't address BC weighting**, our actual failure mode. **De-prioritized**.

### 4B — Fourier Feature Encoding *(deferred)*

- [ ] Replace linear first layer with `γ(x,y,t) = [sin(2π B·[x,y,t]ᵀ), cos(2π B·[x,y,t]ᵀ)]` (B fixed random)
- [ ] Motivation: Kármán shedding at Re=100 creates multi-scale spectral structure; FF-PINN validated on this exact case
- [ ] Train at Re=100; run bench; compare to 4A

### 4C — Causal Training *(deferred)*

- [ ] Implement temporal residual reweighting: `w_i = exp(-ε · Σ_{j<i} L_f(t_j))`
- [ ] Sweep ε ∈ {1, 5, 10}; report which recovers vortex shedding
- [ ] Train at Re=100; run bench; compare to 4A and 4B

### Ablation table (Re=100, later)

- [ ] Fill table: Vanilla / +4A / +4B / +4C / 4A+4B+4C for L2_u, L2_v, Wake resolved?

### 4D — Architecture Improvements *(optional — implement only if loss interventions are insufficient)*

- [ ] Wang 2020 §2.6 improved arch: gated MLP with U/V projections (~100 LOC); companion to 4A
- [ ] Or PirateNet (arXiv:2402.00326): RWF init + adaptive residual blocks (~120 LOC)

---

## Phase 3B — Fix Validated at Re=100

**[BLOCKED on Phase 4 ablation complete]**

- [ ] Re-run multi-seed (3 seeds) for best Phase 4 configuration at Re=100; report mean ± std
- [ ] Schäfer-Turek validation: train with **steady parabolic inlet** (no sin modulation); compute Cd_max, Cl_max, St; compare to Schäfer-Turek 2D-2 benchmark (Cd_max=3.22–3.24, Cl_max=0.990, St=0.300)
- [ ] Note inlet difference honestly in report (sin-modulated vs steady parabolic)

---

## Phase 2 — Inverse Problem: Recover μ from Sparse Sensors *(optional extension)*

**[DEFERRED — start after Phase 3B complete, if time allows]**

- [ ] Make `mu` an `nn.Parameter` in `PINNLaminarFlowTransient.__init__` (unsteady.py:46)
- [ ] Add sensor data loss to `build_loss` (unsteady.py:197): `loss_sensor = MSE(u_pred(x_s,y_s,t_s), u_s) + MSE(v_pred(...), v_s)`
- [ ] Add `model.mu` to optimizer param group (unsteady.py:247)
- [ ] Sample N_s=20 random fluid points from `unsteady_reference.mat` with Gaussian noise σ=0.005; initialize μ at 0.010 (2× true)
- [ ] Run sensor count ablation: N_s ∈ {5, 10, 20, 50, 100}; plot recovery error vs N_s
- [ ] Run noise robustness ablation: fix N_s=20; σ ∈ {0, 0.01, 0.05, 0.1}; plot recovery error vs σ
- [ ] Produce μ(iter) convergence plot and recovery error table

---

## Deferred / stretch

- [ ] Multi-parameter inverse (μ + U_max simultaneously) — stretch; Phase 2 single-param must work first
- [ ] PirateNet backbone (arXiv:2402.00326) — optional Phase 4D; implement only if 4A+4B+4C insufficient
- [ ] Wang 2020 §2.6 improved arch — optional Phase 4D; companion to loss annealing (4A)
- [ ] RAD adaptive sampling (Wu et al.) — optional Phase 4 extension
- [ ] KAN-PINN — stretch only; install risk on GPU

---

## Open questions

- Bench evidence (R3) suggests Wang 2020 fails on hard-Dirichlet-BC problems. Report this as a **primary paper finding** (a known failure mode of a popular algorithm, with mechanism explained) or treat as **implementation note** (we tried it, didn't work, here's our better method)?
- For `self_adap`, should λ_i be reparametrized via softplus (`λ = log(1+exp(α))`, smooth), exponential (`λ = exp(α)`, faster growth), or quadratic (`λ = α²`, used in original McClenny paper)? Original paper uses quadratic; we may need to tune.
- If `self_adap` also fails → hard BC output transform is the strongest fallback. Should it be implemented in `unsteady.py` (replaces vanilla forward) or as a new `unsteady_hardbc.py` (preserves vanilla for ablation)? Latter for cleanliness.
- If no Phase 4A variant beats vanilla on L2, do we still try 4B/4C, or pivot to pure architecture work (Wang 2020 §2.6 / PirateNet)?
- If no Phase 4 intervention succeeds at Re=100, fall back to Re=50 as the stress-test case?
