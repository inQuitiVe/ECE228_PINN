# ECE228-PINN-Workspace

---

## RULE 100 — NEVER EDIT FILES WITHOUT EXPLICIT INSTRUCTION

Do not edit any file — report, docs, code, bib — unless the user has explicitly said to make the change. After research, analysis, or drafting, always report findings and wait for the user to approve before touching any file.

---

## RULE 101 — NEVER INVENT CITATIONS OR RESULTS

**This is the most important rule when assisting with report writing.**

- Never fabricate, guess, or hallucinate a citation key, paper title, author, venue, or year.
- Never invent or estimate numerical results, measurements, or experiment outcomes.
- If a citation or result is needed and not available, write a `\todo{}` placeholder and flag it explicitly.
- When suggesting text that would require a citation, always say "needs citation" rather than inventing one.

## RULE 102 — ONLY ONE ACTIVE REPORT WRITER

Only one agent may work on report text at a time. Do not assign two writing tasks that modify report/ concurrently. If multiple report tasks are ready, queue them in docs/todo.md and wait for the user to choose or approve the next one.

RULE 102 still applies: only one active report writer at a time.

## RULE 103 — PROPOSE BEFORE EXECUTING

Before drafting, coding, refactoring, surveying papers, or changing docs, first produce a short task plan:

- goal
- files/context to inspect
- expected output
- risks or missing information
- what user decision is needed, if any

Wait for user approval before proceeding.

## RULE 104 — LATEX WRITING

All `.tex` file writes and updates anywhere in `report/` are owned by the `latex-writer` teammate in the `fastcim-review` team.

To trigger latex-writer, tell the team-lead to write or update LaTeX. latex-writer will:
- Re-read `docs/writing_style.md` on every task.
- Write `.tex` files only — no `.bib`, no figures.
- Mark anything requiring user input as `\todo{...}` rather than inventing content (RULE 101).
- Report all `\todo{}` markers placed back to team-lead when done.

`implementer-verifier` reviews every latex-writer diff for scope compliance (`.tex` files in `report/` only, no `.bib`, no figures) before the change is considered accepted.



## RULE 105 — LOG PENDING USER DECISIONS TO docs/user/pending_decisions.md

Whenever a question requires the user's decision or approval (scope, design choice, ambiguity, missing information, anything blocking progress), append an entry to `docs/user/pending_decisions.md`. Always log regardless of how small the decision is. Do not rely solely on inline questions in chat.

Match the existing format in that file:

```
### <short title>

<1–3 sentences of context. If a claim or plan is at risk, name it explicitly.>

**Decision needed:** <one-line question, list options if applicable>

---
```

Rules:
- Strictly use `docs/user/pending_decisions.md`. Do not spread pending decisions to other files (`docs/meetings/`, `docs/todo.md`, etc.).
- When the user resolves an entry, delete the entry from the file.
- Update the `**Last updated:**` date at the top whenever you add or remove entries.
- You may still ask the same question inline in chat for fast turnaround, but the file is the source of truth.

---

## RULE 106 — TEAM-LEAD: CONSULT RESEARCHER BEFORE WRITING PLANS

Team-lead is the plan writer and sole decision gate. Before drafting any plan, updating `docs/plan.md`, or updating `docs/todo.md`, consult researcher first according to the task type:

1. **Code improvement tasks** (required): dispatch researcher to survey the relevant code area. Wait for researcher's fact report. Draft the plan from those facts — do not invent findings.
2. **`docs/plan.md` or `docs/todo.md` updates** (required): dispatch researcher to read any relevant raw resources, meeting notes, existing docs, or code areas the plan touches. Wait for researcher's report before writing or revising plan/todo content.
3. **LaTeX writing tasks** (recommended): consult researcher if the task involves understanding code behavior (e.g. writing a section that describes how training works). Skip if the task is pure prose editing with no code dependency.

Researcher's role in this process is advisory only: it reads, summarizes, and suggests — it does NOT write or edit `docs/plan.md`, `docs/todo.md`, or any other file beyond its own edit scope (`docs/ingest_knowledge.html`). Team-lead is solely responsible for writing the final plan/todo content based on researcher's report.

Team-lead must NOT write or update `docs/plan.md`, `docs/todo.md`, or any improvement plan for the user until researcher has reported. Exception: trivial edits (fixing a typo, updating a date, marking a task done) do not require researcher consultation.

---

## RULE 107 — Check issue.md before starting any task

`docs/user/issue.md` is maintained by the user and contains issues found in the current plan or implementation. Before starting any task:

1. Read `docs/user/issue.md` in full.
2. If any unresolved issues are relevant to the task, address them first — they take priority over the planned task.
3. Do NOT delete or mark an issue as resolved without explicit user confirmation. Ask the user: "Issue X appears solved — should I remove it?"
4. If the file is empty or no issues are relevant, proceed with the task normally.

This check is required before every new task, not just at session start.

## Project overview

ECE228-PINN is a PyTorch implementation of physics-informed neural networks (PINNs) for incompressible laminar flow around a cylinder, covering both steady and transient regimes. The project follows the mixed-form PINN setup from Rao, Sun, Liu — *Physics-informed deep learning for incompressible laminar flows* (arXiv:2002.10558). The deliverable is a course report for UCSD ECE 228.

Repository structure is not finalized — inspect the working tree before assuming any code layout.

## Submodule layout
- `report/` — LaTeX report manuscript (course deliverable)
- `docs/` — all project documentation (see below)
- Code, data, and result directories exist but their layout is in flux — re-check before referencing paths.

## Key terminology
- **PINN** — physics-informed neural network; loss combines data, PDE residual, and boundary/initial-condition terms
- **Mixed-form PINN** — the Rao/Sun/Liu formulation that introduces auxiliary stress variables to reduce derivative order
- **Steady** — time-independent flow regime (no temporal dimension in inputs)
- **Transient / unsteady** — time-dependent flow regime (inputs include time `t`)
- **Residual loss** — PDE-residual term of the PINN loss (continuity + Navier–Stokes)
- **Reference CFD** — high-fidelity numerical solution used as ground truth for steady-flow comparison

## Tech stack
- Python 3, PyTorch, NumPy, Matplotlib
- Device selection: `--device {auto,cpu,cuda,mps}` (Apple Silicon = `mps`, NVIDIA = `cuda`)
- Checkpoints: PyTorch `.pt` files with model/optimizer/scheduler state, loss history, resume metadata
- Report: LaTeX

## docs/ index
- `docs/plan.md` — research plan with goals, deliverables, and major risks
- `docs/todo.md` — actionable task list (immediate / blocked / deferred / open questions)
- `docs/user/todo.md` — **items only the user can do** (figures, data lookup, decisions)
- `docs/user/pending_decisions.md` — **decisions and approvals waiting on the user** (delete entries when resolved)
- `docs/user/issue.md` — **issues found by the user** (see RULE 107)

## docs/plan.md format

```
# ECE228-PINN Research Plan
Working title / Target venue / Last updated

## Phase N — Title
**Goal:** one sentence
**Tasks:**
- bullet list
**Deliverable:** one sentence

## Major Risks
### Risk N — Title
**Description:** ...
**Mitigation:** ...
**Status:** ...
```

Rules:
- One `## Phase N` per research phase, numbered sequentially.
- Do not add a "current status summary" block — status lives in `docs/todo.md`.
- Update `Last updated` when adding or retiring phases.
- Do not delete phases; mark completed ones with `*(complete)*` after the title.

## docs/todo.md format

```
# ECE228-PINN TODO
Last updated / priority note

## Immediate (can start now)
### Top priority
- [ ] item (date)
- [x] item — note (date)

### Writing / Implementation / Housekeeping
- [ ] item (Phase N ref)

## User-only TODO
- [ ] item

## Pending collaborator input
### <Collaborator name> [BLOCKED]
- [ ] item

## Deferred until framework stabilizes
- [ ] item [BLOCKED on X]

## Open questions to resolve
- question? (owner)
```

Rules:
- Use `- [x]` for completed items; remove them on the next cleanup pass (they are not archive — delete them).
- Mark blocked items with `**[BLOCKED]**` inline or `[BLOCKED on X]` at end.
- Dates go in parens at end of line: `(2026-05-20)`.
- Phase references go in parens: `(Phase 5)`.
- `Last updated` must be bumped on every edit.
- User-only items belong in `docs/user/todo.md`, not here — keep this file agent-actionable items only.

## RULE 108 — REFERENCE FIGURE STYLE

All figures under `data/reference/figures/` must follow the established style. Do not change colormaps, colorbar limits, labels, or layout without explicit user instruction.

**Standard style** (derived from `data/reference/figures/re100/`):
- Colormap: `rainbow` for all fields (u, v, p)
- Rendering: `pcolormesh` on the full NX×NY grid; cylinder masked with a white `Circle` patch
- Colorbar limits: u → [0.0, 1.0], v → [-0.5, 0.5], p → [-0.2, 3.0]
- Subplot titles: `"u reference (Re=…)"`, `"v reference (Re=…)"`, `"p (φ) reference (Re=…)"` — always use φ (phi), not "g" or "phi"
- Suptitle format: `"Re=… reference  t = X.XXX s"`
- Figure size: `(6, 8)` portrait, 3 vertical subplots
- DPI: 120 for frames, 150 for static figures
- GIF: 120 ms/frame, loop=0

**Generation script:** `data/reference/gen_figures_re100_t2.py` is the canonical template. Copy and adapt it for any new reference dataset — do not inline figure code in the data generator itself.

---

## RunPod GPU Pod Setup

When provisioning a fresh RunPod pod for training, apply these fixes before starting any Python job.

### 1. File transfer (rsync not available)

RunPod pods do not have `rsync`. Use `tar + scp` instead:

```bash
# local → remote
tar --exclude='.git' --exclude='ref' --exclude='results/unsteady/figures' \
    --exclude='__pycache__' --exclude='*.pyc' \
    -czf /tmp/upload.tar.gz .
scp -P <port> -i ~/.ssh/id_ed25519 /tmp/upload.tar.gz root@<host>:/workspace/upload.tar.gz
ssh root@<host> -p <port> -i ~/.ssh/id_ed25519 \
    "mkdir -p /workspace/ECE228_PINN && tar xzf /workspace/upload.tar.gz -C /workspace/ECE228_PINN/"
```

### 2. Python binary

RunPod pods often have two Python installs. The pip at `/usr/local/bin/pip` (or `pip3`) installs to `/usr/local/lib/python3.11/dist-packages`, which is on the sys.path of `/usr/bin/python` (the correct one). Always use `/usr/bin/python` or `python3` — verify with:

```bash
python3 -c "import sys; print(sys.path)"
# should include /usr/local/lib/python3.11/dist-packages
```

Install dependencies with:

```bash
pip install torch numpy scipy matplotlib pyDOE
```

### 3. pyDOE case-sensitivity fix (Linux only)

On Linux, `pip install pyDOE` installs the package directory as lowercase `pydoe/`, but the code does `from pyDOE import lhs`. Fix with a symlink:

```bash
ln -sf /usr/local/lib/python3.11/dist-packages/pydoe \
        /usr/local/lib/python3.11/dist-packages/pyDOE
python3 -c "from pyDOE import lhs; print('ok')"
```

### 4. PyTorch version for RTX 5090 (Blackwell)

The default PyTorch on RunPod pods (~2.4) does not support the RTX 5090 (Blackwell, sm_100). Upgrade before running:

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
# installs PyTorch 2.11+ with CUDA 12.8 — required for Blackwell
python3 -c "import torch; torch.tensor([1.]).cuda(); print('ok')"
```

RTX A6000 (Ampere) and RTX 4090 (Ada) work with the default PyTorch and do not need this step.

---

## CodeGraph usage

This repository has CodeGraph initialized. When answering architecture, call-flow, symbol-location, or impact-analysis questions, prefer CodeGraph tools before broad grep/read exploration.

Use:
- codegraph_search for finding symbols
- codegraph_callers / codegraph_callees for call flow
- codegraph_impact before editing a function/module
- codegraph_files for indexed file structure

For broad exploration, use a dedicated Explore agent and instruct it to use codegraph_explore as the primary tool. Do not blindly scan the whole repository with grep/read unless CodeGraph lacks results.