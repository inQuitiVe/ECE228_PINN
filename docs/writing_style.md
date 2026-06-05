# Writing Style Guide

## Format
- IEEE two-column conference paper
- LaTeX class: `\documentclass[conference]{IEEEtran}`
- No `\IEEEpeerreviewmaketitle`

## Style (follow Rao et al. 2020, arXiv:2002.10558)

### Tone and voice
- Third person, passive where appropriate ("the network is trained", "results are reported")
- Concise and technical; avoid marketing language
- Define every acronym on first use: PINN, MLP, L-BFGS, etc.

### Math notation (follow Rao 2020)
- Flow variables: $u$, $v$ (velocity components), $p$ (pressure), $\psi$ (stream function)
- Auxiliary stress: $\sigma_{11}$, $\sigma_{22}$, $\sigma_{12}$
- Residual loss terms: $\mathcal{L}_f$, $\mathcal{L}_{ic}$, $\mathcal{L}_{wall}$, $\mathcal{L}_{inlet}$, $\mathcal{L}_{outlet}$
- Total loss: $\mathcal{L} = \mathcal{L}_f + \beta_{wall}\mathcal{L}_{wall} + \beta_{inlet}\mathcal{L}_{inlet} + \mathcal{L}_{outlet} + \mathcal{L}_{ic}$
- L2 relative error: $\varepsilon = \|u_{pred} - u_{ref}\|_F / \|u_{ref}\|_F$
- Network: $\mathcal{N}_\theta : \mathbb{R}^3 \to \mathbb{R}^5$, input $(x, y, t)$, output $(\psi, p, \sigma_{11}, \sigma_{22}, \sigma_{12})$

### Figures
- All figures referenced as "Fig.~\ref{fig:...}"
- Captions below figures, concise (one sentence preferred)
- Field plots: rainbow colormap, cylinder masked; follow RULE 108 style

### Tables
- All tables referenced as "Table~\ref{tab:...}"
- Captions above tables
- Use `\toprule`, `\midrule`, `\bottomrule` (booktabs)
- L2 errors in percent with 2 decimal places

### Numbers and units
- Percentages: "5.30\%" not "5.3\%"
- Scientific notation: $2.10 \times 10^{-4}$ not "2.10e-4"
- Reynolds number: $Re = 10$, $Re = 100$

### Citations
- Use `\cite{key}` inline; never invent keys (RULE 101)
- Place citation after the claim, before the period: "...as shown in prior work~\cite{rao2020}."
- For equations from the paper: "following~\cite{rao2020},"

### \todo{} convention
- Missing citation: `\todo{cite: description}`
- Missing number/result: `\todo{result: description}`
- Missing figure: `\todo{fig: description}`
- User decision needed: `\todo{USER: question}`
