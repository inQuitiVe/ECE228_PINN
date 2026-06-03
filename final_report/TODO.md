# Final Report Completion TODO List

This list tracks the remaining tasks to move the report from the current template/migrated state to a submission-ready document.

## 1. Content Writing (LaTeX Sections)
- [ ] **Abstract**: Write ~150 words summarizing the problem, PINN formulation (mixed-variable), Re=10 results, and causality findings.
- [ ] **Introduction**:
    - [ ] Motivate PINNs for fluid dynamics.
    - [ ] Add citations for Raissi (2019) and Karniadakis (2021).
    - [ ] State specific contributions (reproduction accuracy, GPU-resident benchmark, causal extension).
- [ ] **Related Work**: Expand the current list into a cohesive narrative if needed.
- [ ] **Method / Approach**:
    - [ ] **Problem Formulation**: Formally define the 2D NS equations, domain geometry, and Re=10 parameters.
    - [ ] Verify if $\beta$ weights (current: 5) should match Rao's paper (2) or if our choice is preferred.
- [ ] **Results & Analysis**:
    - [ ] Add text analysis for the comparison between `reproduce` and `GPU-resident` backends.
    - [ ] Add analysis for the Causal training results (Frozen vs Uniform polish).
- [ ] **Conclusion**:
    - [ ] Summarize Re=10 fidelity.
    - [ ] Discuss gaps vs Rao's original results.
    - [ ] Suggest future work (Re=100, architectural ablations).

## 2. Figures & Visuals
- [ ] **Field Comparison**: Insert the rainbow colormap figure comparing $u, v, p$ at $t=0.3, 0.4, 0.5$s.
- [ ] **Causal Weight Profiles**: Insert the `causal_weight_curve.png` showing $\min w_i$ vs iteration.
- [ ] **Backend Benchmark**: (Already inserted) Verify caption and labels.

## 3. Tables & Data
- [ ] **Iso-budget Causal Ablation**: Conduct and add data for $\epsilon \in \{0, 2, 5\}$ to isolate the causal effect from the training schedule. Note: $\epsilon=0$ serves as the standard PINN baseline under the matched 20k-iteration schedule.

## 4. Administrative & Final Polish
- [ ] **Code & GitHub**: Add the actual repository link.
- [ ] **Team Contribution**: Write specific roles for Chi-Han Chiu, Chen-Yan Juang, Ming-Yang Wu, and Jing-Hua Chang.
- [ ] **Appendix**: Insert screenshot of the Course Evaluation submission for the 5-point bonus.
- [ ] **Citations**: Final check of `reference.bib` entries for completeness (DOIs, page numbers).
- [ ] **Formatting**: Check for Overfull/Underfull boxes and fix LaTeX warnings.
- [ ] **Final PDF**: Generate the definitive version.
