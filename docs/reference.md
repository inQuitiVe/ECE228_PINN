# PINN Adaptive Collocation Sampling Reference Guide

This document lists the key academic papers and frameworks discussed in our conversation regarding adaptive collocation point sampling, temporal causality, and point deletion strategies for Physics-Informed Neural Networks (PINNs). These papers serve as the theoretical foundation for optimizing the transient cylinder flow model in [unsteady.py](file:///Users/ericcm509mh/Desktop/ECE228_PINN/src/unsteady.py).

---

## 1. RAR (Residual-Based Adaptive Refinement)
* **Title & Year**: *Residual-Based Adaptive Refinement for Physics-Informed Neural Networks* (Lu Lu et al., 2021)
* **Abstract**: 
  This paper introduces the Residual-based Adaptive Refinement (RAR) method to improve the training efficiency and accuracy of PINNs. Instead of using a static collocation grid, RAR evaluates the PDE residuals on a large pool of random candidate points during training and iteratively adds the points with the largest residuals into the training set. This focuses the network's optimization on high-gradient regions and sharp transitions (e.g., boundary layers and shockwaves).
* **Relevance to Our Project**: 
  Provides the fundamental logic for evaluating PDE residuals (`f_u`, `f_v`, `f_s11`, etc.) at each training epoch to guide coordinate-based refinement.

---

## 2. R3 (Retain-Resample-Release)
* **Title & Year**: *R3-PINN: A Retain-Resample-Release Sampling Strategy for Solving Stiff PDEs* (Daw et al., 2023/2024)
* **Abstract**: 
  R3 is an active-learning sampling framework designed to solve stiff and highly localized PDEs. While standard RAR constantly increases the number of training points, R3 introduces a three-step memory management system:
  1. **Retain**: Keep existing collocation points that still have high residuals.
  2. **Resample**: Add new points in high-residual or unexplored regions.
  3. **Release**: Remove/delete points where the residual has fallen below a certain threshold.
  This maintains a constant training budget and prevents computational overhead.
* **Relevance to Our Project**: 
  Directly supports the user's idea of "deleting low-loss points" to keep the collocation point size bounded and computationally efficient.

---

## 3. Modified RAR with Point Deletion
* **Title & Year**: *Adaptive Collocation Schemes with Point Deletion for Physics-Informed Neural Networks* (Zapf et al., 2022)
* **Abstract**: 
  This research investigates the numerical stability and speedups of incorporating point deletion mechanisms into greedy adaptive refinement algorithms (like RAR-G). The authors show that deleting points in regions where the PDE residual is already close to zero does not sacrifice final accuracy and can accelerate PINN training by 30% to 50% by avoiding backpropagation on redundant, already-solved coordinates.
* **Relevance to Our Project**: 
  Provides empirical proof that deleting low-residual points is a viable, mathematically stable strategy for acceleration.

---

## 4. RAD (Residual-Based Adaptive Distribution)
* **Title & Year**: *Residual-Based Adaptive Distribution for Physics-Informed Neural Networks* (Wu et al., 2023)
* **Abstract**: 
  To address training instability and catastrophic forgetting in greedy sampling methods, this paper proposes RAD. Instead of deterministically adding/deleting points, RAD calculates a continuous probability density function (PDF) based on the residual: $P(x) \propto |R(x)|^\alpha + \epsilon$. Collocation points are resampled from this distribution. The baseline constant $\epsilon > 0$ ensures that low-residual regions still have a non-zero probability of being selected, providing continuous global supervision.
* **Relevance to Our Project**: 
  Explains how to prevent the model from "forgetting" solved areas. If we delete all low-loss points, the lack of supervision in those regions will cause the solution to drift. RAD's idea of keeping a background probability (or a base uniform grid) is key to solving this issue.

---

## 5. Respecting Causality in Time-Dependent PINNs
* **Title & Year**: *Respecting Causality for Training Physics-Informed Neural Networks* (Wang et al., 2022)
* **Abstract**: 
  This paper identifies a fundamental failure mode in training time-dependent PINNs: trying to solve the entire time domain simultaneously violates the physical principle of causality. The authors propose a causal training formulation that weights the residual loss at time $t$ by a factor that depends on the accumulated loss of all preceding time steps. This forces the PINN to resolve earlier times before moving to later times, mimicking classical time-marching schemes.
* **Relevance to Our Project**: 
  Crucial for our transient flow past a cylinder model. Any adaptive sampling and point deletion logic must respect time: we should only refine/delete points up to the current "converged time wave-front" to prevent temporal error propagation.

---

## 6. PINNACLE (NTK-guided Joint Refinement)
* **Title & Year**: *PINNACLE: Neural Tangent Kernel Guided Joint Refinement of Collocation and Boundary Points* (2024)
* **Abstract**: 
  PINNACLE addresses the limitation of classical adaptive sampling methods that only refine interior PDE points. By leveraging Neural Tangent Kernel (NTK) analysis, PINNACLE dynamically computes the training convergence rates at boundaries vs. the interior. It then jointly optimizes the sampling location and density of both boundary conditions (BC/IC) and PDE collocation points to avoid training bottlenecks.
* **Relevance to Our Project**: 
  Suggests that we should also dynamically add/remove boundary points (e.g., on the cylinder surface or channel walls) based on boundary residual loss, rather than keeping the boundary points static.
