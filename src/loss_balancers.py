"""Pluggable loss balancers for PINN training."""

TERM_KEYS = ["loss_f", "loss_ic", "loss_wall", "loss_inlet", "loss_outlet"]
ANCHOR = "loss_f"  # anchor term; its λ is always 1.0


class Balancer:
    def compute_total(self, losses: dict):
        raise NotImplementedError

    def update(self, losses: dict, params) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def log_lambdas(self) -> dict:
        return {}


class NoneBalancer(Balancer):
    """Fixed β weights matching vanilla unsteady.py (5/5/1/1)."""

    WEIGHTS = {"loss_f": 1.0, "loss_ic": 1.0, "loss_wall": 5.0,
               "loss_inlet": 5.0, "loss_outlet": 1.0}

    def compute_total(self, losses: dict):
        return sum(self.WEIGHTS[k] * losses[k] for k in TERM_KEYS)

    def log_lambdas(self) -> dict:
        return dict(self.WEIGHTS)


class GradNormBalancer(Balancer):
    """
    Gradient Norm Annealing — Wang, Teng, Perdikaris 2020 Algorithm 1.

    λ̂_i = max_j ‖∇_θ L_j‖ / ‖∇_θ L_i‖
    λ_i ← (1 − α)·λ_i + α·λ̂_i   (EMA)

    loss_f is the anchor (λ_f = 1 always); the four BC/IC terms are scaled.
    λ frozen during L-BFGS phase.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        # Warm-start from vanilla fixed weights (excluding anchor)
        self.lambdas = {"loss_ic": 1.0, "loss_wall": 5.0,
                        "loss_inlet": 5.0, "loss_outlet": 1.0}

    def compute_total(self, losses: dict):
        total = losses[ANCHOR]
        for k in TERM_KEYS:
            if k != ANCHOR:
                total = total + self.lambdas[k] * losses[k]
        return total

    def update(self, losses: dict, params) -> None:
        """
        Measure per-term gradient norms, then EMA-update λ_i.
        Uses retain_graph=True so the graph remains live for the actual step.
        Zeros grads after measurement.
        """
        grad_norms = {}
        for k in TERM_KEYS:
            for p in params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            losses[k].backward(retain_graph=True)
            norm_sq = sum(
                p.grad.detach().norm() ** 2
                for p in params
                if p.grad is not None
            )
            grad_norms[k] = float(norm_sq ** 0.5)

        # Zero grads — caller will call compute_total().backward() next
        for p in params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        max_norm = max(grad_norms.values())
        if max_norm < 1e-12:
            return

        for k in TERM_KEYS:
            if k == ANCHOR:
                continue
            norm_i = grad_norms[k]
            if norm_i < 1e-12:
                continue
            lhat = max_norm / norm_i
            lhat = min(max(lhat, 1e-2), 1e3)
            self.lambdas[k] = (1.0 - self.alpha) * self.lambdas[k] + self.alpha * lhat

    def state_dict(self) -> dict:
        return {"lambdas": dict(self.lambdas), "alpha": self.alpha}

    def load_state_dict(self, state: dict) -> None:
        self.lambdas.update(state.get("lambdas", {}))
        self.alpha = state.get("alpha", self.alpha)

    def log_lambdas(self) -> dict:
        return {ANCHOR: 1.0, **self.lambdas}


class StrictGradNormBalancer(Balancer):
    """
    Paper-faithful Wang, Teng, Perdikaris 2020 Algorithm 1, eq. (40-41).

    λ̂_i = max_θ{|∇_θ L_r(θ)|}  /  mean_θ(|∇_θ L_i(θ)|)
    λ_i ← (1 − α)·λ_i + α·λ̂_i

    Differences vs `GradNormBalancer`:
      - Numerator = max element of the RESIDUAL (anchor) gradient (scalar),
        NOT the L2 norm of whichever loss has the largest gradient.
      - Denominator = MEAN of |∇L_i| over all params (sensitive to gradient
        sparsity; BC terms with concentrated gradient get a small mean →
        large λ̂ — the desired behavior).
      - Default α = 0.9 (paper recommendation: fast adoption of λ̂).
      - Warm-start λ = 1.0 (paper convention).
      - λ frozen during L-BFGS (our extension; paper used Adam only).
    """

    def __init__(self, alpha: float = 0.9, clip_min: float = 1e-2, clip_max: float = 1e3):
        self.alpha = alpha
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.lambdas = {"loss_ic": 1.0, "loss_wall": 1.0,
                        "loss_inlet": 1.0, "loss_outlet": 1.0}
        # Saturation counters: how many times λ̂_i hit each clip boundary
        self.clip_hits = {k: {"min": 0, "max": 0} for k in self.lambdas}
        self.update_count = 0

    def compute_total(self, losses: dict):
        total = losses[ANCHOR]
        for k in TERM_KEYS:
            if k != ANCHOR:
                total = total + self.lambdas[k] * losses[k]
        return total

    def update(self, losses: dict, params) -> None:
        import torch as _torch
        flat_abs_grad = {}
        for k in TERM_KEYS:
            for p in params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            losses[k].backward(retain_graph=True)
            chunks = [p.grad.detach().abs().flatten()
                      for p in params if p.grad is not None]
            flat_abs_grad[k] = _torch.cat(chunks) if chunks else None

        for p in params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        anchor_grad = flat_abs_grad[ANCHOR]
        if anchor_grad is None or anchor_grad.numel() == 0:
            return
        num = float(anchor_grad.max().item())
        if num < 1e-12:
            return

        self.update_count += 1
        for k in TERM_KEYS:
            if k == ANCHOR:
                continue
            g = flat_abs_grad[k]
            if g is None or g.numel() == 0:
                continue
            denom = float(g.mean().item())
            if denom < 1e-12:
                continue
            lhat_raw = num / denom
            if lhat_raw <= self.clip_min:
                self.clip_hits[k]["min"] += 1
                lhat = self.clip_min
            elif lhat_raw >= self.clip_max:
                self.clip_hits[k]["max"] += 1
                lhat = self.clip_max
            else:
                lhat = lhat_raw
            self.lambdas[k] = (1.0 - self.alpha) * self.lambdas[k] + self.alpha * lhat

    def saturation_summary(self) -> dict:
        """Returns per-term clip-hit fractions; useful for end-of-run logging."""
        if self.update_count == 0:
            return {}
        return {k: {"min_frac": h["min"] / self.update_count,
                    "max_frac": h["max"] / self.update_count}
                for k, h in self.clip_hits.items()}

    def state_dict(self) -> dict:
        return {"lambdas": dict(self.lambdas), "alpha": self.alpha,
                "clip_min": self.clip_min, "clip_max": self.clip_max,
                "clip_hits": {k: dict(v) for k, v in self.clip_hits.items()},
                "update_count": self.update_count}

    def load_state_dict(self, state: dict) -> None:
        self.lambdas.update(state.get("lambdas", {}))
        self.alpha = state.get("alpha", self.alpha)
        self.clip_min = state.get("clip_min", self.clip_min)
        self.clip_max = state.get("clip_max", self.clip_max)
        if "clip_hits" in state:
            for k, v in state["clip_hits"].items():
                if k in self.clip_hits:
                    self.clip_hits[k].update(v)
        self.update_count = state.get("update_count", self.update_count)

    def log_lambdas(self) -> dict:
        return {ANCHOR: 1.0, **self.lambdas}


def make_balancer(name: str, **kwargs) -> Balancer:
    if name == "none":
        return NoneBalancer()
    if name == "grad_norm":
        return GradNormBalancer(**kwargs)
    if name == "strict_grad_norm":
        return StrictGradNormBalancer(**kwargs)
    raise ValueError(f"Unknown balancer {name!r}. Choices: none, grad_norm, strict_grad_norm")
