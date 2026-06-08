"""Result dataclasses for MCMC-based Bayesian inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.figure import Figure


@dataclass
class MCMCResult:
    """Result of a Metropolis-Hastings MCMC run."""

    chain: np.ndarray               # full chain including burn-in: shape (n_iter,)
    param_names: list[str]          # parameter names
    n_iter: int
    n_burnin: int
    acceptance_rate: float
    posterior_samples: np.ndarray   # post-burnin samples
    posterior_mean: np.ndarray      # per-parameter
    posterior_std: np.ndarray
    credible_intervals: list[tuple[float, float]]
    alpha: float
    model_name: str

    def summary(self) -> None:
        pct = int((1 - self.alpha) * 100)
        w = 72
        print("=" * w)
        print(f"  MCMC Posterior Summary — {self.model_name}")
        print("=" * w)
        print(f"  Iterations: {self.n_iter}   Burn-in: {self.n_burnin}   "
              f"Effective samples: {len(self.posterior_samples)}")
        print(f"  Acceptance rate: {self.acceptance_rate:.3f}")
        print("-" * w)
        print(f"  {'Parameter':<18} {'Mean':>10} {'Std':>10} "
              f"  {pct}% CI Lower  {pct}% CI Upper")
        print("  " + "-" * (w - 4))
        for i, name in enumerate(self.param_names):
            lo, hi = self.credible_intervals[i]
            print(f"  {name:<18} {self.posterior_mean[i]:>10.5f} "
                  f"{self.posterior_std[i]:>10.5f}   {lo:>11.5f}   {hi:>11.5f}")
        print("=" * w)

    def plot(self) -> Figure:
        from matplotlib import pyplot as plt
        from scipy.stats import gaussian_kde

        p = len(self.param_names)
        fig, axes = plt.subplots(p, 2, figsize=(12, 3.5 * p))
        if p == 1:
            axes = axes.reshape(1, 2)

        samples = self.posterior_samples
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)

        full = self.chain
        if full.ndim == 1:
            full = full.reshape(-1, 1)

        for i, name in enumerate(self.param_names):
            ax_trace = axes[i, 0]
            ax_hist = axes[i, 1]

            ax_trace.plot(full[:, i], color="steelblue", linewidth=0.6, alpha=0.8)
            ax_trace.axvline(self.n_burnin, color="crimson", linestyle="--",
                             linewidth=1.2, label="End of burn-in")
            ax_trace.set_title(f"Trace — {name}")
            ax_trace.set_xlabel("Iteration")
            ax_trace.set_ylabel(name)
            ax_trace.legend(fontsize=8)
            ax_trace.grid(True, alpha=0.3)

            s = samples[:, i]
            try:
                kde = gaussian_kde(s)
                x_grid = np.linspace(s.min(), s.max(), 300)
                ax_hist.plot(x_grid, kde(x_grid), color="steelblue", linewidth=2)
                lo, hi = self.credible_intervals[i]
                mask = (x_grid >= lo) & (x_grid <= hi)
                ax_hist.fill_between(x_grid, kde(x_grid), where=mask,
                                     alpha=0.35, color="steelblue",
                                     label=f"{int((1-self.alpha)*100)}% CI")
            except Exception:
                ax_hist.hist(s, bins=40, density=True, color="steelblue", alpha=0.7)
            ax_hist.axvline(self.posterior_mean[i], color="crimson", linewidth=1.5,
                            linestyle="-", label=f"Mean={self.posterior_mean[i]:.4f}")
            ax_hist.set_title(f"Posterior — {name}")
            ax_hist.set_xlabel(name)
            ax_hist.set_ylabel("Density")
            ax_hist.legend(fontsize=8)
            ax_hist.grid(True, alpha=0.3)

        fig.suptitle(f"MCMC Diagnostics — {self.model_name}", fontsize=13)
        fig.tight_layout()
        return fig


@dataclass
class ConjugateModelResult:
    """Result of a conjugate Bayesian model (Beta-Binomial or Gamma-Poisson)."""

    model_name: str
    prior_params: dict[str, float]
    posterior_params: dict[str, float]
    posterior_mean: float
    posterior_variance: float
    posterior_std: float
    credible_interval: tuple[float, float]
    n: int
    alpha: float
    data_summary: dict[str, float] = field(default_factory=dict)

    def summary(self) -> None:
        pct = int((1 - self.alpha) * 100)
        w = 60
        print("=" * w)
        print(f"  Conjugate Bayesian Model — {self.model_name}")
        print("=" * w)
        print(f"  n = {self.n}", end="")
        for k, v in self.data_summary.items():
            print(f"   {k} = {v:.4g}", end="")
        print()
        print("-" * w)
        print("  Prior parameters:")
        for k, v in self.prior_params.items():
            print(f"    {k} = {v:.6g}")
        print("  Posterior parameters:")
        for k, v in self.posterior_params.items():
            print(f"    {k} = {v:.6g}")
        print("-" * w)
        print(f"  Posterior mean:    {self.posterior_mean:.6f}")
        print(f"  Posterior std:     {self.posterior_std:.6f}")
        print(f"  {pct}% Credible interval: "
              f"({self.credible_interval[0]:.6f}, {self.credible_interval[1]:.6f})")
        print("=" * w)

    def plot(self) -> Figure:
        from matplotlib import pyplot as plt
        from scipy import stats

        fig, ax = plt.subplots(figsize=(8, 5))
        pm = self.posterior_mean
        ps = self.posterior_std
        x = np.linspace(max(0, pm - 5 * ps), pm + 5 * ps, 500)

        if self.model_name == "Beta-Binomial":
            a0 = self.prior_params["alpha0"]
            b0 = self.prior_params["beta0"]
            an = self.posterior_params["alpha_n"]
            bn = self.posterior_params["beta_n"]
            prior_pdf = stats.beta.pdf(x, a0, b0)
            post_pdf = stats.beta.pdf(x, an, bn)
        elif self.model_name == "Gamma-Poisson":
            a0 = self.prior_params["alpha0"]
            b0 = self.prior_params["beta0"]
            an = self.posterior_params["alpha_n"]
            bn = self.posterior_params["beta_n"]
            prior_pdf = stats.gamma.pdf(x, a0, scale=1.0 / b0)
            post_pdf = stats.gamma.pdf(x, an, scale=1.0 / bn)
        else:
            ax.text(0.5, 0.5, "Plot not available for this model.",
                    ha="center", va="center", transform=ax.transAxes)
            fig.tight_layout()
            return fig

        ax.plot(x, prior_pdf, color="gray", linestyle="--", linewidth=1.5, label="Prior")
        ax.plot(x, post_pdf, color="steelblue", linewidth=2, label="Posterior")
        lo, hi = self.credible_interval
        ci_mask = ((x >= lo) & (x <= hi)).tolist()
        ax.fill_between(x, post_pdf, where=ci_mask, alpha=0.3, color="steelblue",
                        label=f"{int((1-self.alpha)*100)}% CI")
        ax.axvline(pm, color="crimson", linewidth=1.5, linestyle="-",
                   label=f"Post. mean = {pm:.4f}")
        ax.set_xlabel("θ")
        ax.set_ylabel("Density")
        ax.set_title(f"Prior vs Posterior — {self.model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
