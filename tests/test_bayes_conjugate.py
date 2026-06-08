"""Tests for Bayesian conjugate prior inference functions."""

import numpy as np
import pytest

from statscore.bayes.conjugate import (
    bayes_normal_mean_known_var,
    bayes_normal_mean_unknown_var,
)


class TestBayesNormalMeanKnownVar:
    def test_posterior_hyperparameters(self):
        """Check posterior kappa_n and mu_n computation."""
        x = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        result = bayes_normal_mean_known_var(x, sigma_sq=4.0, mu0=0.0, kappa0=1.0)
        np.testing.assert_allclose(result.kappa_n, 6.0, rtol=1e-6)
        np.testing.assert_allclose(result.mu_n, (1.0 * 0.0 + 5 * 7.0) / 6.0, rtol=1e-6)

    def test_large_kappa0_dominated_by_prior(self):
        """With very large kappa0, posterior should be near the prior mean."""
        x = np.array([100.0, 200.0, 300.0])
        result = bayes_normal_mean_known_var(x, sigma_sq=1.0, mu0=0.0, kappa0=1e6)
        np.testing.assert_allclose(result.posterior_mean, 0.0, atol=1.0)

    def test_kappa0_small_approaches_mle(self):
        """With kappa0 near 0 (but positive), posterior mean approaches x_bar."""
        x = np.array([10.0, 12.0, 11.0, 13.0, 9.0])
        result = bayes_normal_mean_known_var(x, sigma_sq=2.0, mu0=0.0, kappa0=1e-10)
        np.testing.assert_allclose(result.posterior_mean, x.mean(), rtol=1e-4)

    def test_credible_interval_contains_true_value(self):
        """95% CI should contain the true mean for a typical sample."""
        np.random.seed(42)
        true_mu = 5.0
        x = np.random.normal(true_mu, 2.0, 100)
        result = bayes_normal_mean_known_var(x, sigma_sq=4.0, mu0=5.0, kappa0=1.0, alpha=0.05)
        lower, upper = result.credible_interval
        assert lower < true_mu < upper

    def test_predictive_variance_larger_than_posterior(self):
        """Predictive variance should always exceed posterior variance."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bayes_normal_mean_known_var(x, sigma_sq=1.0, mu0=3.0, kappa0=2.0)
        assert result.predictive_variance > result.posterior_variance

    def test_predictive_interval_wider_than_credible(self):
        """Predictive interval should be wider than credible interval."""
        x = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        result = bayes_normal_mean_known_var(x, sigma_sq=2.0, mu0=5.0, kappa0=1.0)
        ci_width = result.credible_interval[1] - result.credible_interval[0]
        pi_width = result.predictive_interval[1] - result.predictive_interval[0]
        assert pi_width > ci_width


class TestBayesNormalMeanUnknownVar:
    def test_posterior_hyperparameters(self):
        """Check posterior hyperparameter updates."""
        x = np.array([2.0, 4.0, 6.0, 8.0])
        result = bayes_normal_mean_unknown_var(x, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        n = 4
        x_bar = 5.0
        kappa_n = 1.0 + n
        mu_n = (1.0 * 0.0 + n * x_bar) / kappa_n
        alpha_n = 1.0 + n / 2.0
        ss = float(np.sum((x - x_bar) ** 2))
        beta_n = 1.0 + 0.5 * ss + (1.0 * n * (x_bar - 0.0) ** 2) / (2.0 * kappa_n)
        np.testing.assert_allclose(result.kappa_n, kappa_n, rtol=1e-6)
        np.testing.assert_allclose(result.mu_n, mu_n, rtol=1e-6)
        np.testing.assert_allclose(result.alpha_n, alpha_n, rtol=1e-6)
        np.testing.assert_allclose(result.beta_n, beta_n, rtol=1e-6)

    def test_posterior_mean_precision(self):
        """E[tau] = alpha_n / beta_n."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bayes_normal_mean_unknown_var(x, mu0=3.0, kappa0=1.0, alpha0=2.0, beta0=2.0)
        np.testing.assert_allclose(
            result.posterior_mean_precision, result.alpha_n / result.beta_n, rtol=1e-6
        )

    def test_posterior_mean_variance(self):
        """E[sigma^2] = beta_n / (alpha_n - 1) when alpha_n > 1."""
        x = np.array([5.0, 5.5, 4.5, 6.0, 4.0])
        result = bayes_normal_mean_unknown_var(x, mu0=5.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        assert result.posterior_mean_variance is not None
        np.testing.assert_allclose(
            result.posterior_mean_variance, result.beta_n / (result.alpha_n - 1), rtol=1e-6
        )

    def test_mu_credible_interval_contains_sample_mean(self):
        """With a diffuse prior, CI should contain sample mean."""
        np.random.seed(123)
        x = np.random.normal(10, 2, 50)
        result = bayes_normal_mean_unknown_var(
            x, mu0=10.0, kappa0=0.01, alpha0=0.01, beta0=0.01, alpha=0.05
        )
        lower, upper = result.mu_credible_interval
        assert lower < x.mean() < upper

    def test_variance_credible_interval_positive(self):
        """Variance CI bounds should be positive."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = bayes_normal_mean_unknown_var(x, mu0=4.0, kappa0=1.0, alpha0=2.0, beta0=2.0)
        lower, upper = result.variance_credible_interval
        assert lower > 0
        assert upper > lower

    def test_summary_prints(self, capsys):
        """summary() method should produce output."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bayes_normal_mean_unknown_var(x, mu0=3.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        result.summary()
        captured = capsys.readouterr()
        assert "Posterior Hyperparameters" in captured.out
        assert "mu_n" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
