# Architecture Reference

## Dependency Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│                      statscore (public API)                          │
│                      top-level __init__.py                           │
├──────────────────────────────────────────────────────────────────────┤
│                  statscore.methods (domain layer)                    │
│   ├── methods.anova         methods.regression                       │
│   ├── methods.bayes         methods.testing                          │
│   └── methods.diagnostics                                            │
├──────────────────────────────────────────────────────────────────────┤
│         statscore.plots   statscore.io   statscore.cli               │
│         statscore.app                                                │
├──────────────────────────────────────────────────────────────────────┤
│                       statscore.utils                                │
│   ├── enums.py             (type definitions)                        │
│   ├── distributions.py     (scipy wrappers for critical values)      │
│   └── validation.py        (shared input guards)                     │
└──────────────────────────────────────────────────────────────────────┘
```

### Layer Rules

| Layer | May import from | Must NOT import from |
|-------|----------------|----------------------|
| `utils` | External packages only (numpy, scipy) | Any `statscore.*` module |
| `methods.anova` | `statscore.utils`, other `methods.anova` submodules | Any other domain module; `plots`; `io` |
| `methods.regression` | `statscore.utils`, other `methods.regression` submodules | Any other domain module; `plots`; `io` |
| `methods.testing` | `statscore.utils` | All other domain modules |
| `methods.bayes` | `statscore.utils` | All other domain modules |
| `methods.diagnostics` | `statscore.utils`, `methods.regression.least_squares` | `methods.anova`; `methods.testing`; `methods.bayes`; `plots`; `io` |
| `plots` | External packages only (numpy, scipy, matplotlib) | All statscore domain modules |
| `io` | External packages only (pandas) | All statscore domain modules |
| `cli` | `statscore.methods.*`, `statscore.plots`, `statscore.io` | `statscore.utils` internals |
| `app` | `statscore.methods.*`, `statscore.plots`, `statscore.io` | `statscore.utils` internals |
| top-level `__init__` | `statscore.methods.*`, `statscore.plots`, `statscore.io`, `statscore.utils.enums` | — |

No circular dependencies exist. The dependency graph is a strict DAG.

**Note on lazy imports in `plot()` methods:** Result dataclasses call `from statscore.plots import …` lazily inside their `.plot()` method body. This is an allowed exception — it avoids a circular import at module load time while keeping the dependency direction correct (domain → plots is a downward call, not upward).

### Dependency DAG (key edges)

```
utils.enums         ← methods.anova.*, methods.regression.*, methods.testing.*, utils.*
utils.distributions ← methods.anova.*, methods.regression.*, methods.testing.*, methods.bayes.conjugate
utils.validation    ← methods.anova.*, methods.regression.*, methods.testing.*, methods.bayes.*, methods.diagnostics

methods.anova.one_way      ← methods.anova.multiple_tests  (intra-layer)
methods.regression.ls      ← methods.regression.inference, .prediction, .summary  (intra-layer)
                           ← methods.diagnostics  (cross-layer, read-only OLS call)

plots              ← result dataclasses (lazy, inside .plot() method bodies only)
cli                ← methods.*, plots, io  (runtime imports inside handler functions)
app                ← methods.*, plots, io  (runtime imports inside page functions)
```

---

## Package Structure

```
statscore/
├── __init__.py              # Public API — ~52 symbols, __version__ = "0.0.3"
├── __main__.py              # python -m statscore → cli.main()
│
├── app/
│   └── __init__.py          # Streamlit browser UI — 6 pages; ~1350 lines
│
├── plots/
│   └── __init__.py          # Shared plot utilities — 8 functions returning Figure
│
├── io/
│   └── __init__.py          # load_data(path, **kwargs) → LoadedData
│
├── cli/
│   ├── __init__.py          # main(), _MENU_HANDLERS dict, _print_menu() — 21 items
│   ├── _anova.py            # _run_one_way_anova, _run_two_way_anova,
│   │                        # _run_anova_multiple_comparisons, _run_anova_multiple_comparisons_with
│   ├── _testing.py          # _run_z_test, _run_one/two_sample_t_test, _run_paired_t_test,
│   │                        # _run_chi2_test, _run_f_test, _run_normality_check,
│   │                        # _run_levene_check, _run_mean_ci
│   ├── _regression.py       # _run_simple_regression, _run_multiple_regression,
│   │                        # _run_regression_diagnostics, _run_bayes_known_var,
│   │                        # _run_bayes_unknown_var, _run_bayes_beta_binomial,
│   │                        # _run_bayes_gamma_poisson, _run_mcmc_normal, _run_mcmc_regression
│   └── _io.py               # _parse_data_input, _parse_matrix_input,
│                            # _parse_raw_string, _parse_groups_input
│
├── methods/
│   ├── __init__.py
│   │
│   ├── anova/
│   │   ├── __init__.py      # Re-exports all anova symbols
│   │   ├── _results.py      # ANOVA1PartitionResult, ANOVA1TestResult, ANOVA2PartitionResult,
│   │   │                    # ANOVA2MLEResult, ANOVA2TestResult, SimultaneousCIResult,
│   │   │                    # SimultaneousTestResult
│   │   ├── one_way.py       # anova1_partition_tss, anova1_test_equality
│   │   ├── two_way.py       # anova2_partition_tss, anova2_mle, anova2_test_equality
│   │   └── multiple_tests.py  # anova1_is_contrast, anova1_is_orthogonal,
│   │                          # bonferroni_correction, sidak_correction,
│   │                          # anova1_ci_linear_combs, anova1_test_linear_combs
│   │
│   ├── bayes/
│   │   ├── __init__.py      # Re-exports all Bayesian symbols
│   │   ├── _results.py      # NormalMeanKnownVarResult, NormalMeanUnknownVarResult
│   │   ├── _mcmc_results.py # MCMCResult, ConjugateModelResult
│   │   ├── conjugate.py     # bayes_normal_mean_known_var, bayes_normal_mean_unknown_var
│   │   └── mcmc.py          # run_mcmc, mcmc_normal_mean_unknown_var, mcmc_linear_regression,
│   │                        # bayes_beta_binomial, bayes_gamma_poisson
│   │
│   ├── diagnostics/
│   │   ├── __init__.py      # shapiro_wilk_test, levene_test,
│   │   │                    # regression_diagnostics, mean_confidence_interval
│   │   └── _results.py      # ShapiroWilkResult, LeveneResult,
│   │                        # RegressionDiagnosticsResult, MeanConfidenceIntervalResult
│   │
│   ├── regression/
│   │   ├── __init__.py      # Re-exports all regression symbols
│   │   ├── _results.py      # OLSResult, PartitionTSSResult, SimultaneousCIBetaResult,
│   │   │                    # ConfidenceRegionResult, HypothesisTestResult,
│   │   │                    # PredictionCIResult, RegressionSummaryResult
│   │   ├── least_squares.py # mult_lr_least_squares, mult_lr_partition_tss
│   │   ├── inference.py     # mult_norm_lr_simul_ci, mult_norm_lr_cr,
│   │   │                    # mult_norm_lr_is_in_cr, mult_norm_lr_test_general,
│   │   │                    # mult_norm_lr_test_comp, mult_norm_lr_test_linear_reg
│   │   ├── prediction.py    # mult_norm_lr_pred_ci
│   │   └── summary.py       # regression_summary
│   │
│   └── testing/
│       ├── __init__.py      # Re-exports all testing symbols
│       ├── _results.py      # ZTestResult, TTestOneSampleResult, TTestTwoSampleResult,
│       │                    # TTestPairedResult, Chi2VarianceTestResult, FTestVariancesResult
│       ├── one_sample.py    # z_test_mean, t_test_mean, chi2_test_variance
│       └── two_sample.py    # t_test_two_sample, t_test_paired, f_test_variances
│
└── utils/
    ├── __init__.py
    ├── enums.py             # AlternativeHypothesis, CorrectionMethod,
    │                        # PredictionMethod, TwoWayTestFactor
    ├── distributions.py     # f_critical, f_pvalue, t_ppf, chi2_ppf, norm_ppf,
    │                        # q_ppf (Studentised range), z_pvalue, t_cdf
    └── validation.py        # validate_positive, validate_non_negative,
                             # validate_1d_sample, validate_alternative,
                             # validate_design_matrix, validate_data_groups
```

---

## Type System

### Enums

All categorical function parameters use enums defined in `statscore/utils/enums.py`:

| Enum | Members | Controls |
|------|---------|----------|
| `AlternativeHypothesis` | `TWO_SIDED`, `LESS`, `GREATER` | Alternative direction in all 6 tests |
| `CorrectionMethod` | `SCHEFFE`, `TUKEY`, `BONFERRONI`, `SIDAK`, `BEST` | Multiple comparison method selection |
| `PredictionMethod` | `SCHEFFE`, `BONFERRONI`, `BEST` | Prediction interval method |
| `TwoWayTestFactor` | `A`, `B`, `AB` | Two-way ANOVA test target |

Passing a raw string where an enum is expected raises `TypeError` immediately.

### Typing Rules

1. Every function has an explicit `-> ReturnType` annotation.
2. All parameters are typed — no bare `def f(data, alpha, method)`.
3. Collections use parameterized generics: `list[tuple[float, float]]`, `dict[str, float]`.
4. `T | None` (PEP 604) for optional fields; `Optional[T]` not used in new code.
5. No `Any` in the public API.
6. All return objects are `@dataclass` with fully-typed fields.

### Dataclass Output Contract

Every public function returns a typed dataclass. The pattern:

```python
@dataclass
class ZTestResult:
    z_statistic: float
    z_critical: float
    p_value: float
    reject_H0: bool
    alpha: float
    alternative: AlternativeHypothesis   # enum, not str
    n: int
    x_bar: float
    mu0: float
    sigma: float

    def summary(self) -> None: ...
    def plot(self) -> Figure: ...
```

Rules:
- Result dataclasses live in `_results.py` files, separate from the computation logic.
- Enum fields store the enum value, never a string.
- All result types expose `.summary()` (returns `None`, prints to stdout) and `.plot()` (returns `matplotlib.figure.Figure`).

### MCMCResult and ConjugateModelResult

MCMC results live in `methods/bayes/_mcmc_results.py` (separate from `_results.py`) because they are shared across `mcmc.py` functions that have no conjugate counterpart:

```python
@dataclass
class MCMCResult:
    chain: np.ndarray              # (n_iter, n_params) — includes burn-in
    param_names: list[str]
    n_iter: int
    n_burnin: int
    acceptance_rate: float
    posterior_samples: np.ndarray  # post-burn-in only
    posterior_mean: dict[str, float]
    posterior_std: dict[str, float]
    credible_intervals: dict[str, tuple[float, float]]
    alpha: float
    model_name: str

    def summary(self) -> None: ...
    def plot(self) -> Figure: ...   # trace + KDE panels per parameter
```

---

## Import Style

All internal imports use **absolute paths** rooted at `statscore`:

```python
# Correct
from statscore.methods.regression._results import PredictionCIResult
from statscore.methods.regression.least_squares import mult_lr_least_squares
from statscore.utils.distributions import f_critical
from statscore.utils.enums import CorrectionMethod, AlternativeHypothesis

# Wrong — do not use relative imports
from ..utils.distributions import f_critical
from .least_squares import mult_lr_least_squares
```

**CLI and app handlers** import their dependencies lazily (inside function bodies) to avoid importing heavy libraries (matplotlib, scipy) at startup and to prevent potential circular import chains:

```python
def _run_one_way_anova() -> None:
    from statscore.methods.anova import anova1_test_equality   # lazy OK
    from statscore.plots import plot_anova_groups              # lazy OK
    ...
```

---

## Public API Counts

Symbols exported through `statscore.__all__`:

| Category | Count | Symbols |
|----------|-------|---------|
| Enums | 4 | `AlternativeHypothesis`, `CorrectionMethod`, `PredictionMethod`, `TwoWayTestFactor` |
| ANOVA functions | 11 | `anova1_partition_tss`, `anova1_test_equality`, `anova1_is_contrast`, `anova1_is_orthogonal`, `bonferroni_correction`, `sidak_correction`, `anova1_ci_linear_combs`, `anova1_test_linear_combs`, `anova2_partition_tss`, `anova2_mle`, `anova2_test_equality` |
| Testing functions | 6 | `z_test_mean`, `t_test_mean`, `chi2_test_variance`, `t_test_two_sample`, `t_test_paired`, `f_test_variances` |
| Regression functions | 11 | `mult_lr_least_squares`, `mult_lr_partition_tss`, `regression_summary`, `mult_norm_lr_simul_ci`, `mult_norm_lr_cr`, `mult_norm_lr_is_in_cr`, `mult_norm_lr_test_general`, `mult_norm_lr_test_comp`, `mult_norm_lr_test_linear_reg`, `mult_norm_lr_pred_ci`, `RegressionSummaryResult` |
| Bayesian functions | 9 | `bayes_normal_mean_known_var`, `bayes_normal_mean_unknown_var`, `bayes_beta_binomial`, `bayes_gamma_poisson`, `run_mcmc`, `mcmc_normal_mean_unknown_var`, `mcmc_linear_regression`, `MCMCResult`, `ConjugateModelResult` |
| Diagnostic functions | 8 | `shapiro_wilk_test`, `levene_test`, `regression_diagnostics`, `mean_confidence_interval`, `ShapiroWilkResult`, `LeveneResult`, `RegressionDiagnosticsResult`, `MeanConfidenceIntervalResult` |
| Plot utilities | 8 | `plot_regression`, `plot_residuals`, `plot_qq`, `plot_anova_groups`, `plot_t_test`, `plot_f_test`, `plot_simultaneous_ci`, `plot_posterior_normal` |
| I/O | 2 | `load_data`, `LoadedData` |
| **Total** | **~59** | |

---

## Adding New Modules

Follow these steps to add a new statistical method:

1. **Choose the right layer.** Place computation under `statscore/methods/<domain>/`. If it is a new domain, create a new subpackage with `__init__.py` and `_results.py`.

2. **Define result types first.** Add dataclasses to `statscore/methods/<domain>/_results.py`. Every result must:
   - Be a `@dataclass` with fully-typed fields
   - Implement `summary(self) -> None`
   - Implement `plot(self) -> Figure` (import from `statscore.plots` lazily inside the method body)
   - Use enum types for categorical fields

3. **Write the computation module.** Import from `_results.py` and sibling modules directly. Never import from the package `__init__.py` (circular imports). Use only absolute paths.

4. **Respect layer rules.** Domain modules must not import from each other unless the dependency is explicitly justified and follows the DAG.

5. **Use shared utilities.** Use `statscore.utils.validation` for input checks, `statscore.utils.distributions` for critical values, and `statscore.utils.enums` for new categoricals. Do not define local `_validate_*` copies.

6. **Export through `__init__.py`.** Add to the subpackage `__init__.py` and its `__all__`. Then re-export from the top-level `statscore/__init__.py` if it belongs to the public API.

7. **Add CLI handler** in the appropriate `statscore/cli/_<domain>.py` file, register it in `_MENU_HANDLERS` in `statscore/cli/__init__.py`, and update `_print_menu()`.

8. **Add tests** in `tests/test_<domain>.py`. Cover: normal inputs, edge cases, error conditions, summary output format, plot return type.

9. **Update documentation:**
   - `CHANGELOG.md` — new entry under the current version
   - `ARCHITECTURE.md` — package tree, public API counts, DAG if new dependencies
   - `README.md` — feature table, Quick Start example, API reference table
   - `USER_GUIDE.md` — "When to use", worked example, output interpretation
