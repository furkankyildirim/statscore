# Import Architecture & Type System

## Dependency Layers

```
┌─────────────────────────────────────────────────────────────┐
│                   statscore (public API)                    │  ← top-level __init__.py
├─────────────────────────────────────────────────────────────┤
│              statscore.methods (domain layer)               │
│  ├── methods.anova        methods.regression                │
│  ├── methods.bayes        methods.testing                   │
│  └── methods.diagnostics                                    │
├─────────────────────────────────────────────────────────────┤
│        statscore.plots   statscore.io   statscore.cli       │  ← cross-cutting modules
├─────────────────────────────────────────────────────────────┤
│                      statscore.utils                        │  ← base layer
│   ├── enums.py          (type definitions)                  │
│   ├── distributions.py  (scipy wrappers)                    │
│   └── validation.py     (shared input guards)               │
└─────────────────────────────────────────────────────────────┘
```

### Layer Rules

| Layer | May import from | Must NOT import from |
|-------|----------------|---------------------|
| `utils` | External packages only (numpy, scipy) | any `statscore.*` domain module |
| `methods.anova` | `statscore.utils`, other `methods.anova` submodules | `methods.regression`, `methods.testing`, `methods.bayes`, `methods.diagnostics`, `plots`, `io` |
| `methods.regression` | `statscore.utils`, other `methods.regression` submodules | `methods.anova`, `methods.testing`, `methods.bayes`, `methods.diagnostics`, `plots`, `io` |
| `methods.testing` | `statscore.utils` | all other domain modules |
| `methods.bayes` | `statscore.utils` | all other domain modules |
| `methods.diagnostics` | `statscore.utils`, `methods.regression.least_squares` | `methods.anova`, `methods.testing`, `methods.bayes`, `plots`, `io` |
| `utils.plots` | External packages only (numpy, scipy, matplotlib) | all statscore domain modules |
| `io` | External packages only (pandas) | all statscore domain modules |
| `cli` | `statscore.methods.*`, `statscore.plots`, `statscore.io` | `statscore.utils` internals |
| top-level `__init__` | `statscore.methods.*`, `statscore.plots`, `statscore.io`, `statscore.utils.enums` | — |

No circular dependencies exist. The dependency graph is a strict DAG:

```
utils.enums         ← methods.anova.two_way, methods.anova.multiple_tests
                    ← methods.regression.prediction
                    ← methods.testing.one_sample, methods.testing.two_sample
                    ← utils.distributions, utils.validation
utils.distributions ← methods.anova.one_way, methods.anova.two_way, methods.anova.multiple_tests
                    ← methods.regression.least_squares, methods.regression.inference
                    ← methods.regression.prediction, methods.regression.summary
                    ← methods.testing.one_sample, methods.testing.two_sample
                    ← methods.bayes.conjugate
                    ← methods.diagnostics
utils.validation    ← methods.anova.one_way, methods.anova.two_way, methods.anova.multiple_tests
                    ← methods.regression.least_squares, methods.regression.inference
                    ← methods.regression.prediction
                    ← methods.testing.one_sample, methods.testing.two_sample
                    ← methods.bayes.conjugate
                    ← methods.diagnostics
methods.anova.one_way       ← methods.anova.multiple_tests (intra-layer)
methods.regression.least_squares ← methods.regression.inference, methods.regression.prediction
                                 ← methods.regression.summary (intra-layer)
                                 ← methods.diagnostics (cross-layer, read-only)
```

## Package Structure

```
statscore/
├── __init__.py              # Top-level public API — re-exports everything
├── __main__.py              # python -m statscore entry point
├── plots.py                 # Shared plot utilities (7 functions)
├── io/
│   └── __init__.py          # load_data → LoadedData
├── cli/
│   ├── __init__.py          # main() entry point
│   ├── _anova.py            # CLI handlers: ANOVA
│   ├── _testing.py          # CLI handlers: hypothesis tests + diagnostics
│   ├── _regression.py       # CLI handlers: regression + Bayesian
│   └── _io.py               # CLI helpers: data input parsing
├── methods/                 # All statistical domain logic lives here
│   ├── __init__.py
│   ├── anova/
│   │   ├── __init__.py      # Re-exports all anova symbols
│   │   ├── _results.py      # Result dataclasses (ANOVA1*, ANOVA2*, Simultaneous*)
│   │   ├── one_way.py       # anova1_partition_tss, anova1_test_equality
│   │   ├── two_way.py       # anova2_partition_tss, anova2_mle, anova2_test_equality
│   │   └── multiple_tests.py # Contrasts, corrections, simultaneous CIs/tests
│   ├── bayes/
│   │   ├── __init__.py      # Re-exports all bayes symbols
│   │   ├── _results.py      # NormalMeanKnownVarResult, NormalMeanUnknownVarResult
│   │   └── conjugate.py     # bayes_normal_mean_known_var, bayes_normal_mean_unknown_var
│   ├── diagnostics/
│   │   ├── __init__.py      # All diagnostic functions + re-exports
│   │   └── _results.py      # ShapiroWilkResult, LeveneResult, RegressionDiagnosticsResult, MeanConfidenceIntervalResult
│   ├── regression/
│   │   ├── __init__.py      # Re-exports all regression symbols
│   │   ├── _results.py      # SimultaneousCIBetaResult, ConfidenceRegionResult, HypothesisTestResult, PredictionCIResult
│   │   ├── least_squares.py # mult_lr_least_squares, mult_lr_partition_tss
│   │   ├── inference.py     # Simultaneous CI, CR, general/component/linear tests
│   │   ├── prediction.py    # mult_norm_lr_pred_ci
│   │   └── summary.py       # regression_summary → RegressionSummaryResult
│   └── testing/
│       ├── __init__.py      # Re-exports all testing symbols
│       ├── _results.py      # ZTestResult, TTest*, Chi2*, FTest* result dataclasses
│       ├── one_sample.py    # z_test_mean, t_test_mean, chi2_test_variance
│       └── two_sample.py    # t_test_two_sample, t_test_paired, f_test_variances
└── utils/
    ├── enums.py             # AlternativeHypothesis, CorrectionMethod, PredictionMethod, TwoWayTestFactor
    ├── distributions.py     # Critical values and p-values (F, t, chi2, z, q)
    ├── plots.py             # Internal plot helpers (re-exported via statscore.plots)
    └── validation.py        # Shared input validation helpers
```

## Import Style

All internal imports use **absolute paths** rooted at `statscore`:

```python
# Correct — within methods.regression submodules
from statscore.methods.regression._results import PredictionCIResult
from statscore.methods.regression.least_squares import mult_lr_least_squares
from statscore.utils.distributions import f_critical
from statscore.utils.enums import CorrectionMethod, AlternativeHypothesis

# Incorrect (do not use)
from ..utils.distributions import f_critical
from .least_squares import mult_lr_least_squares
from statscore.regression import ...   # old path, no longer valid
```

## Type System

### Enum-Based Categorical Parameters

All categorical function parameters use strongly-typed enums defined in `statscore/utils/enums.py`:

| Enum | Members | Controls |
|------|---------|----------|
| `AlternativeHypothesis` | `TWO_SIDED`, `LESS`, `GREATER` | Alternative direction in all tests |
| `CorrectionMethod` | `SCHEFFE`, `TUKEY`, `BONFERRONI`, `SIDAK`, `BEST` | Multiple comparison method |
| `PredictionMethod` | `SCHEFFE`, `BONFERRONI`, `BEST` | Prediction interval method |
| `TwoWayTestFactor` | `A`, `B`, `AB` | Two-way ANOVA test target |

Usage:
```python
from statscore import t_test_mean, AlternativeHypothesis

result = t_test_mean(x, mu0=0.0, alpha=0.05,
                     alternative=AlternativeHypothesis.TWO_SIDED)
```

Passing a raw string to an enum parameter raises `TypeError` with a clear message.

### Typing Rules

1. Every function has explicit `-> ReturnType` annotations.
2. All parameters are typed — no bare `def f(data, alpha, method)`.
3. Collections use parameterized types: `list[tuple[float, float]]`, `dict[str, dict[str, float]]`.
4. `T | None` (PEP 604 union syntax) for optional fields; `Optional[T]` no longer used in new code.
5. No `Any` in the public API.
6. All return objects are `@dataclass` with fully-typed fields.

### Dataclass Output Contract

Every public function returns a typed dataclass. Enum fields in results use the enum type, not strings:

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
```

Result dataclasses are separated into `_results.py` files within each subpackage to decouple data definitions from computation logic.

### Shared Validation Helpers (`statscore.utils.validation`)

Four module-level helpers are centralised here and imported by all domain modules:

| Helper | Purpose |
|--------|---------|
| `validate_positive(value, name)` | Raises `ValueError` if `value <= 0` |
| `validate_non_negative(value, name)` | Raises `ValueError` if `value < 0` |
| `validate_1d_sample(x, name, min_obs)` | Raises `ValueError` if `x` is not 1-D or too short |
| `validate_alternative(alternative)` | Raises `TypeError` if not an `AlternativeHypothesis` member |

Domain-specific validators (`validate_design_matrix`, `validate_data_groups`, etc.) remain in the same file.

## Public API

The public API is defined exclusively through `__all__` in each `__init__.py`:

- `statscore.__all__` — 4 enums + 29 statistical functions + 2 regression summary + 4 diagnostics result classes + 4 diagnostic functions + 7 shared plot utilities + 2 I/O (total ~52 symbols)
- `statscore.methods.anova.__all__` — 8 result classes + 11 ANOVA functions
- `statscore.methods.regression.__all__` — 5 result classes + 9 regression functions + `RegressionSummaryResult`, `regression_summary`
- `statscore.methods.testing.__all__` — 6 result classes + 6 testing functions
- `statscore.methods.bayes.__all__` — 2 result classes + 2 Bayesian functions
- `statscore.methods.diagnostics.__all__` — 4 result classes + 4 diagnostic functions

Users should import from the top-level namespace:

```python
from statscore import (
    anova1_test_equality,
    AlternativeHypothesis,
    mult_lr_least_squares,
    CorrectionMethod,
    t_test_mean,
    bayes_normal_mean_unknown_var,
)
```

Direct subpackage imports are also valid:

```python
from statscore.methods.anova import anova1_test_equality
from statscore.methods.regression.least_squares import mult_lr_least_squares
from statscore.methods.testing import z_test_mean
```

## Adding New Modules

1. Place the module under `statscore/methods/<domain>/` in the correct layer.
2. Define result dataclasses in `statscore/methods/<domain>/_results.py`.
3. Import from `_results.py` and sibling submodules directly (never from the package `__init__` to avoid circular imports).
4. Use only absolute imports from `statscore.*`.
5. Respect the layer dependency rules above — domain modules must not import from each other except as noted.
6. Add complete type annotations to all functions and dataclass fields.
7. Define new enums in `statscore/utils/enums.py` for any categorical parameter; never accept raw strings.
8. Use shared helpers from `statscore.utils.validation` instead of redefining local `_validate_*` functions.
9. Export public symbols through the subpackage `__init__.py` and add to `__all__`.
10. Re-export from the top-level `__init__.py` if it belongs to the user-facing API.
11. Add tests in `tests/` and extend `examples/demo.py`.
12. Update `CHANGELOG.md` (new `[x.y.z]` entry) and `ARCHITECTURE.md` (layer rules, DAG, public API counts).
