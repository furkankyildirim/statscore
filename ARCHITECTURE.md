# Import Architecture & Type System

## Dependency Layers

```
┌─────────────────────────────────────────────────────────────┐
│                   statscore (public API)                    │  ← top-level __init__.py
├─────────────────────────────────────────────────────────────┤
│  statscore.anova  statscore.regression  statscore.testing   │  ← domain modules
│  statscore.bayes  statscore.diagnostics statscore.plots     │
│  statscore.io                                               │
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
| `utils` | External packages only (numpy, scipy) | `anova`, `regression`, `testing`, `bayes`, `diagnostics`, `plots`, `io` |
| `anova` | `statscore.utils`, other `anova` submodules | `regression`, `testing`, `bayes`, `diagnostics`, `plots`, `io` |
| `regression` | `statscore.utils`, other `regression` submodules | `anova`, `testing`, `bayes`, `diagnostics`, `plots`, `io` |
| `testing` | `statscore.utils` | `anova`, `regression`, `bayes`, `diagnostics`, `plots`, `io` |
| `bayes` | `statscore.utils` | `anova`, `regression`, `testing`, `diagnostics`, `plots`, `io` |
| `diagnostics` | `statscore.utils`, `statscore.regression.least_squares` | `anova`, `testing`, `bayes`, `plots`, `io` |
| `plots` | External packages only (numpy, scipy, matplotlib) | all statscore domain modules (accepts result dataclasses duck-typed) |
| `io` | External packages only (pandas) | all statscore domain modules |
| top-level `__init__` | all domain modules, `statscore.utils.enums` | — |

No circular dependencies exist. The dependency graph is a strict DAG:

```
utils.enums         ← anova.two_way, anova.multiple_tests, regression.prediction
                    ← testing.one_sample, testing.two_sample
                    ← utils.distributions, utils.validation
utils.distributions ← anova.one_way, anova.two_way, anova.multiple_tests
                    ← regression.least_squares, regression.inference, regression.prediction
                    ← regression.summary
                    ← testing.one_sample, testing.two_sample
                    ← bayes.conjugate
                    ← diagnostics
utils.validation    ← anova.one_way, anova.two_way, anova.multiple_tests
                    ← regression.least_squares, regression.inference, regression.prediction
                    ← testing.one_sample, testing.two_sample
                    ← bayes.conjugate
                    ← diagnostics
anova.one_way       ← anova.multiple_tests (intra-layer)
regression.least_squares ← regression.inference, regression.prediction, regression.summary (intra-layer)
                         ← diagnostics (cross-layer, read-only)
```

## Import Style

All internal imports use **absolute paths** rooted at `statscore`:

```python
# Correct
from statscore.utils.distributions import f_critical
from statscore.utils.enums import CorrectionMethod, AlternativeHypothesis
from statscore.regression.least_squares import Mult_LR_Least_squares

# Incorrect (do not use)
from ..utils.distributions import f_critical
from .least_squares import Mult_LR_Least_squares
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

- `statscore.__all__` — 4 enums + 29 statistical functions + 2 regression summary + 4 diagnostics + 5 plots + 2 I/O (46 total symbols)
- `statscore.anova.__all__` — 11 ANOVA functions
- `statscore.regression.__all__` — 9 regression functions + `RegressionSummaryResult`, `regression_summary` (11 total)
- `statscore.testing.__all__` — 6 testing functions
- `statscore.bayes.__all__` — 2 Bayesian functions

Users should import from the top-level namespace:

```python
from statscore import (
    ANOVA1_test_equality,
    AlternativeHypothesis,
    Mult_LR_Least_squares,
    CorrectionMethod,
    t_test_mean,
    bayes_normal_mean_unknown_var,
)
```

## Adding New Modules

1. Place the module in the correct layer (`utils/`, `anova/`, `regression/`, `testing/`, `bayes/`, or top-level for cross-cutting concerns like `diagnostics`, `plots`, `io`).
2. Use only absolute imports from `statscore.*`.
3. Respect the layer dependency rules above — domain modules must not import from each other.
4. Add complete type annotations to all functions and dataclass fields.
5. Define new enums in `statscore/utils/enums.py` for any categorical parameter; never accept raw strings.
6. Use shared helpers from `statscore.utils.validation` instead of redefining local `_validate_*` functions.
7. Export public symbols through the subpackage `__init__.py` and add to `__all__`.
8. Re-export from the top-level `__init__.py` if it belongs to the user-facing API.
9. Add tests in `tests/` and extend `examples/demo.py`.
10. Update `CHANGELOG.md` (new `[x.y.z]` entry) and `ARCHITECTURE.md` (layer rules, DAG, public API counts).
