# Import Architecture & Type System

## Dependency Layers

```
┌───────────────────────────────────────────────────┐
│              stats_toolbox (public API)           │  ← top-level __init__.py
├───────────────────────────────────────────────────┤
│   stats_toolbox.anova    stats_toolbox.regression │  ← domain modules
├───────────────────────────────────────────────────┤
│              stats_toolbox.utils                  │  ← base layer
│   ├── enums.py         (type definitions)         │
│   ├── distributions.py (scipy wrappers)           │
│   └── validation.py    (input guards)             │
└───────────────────────────────────────────────────┘
```

### Layer Rules

| Layer | May import from | Must NOT import from |
|-------|----------------|---------------------|
| `utils` | External packages only (numpy, scipy) | `anova`, `regression` |
| `anova` | `stats_toolbox.utils`, other `anova` submodules | `regression` |
| `regression` | `stats_toolbox.utils`, other `regression` submodules | `anova` |
| top-level `__init__` | `stats_toolbox.anova`, `stats_toolbox.regression`, `stats_toolbox.utils.enums` | — |

No circular dependencies exist. The dependency graph is a strict DAG:

```
utils.enums         ← anova.two_way, anova.multiple_tests, regression.prediction
utils.distributions ← anova.one_way, anova.two_way, anova.multiple_tests
utils.validation    ← anova.one_way, anova.two_way, anova.multiple_tests
                    ← regression.least_squares, regression.inference, regression.prediction
anova.one_way       ← anova.multiple_tests (intra-layer)
regression.least_squares ← regression.inference, regression.prediction (intra-layer)
```

## Import Style

All internal imports use **absolute paths** rooted at `stats_toolbox`:

```python
# Correct
from stats_toolbox.utils.distributions import f_critical
from stats_toolbox.utils.enums import CorrectionMethod
from stats_toolbox.regression.least_squares import Mult_LR_Least_squares

# Incorrect (do not use)
from ..utils.distributions import f_critical
from .least_squares import Mult_LR_Least_squares
```

## Type System

### Enum-Based Categorical Parameters

All categorical function parameters use strongly-typed enums defined in `stats_toolbox/utils/enums.py`:

| Enum | Members | Controls |
|------|---------|----------|
| `CorrectionMethod` | `SCHEFFE`, `TUKEY`, `BONFERRONI`, `SIDAK`, `BEST` | Multiple comparison method |
| `PredictionMethod` | `SCHEFFE`, `BONFERRONI`, `BEST` | Prediction interval method |
| `TwoWayTestFactor` | `A`, `B`, `AB` | Two-way ANOVA test target |

Usage:
```python
from stats_toolbox import ANOVA1_CI_linear_combs, CorrectionMethod

result = ANOVA1_CI_linear_combs(data, 0.05, C, method=CorrectionMethod.BONFERRONI)
```

### Typing Rules

1. Every function has explicit `-> ReturnType` annotations.
2. All parameters are typed — no bare `def f(data, alpha, method)`.
3. Collections use parameterized types: `list[tuple[float, float]]`, `dict[str, dict[str, float]]`.
4. `Optional[T]` only when `None` is semantically meaningful.
5. No `Any` in the public API.
6. All return objects are `@dataclass` with fully-typed fields.

### Dataclass Output Contract

Every public function returns a typed dataclass. Enum fields in results use the enum type, not strings:

```python
@dataclass
class SimultaneousCIResult:
    intervals: list[tuple[float, float]]
    method_used: CorrectionMethod        # enum, not str
    point_estimates: np.ndarray
    half_widths: np.ndarray
```

## Public API

The public API is defined exclusively through `__all__` in each `__init__.py`:

- `stats_toolbox.__all__` — 3 enums + 20 functions (23 total symbols)
- `stats_toolbox.anova.__all__` — 11 ANOVA functions
- `stats_toolbox.regression.__all__` — 9 regression functions
- `stats_toolbox.utils.__all__` — internal utilities + enums

Users should import from the top-level namespace:

```python
from stats_toolbox import (
    ANOVA1_test_equality,
    Mult_LR_Least_squares,
    CorrectionMethod,
    PredictionMethod,
    TwoWayTestFactor,
)
```

## Adding New Modules

1. Place the module in the correct layer (`utils/`, `anova/`, or `regression/`).
2. Use only absolute imports from `stats_toolbox.*`.
3. Respect the layer dependency rules above.
4. Add complete type annotations to all functions and dataclass fields.
5. Define new enums in `stats_toolbox/utils/enums.py` for any categorical parameter.
6. Export public symbols through the subpackage `__init__.py` and add to `__all__`.
7. Re-export from the top-level `__init__.py` if it belongs to the user-facing API.
