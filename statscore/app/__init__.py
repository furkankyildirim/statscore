"""Streamlit browser UI for statscore."""

from __future__ import annotations

import contextlib
import re

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from scipy import stats

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="statscore",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "df" not in st.session_state:
    st.session_state["df"] = None


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[\s,;]+")


def _parse_numbers(text: str) -> np.ndarray:
    tokens = _TOKEN_RE.split(text.strip())
    return np.array([float(t) for t in tokens if t], dtype=float)


def _parse_matrix(text: str) -> np.ndarray:
    rows = [
        [float(t) for t in _TOKEN_RE.split(line.strip()) if t]
        for line in text.strip().splitlines()
        if line.strip()
    ]
    return np.array(rows, dtype=float)


def _show_fig(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _alt_enum(choice: str):
    from statscore.utils.enums import AlternativeHypothesis
    return {
        "two-sided": AlternativeHypothesis.TWO_SIDED,
        "less":      AlternativeHypothesis.LESS,
        "greater":   AlternativeHypothesis.GREATER,
    }[choice]


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: DATA INPUT
# ─────────────────────────────────────────────────────────────────────────────

def page_data_input():
    st.header("Data Input")

    tab_file, tab_paste = st.tabs(["Upload file", "Paste numbers"])

    with tab_file:
        uploaded = st.file_uploader(
            "Upload CSV, TSV, XLSX, or JSON",
            type=["csv", "tsv", "xlsx", "xls", "json"],
        )
        if uploaded is not None:
            try:
                ext = Path(uploaded.name).suffix.lower()
                readers = {
                    ".csv":  lambda f: pd.read_csv(f),
                    ".tsv":  lambda f: pd.read_csv(f, sep="\t"),
                    ".xlsx": lambda f: pd.read_excel(f),
                    ".xls":  lambda f: pd.read_excel(f),
                    ".json": lambda f: pd.read_json(f),
                }
                if ext not in readers:
                    st.error(f"Unsupported extension: {ext}")
                else:
                    df = readers[ext](uploaded)
                    st.session_state["df"] = df
                    st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as exc:
                st.error(f"Could not load file: {exc}")

    with tab_paste:
        st.write(
            "Paste raw numbers (one column per line for multi-group data, "
            "or comma-separated rows for a matrix)."
        )
        raw = st.text_area("Raw data", height=200, placeholder="1.2, 3.4, 5.6\n7.8, 9.0, 2.1")
        col_names = st.text_input("Column names (comma-separated, optional)", "")
        if st.button("Load pasted data"):
            try:
                mat = _parse_matrix(raw)
                if mat.ndim == 1 or mat.shape[1] == 1:
                    mat = mat.reshape(-1, 1)
                names = [n.strip() for n in col_names.split(",") if n.strip()] if col_names else None
                if names and len(names) == mat.shape[1]:
                    df = pd.DataFrame(mat, columns=names)
                else:
                    df = pd.DataFrame(mat, columns=[f"col{i+1}" for i in range(mat.shape[1])])
                st.session_state["df"] = df
                st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as exc:
                st.error(f"Parse error: {exc}")

    df = st.session_state["df"]
    if df is not None:
        st.subheader("Preview")
        st.dataframe(df, use_container_width=True)
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.write("Numeric summary:")
        st.dataframe(df.describe(), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: ANOVA
# ─────────────────────────────────────────────────────────────────────────────

def page_anova():
    st.header("ANOVA")
    kind = st.radio("Type", ["One-Way", "Two-Way"], horizontal=True)
    alpha = st.number_input("Significance level α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)

    if kind == "One-Way":
        _anova_one_way(alpha)
    else:
        _anova_two_way(alpha)


def _anova_one_way(alpha: float):
    from statscore.methods.anova.one_way import anova1_test_equality
    from statscore.plots import plot_anova_groups

    st.subheader("One-Way ANOVA")
    df = st.session_state["df"]
    input_mode = st.radio("Data source", ["From loaded dataset", "Manual entry"], horizontal=True)
    data: list[np.ndarray] = []
    group_labels: list[str] = []

    if input_mode == "From loaded dataset" and df is not None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        selected = st.multiselect("Select columns (each = one group)", numeric_cols,
                                  default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)
        for col in selected:
            data.append(df[col].dropna().values.astype(float))
            group_labels.append(col)
    else:
        n_groups = st.number_input("Number of groups", min_value=2, max_value=10, value=3, step=1)
        for i in range(int(n_groups)):
            raw   = st.text_input(f"Group {i+1} data (comma/space separated)", key=f"anova1_g{i}")
            label = st.text_input(f"Group {i+1} label", value=f"Group {i+1}", key=f"anova1_l{i}")
            if raw:
                try:
                    data.append(_parse_numbers(raw))
                    group_labels.append(label)
                except Exception:
                    st.warning(f"Could not parse group {i+1}")

    if st.button("Run One-Way ANOVA") and len(data) >= 2:
        try:
            result = anova1_test_equality(data, alpha=alpha)

            anova_df = pd.DataFrame({
                "Source": ["Between", "Within", "Total"],
                "df":     [result.df_between, result.df_within, result.df_total],
                "SS":     [result.SS_between, result.SS_within, result.SS_total],
                "MS":     [result.MS_between, result.MS_within, ""],
                "F":      [f"{result.F_statistic:.4f}", "", ""],
                "p-value":[f"{result.p_value:.4f}", "", ""],
            })
            st.subheader("ANOVA Table")
            st.dataframe(anova_df, use_container_width=True, hide_index=True)

            dec_color = "red" if result.reject_H0 else "green"
            decision  = "Reject H₀ — means differ" if result.reject_H0 else "Fail to reject H₀"
            st.markdown(
                f"**F critical:** {result.F_critical:.4f} &nbsp;|&nbsp; "
                f"**p-value:** {result.p_value:.4f} &nbsp;|&nbsp; "
                f":{dec_color}[**{decision}**]"
            )

            col1, col2 = st.columns(2)
            with col1:
                _show_fig(plot_anova_groups(data, group_labels=group_labels, title="Group Distributions"))
            with col2:
                _show_fig(result.plot())

        except Exception as exc:
            st.error(f"ANOVA error: {exc}")


def _anova_two_way(alpha: float):
    from statscore.methods.anova.two_way import anova2_test_equality
    from statscore.utils.enums import TwoWayTestFactor

    st.subheader("Two-Way ANOVA")
    st.info(
        "Enter data as a 3-D array (I × J × K). "
        "Provide each cell (i,j) as a line of K values; rows = Factor A levels, "
        "blocks of J lines = one level of Factor A."
    )

    I = int(st.number_input("Levels of Factor A (I)", min_value=2, max_value=8, value=2, step=1))
    J = int(st.number_input("Levels of Factor B (J)", min_value=2, max_value=8, value=2, step=1))
    K = int(st.number_input("Replications per cell (K)", min_value=1, max_value=20, value=2, step=1))

    st.write(f"Enter {I*J} cells, {K} values each. Row order: A=1,B=1 | A=1,B=2 | … | A={I},B={J}")
    cell_inputs = [
        st.text_input(f"A={i+1}, B={j+1}", key=f"anova2_{i}_{j}", placeholder=f"{K} values")
        for i in range(I) for j in range(J)
    ]

    factor_map = {"A": TwoWayTestFactor.A, "B": TwoWayTestFactor.B, "AB": TwoWayTestFactor.AB}
    test_factor = st.selectbox("Test factor", list(factor_map))

    if st.button("Run Two-Way ANOVA"):
        try:
            cells = []
            for raw in cell_inputs:
                vals = _parse_numbers(raw)
                if len(vals) != K:
                    raise ValueError(f"Expected {K} values per cell, got {len(vals)}")
                cells.append(vals)
            data   = np.array(cells).reshape(I, J, K)
            result = anova2_test_equality(data, alpha=alpha, test=factor_map[test_factor])

            src_labels = [("A", "Factor A"), ("B", "Factor B"), ("AB", "Interaction AB"),
                          ("within", "Within (Error)"), ("total", "Total")]
            rows = [
                {
                    "Source": label,
                    "df":     int(entry["df"]),
                    "SS":     f"{entry['SS']:.4f}",
                    "MS":     f"{entry['MS']:.4f}" if "MS" in entry else "",
                    "F":      f"{entry['F']:.4f}"  if "F"  in entry else "",
                }
                for src, label in src_labels
                for entry in [result.full_table[src]]
            ]
            st.subheader("ANOVA Table")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            decision = "Reject H₀" if result.reject_H0 else "Fail to reject H₀"
            st.markdown(
                f"**Testing:** {test_factor} &nbsp;|&nbsp; **F:** {result.F_statistic:.4f} "
                f"&nbsp;|&nbsp; **F crit:** {result.F_critical:.4f} &nbsp;|&nbsp; "
                f"**p:** {result.p_value:.4f} &nbsp;|&nbsp; **{decision}**"
            )
            _show_fig(result.plot())

        except Exception as exc:
            st.error(f"Two-Way ANOVA error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: SIGNIFICANCE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def page_significance_tests():
    st.header("Significance Tests")

    test_type = st.selectbox("Test", [
        "Z-test (mean, σ known)",
        "t-test (one-sample)",
        "t-test (two-sample)",
        "t-test (paired)",
        "Chi² test (variance)",
        "F-test (two variances)",
    ])
    alpha = st.number_input("α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)
    alt   = st.selectbox("Alternative", ["two-sided", "greater", "less"])

    dispatch = {
        "Z-test (mean, σ known)": _test_z,
        "t-test (one-sample)":    _test_t_one,
        "t-test (two-sample)":    _test_t_two,
        "t-test (paired)":        _test_t_paired,
        "Chi² test (variance)":   _test_chi2,
        "F-test (two variances)": _test_f_var,
    }
    dispatch[test_type](alpha, alt)


def _sample_input(label: str, key: str, df_col_key: str) -> np.ndarray | None:
    df   = st.session_state["df"]
    mode = st.radio(f"{label} source", ["Manual", "From dataset"], horizontal=True, key=key + "_mode")
    if mode == "Manual":
        raw = st.text_input(f"{label} (comma/space separated)", key=key)
        if raw:
            try:
                return _parse_numbers(raw)
            except Exception as e:
                st.warning(f"Parse error: {e}")
        return None
    if df is None:
        st.info("No dataset loaded. Use the Data Input page first.")
        return None
    cols = df.select_dtypes(include="number").columns.tolist()
    col  = st.selectbox(f"Column for {label}", cols, key=df_col_key)
    return df[col].dropna().values.astype(float)


def _test_z(alpha, alt):
    from statscore.methods.testing.one_sample import z_test_mean
    from statscore.plots import plot_t_test

    x    = _sample_input("Sample x", "z_x", "z_x_col")
    mu0   = st.number_input("μ₀ (hypothesized mean)", value=0.0, key="z_mu0")
    sigma = st.number_input("σ (known std)", value=1.0, min_value=1e-9, key="z_sigma")

    if st.button("Run Z-test") and x is not None:
        try:
            r = z_test_mean(x, mu0=mu0, sigma=sigma, alpha=alpha, alternative=_alt_enum(alt))
            _show_test_result("Z-test", {"z-stat": r.z_statistic, "z-crit": r.z_critical,
                                         "p-value": r.p_value, "n": r.n, "x̄": r.x_bar}, r.reject_H0)
            _show_fig(plot_t_test(r.z_statistic, r.z_critical, 10000, alt, title="Z-Test (Normal)"))
        except Exception as e:
            st.error(str(e))


def _test_t_one(alpha, alt):
    from statscore.methods.testing.one_sample import t_test_mean
    from statscore.plots import plot_t_test

    x   = _sample_input("Sample x", "t1_x", "t1_x_col")
    mu0 = st.number_input("μ₀", value=0.0, key="t1_mu0")

    if st.button("Run t-test (one-sample)") and x is not None:
        try:
            r = t_test_mean(x, mu0=mu0, alpha=alpha, alternative=_alt_enum(alt))
            _show_test_result("One-Sample t-test",
                              {"t-stat": r.t_statistic, "t-crit": r.t_critical,
                               "p-value": r.p_value, "n": r.n, "x̄": r.x_bar, "s": r.s, "df": r.df},
                              r.reject_H0)
            _show_fig(plot_t_test(r.t_statistic, r.t_critical, r.df, alt, title="One-Sample t-Test"))
        except Exception as e:
            st.error(str(e))


def _test_t_two(alpha, alt):
    from statscore.methods.testing.two_sample import t_test_two_sample
    from statscore.plots import plot_t_test

    x1        = _sample_input("Sample x₁", "t2_x1", "t2_x1_col")
    x2        = _sample_input("Sample x₂", "t2_x2", "t2_x2_col")
    equal_var = st.checkbox("Assume equal variances (pooled)", value=True)

    if st.button("Run t-test (two-sample)") and x1 is not None and x2 is not None:
        try:
            r = t_test_two_sample(x1, x2, alpha=alpha, alternative=_alt_enum(alt), equal_var=equal_var)
            _show_test_result("Two-Sample t-test",
                              {"t-stat": r.t_statistic, "t-crit": r.t_critical,
                               "p-value": r.p_value, "n₁": r.n1, "n₂": r.n2,
                               "x̄₁": r.x1_bar, "x̄₂": r.x2_bar, "df": r.df},
                              r.reject_H0)
            _show_fig(plot_t_test(r.t_statistic, r.t_critical, r.df, alt, title="Two-Sample t-Test"))
        except Exception as e:
            st.error(str(e))


def _test_t_paired(alpha, alt):
    from statscore.methods.testing.two_sample import t_test_paired
    from statscore.plots import plot_t_test

    x1 = _sample_input("Sample x₁", "tp_x1", "tp_x1_col")
    x2 = _sample_input("Sample x₂", "tp_x2", "tp_x2_col")

    if st.button("Run t-test (paired)") and x1 is not None and x2 is not None:
        try:
            r = t_test_paired(x1, x2, alpha=alpha, alternative=_alt_enum(alt))
            _show_test_result("Paired t-test",
                              {"t-stat": r.t_statistic, "t-crit": r.t_critical,
                               "p-value": r.p_value, "n": r.n, "d̄": r.d_bar, "s_d": r.s_d, "df": r.df},
                              r.reject_H0)
            _show_fig(plot_t_test(r.t_statistic, r.t_critical, r.df, alt, title="Paired t-Test"))
        except Exception as e:
            st.error(str(e))


def _test_chi2(alpha, alt):
    from statscore.methods.testing.one_sample import chi2_test_variance

    x         = _sample_input("Sample x", "chi2_x", "chi2_x_col")
    sigma0_sq = st.number_input("σ₀² (hypothesized variance)", value=1.0, min_value=1e-9, key="chi2_s0")

    if st.button("Run Chi² test") and x is not None:
        try:
            r = chi2_test_variance(x, sigma0_sq=sigma0_sq, alpha=alpha, alternative=_alt_enum(alt))
            _show_test_result("Chi² Variance Test",
                              {"χ²-stat": r.chi2_statistic,
                               "χ²-crit lower": r.chi2_critical_lower,
                               "χ²-crit upper": r.chi2_critical_upper,
                               "p-value": r.p_value, "n": r.n, "s²": r.s2, "df": r.df},
                              r.reject_H0)
            fig, ax = plt.subplots(figsize=(8, 4))
            x_v = np.linspace(0.01, max(r.chi2_statistic * 2, stats.chi2.ppf(0.999, r.df)), 400)
            pdf = stats.chi2.pdf(x_v, r.df)
            ax.plot(x_v, pdf, color="steelblue", lw=2, label=f"χ²(df={r.df})")
            if alt == "two-sided":
                mask = (x_v <= r.chi2_critical_lower) | (x_v >= r.chi2_critical_upper)
            elif alt == "greater":
                mask = x_v >= r.chi2_critical_upper
            else:
                mask = x_v <= r.chi2_critical_lower
            ax.fill_between(x_v, pdf, where=mask, alpha=0.3, color="crimson", label="Rejection region")
            ax.axvline(r.chi2_statistic, color="darkgreen", lw=2, label=f"χ² = {r.chi2_statistic:.4f}")
            ax.set(xlabel="χ²", ylabel="Density", title="Chi² Test")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            _show_fig(fig)
        except Exception as e:
            st.error(str(e))


def _test_f_var(alpha, alt):
    from statscore.methods.testing.two_sample import f_test_variances
    from statscore.plots import plot_f_test

    x1 = _sample_input("Sample x₁", "fv_x1", "fv_x1_col")
    x2 = _sample_input("Sample x₂", "fv_x2", "fv_x2_col")

    if st.button("Run F-test (variances)") and x1 is not None and x2 is not None:
        try:
            r = f_test_variances(x1, x2, alpha=alpha, alternative=_alt_enum(alt))
            _show_test_result("F-Test for Variances",
                              {"F-stat": r.f_statistic,
                               "F-crit lower": r.f_critical_lower,
                               "F-crit upper": r.f_critical_upper,
                               "p-value": r.p_value,
                               "n₁": r.n1, "n₂": r.n2,
                               "s₁²": r.s1_sq, "s₂²": r.s2_sq,
                               "df₁": r.df1, "df₂": r.df2},
                              r.reject_H0)
            _show_fig(plot_f_test(r.f_statistic, r.f_critical_lower, r.f_critical_upper,
                                  r.df1, r.df2, alt, title="F-Test (Variances)"))
        except Exception as e:
            st.error(str(e))


def _show_test_result(title: str, metrics: dict, reject: bool) -> None:
    st.subheader(title)
    cols = st.columns(min(len(metrics), 4))
    for i, (k, v) in enumerate(metrics.items()):
        with cols[i % len(cols)]:
            st.metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    color = "red" if reject else "green"
    st.markdown(f"### :{color}[{'Reject H₀' if reject else 'Fail to reject H₀'}]")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def page_regression():
    st.header("Regression")

    alpha = st.number_input("α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)
    df    = st.session_state["df"]
    input_mode = st.radio("Data source", ["From loaded dataset", "Manual entry"], horizontal=True)

    X_mat: np.ndarray | None = None
    y_vec: np.ndarray | None = None
    feature_names: list[str] = []

    if input_mode == "From loaded dataset" and df is not None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        y_col  = st.selectbox("Response column (y)", numeric_cols)
        x_cols = st.multiselect("Predictor columns (X)", [c for c in numeric_cols if c != y_col],
                                default=[c for c in numeric_cols if c != y_col][:1])
        intercept = st.checkbox("Include intercept column", value=True)
        if x_cols and y_col:
            sub   = df[[y_col] + x_cols].dropna()
            y_vec = sub[y_col].values.astype(float)
            X_raw = sub[x_cols].values.astype(float)
            if intercept:
                X_mat = np.column_stack([np.ones(len(X_raw)), X_raw])
                feature_names = ["(Intercept)"] + x_cols
            else:
                X_mat = X_raw
                feature_names = x_cols
    else:
        st.write("Enter the design matrix X (one row per observation, columns separated by spaces/commas).")
        intercept = st.checkbox("Prepend intercept column of 1s", value=True)
        x_raw    = st.text_area("X matrix (rows = observations)", height=150,
                                placeholder="1 2.3\n1 4.5\n1 6.7")
        y_raw    = st.text_input("y vector (comma/space separated)", placeholder="10.1 12.3 15.6")
        feat_raw = st.text_input("Feature names (comma-separated, optional)", "")
        if x_raw and y_raw:
            try:
                X_parsed = _parse_matrix(x_raw)
                y_vec    = _parse_numbers(y_raw)
                if intercept and not np.allclose(X_parsed[:, 0], 1.0):
                    X_mat = np.column_stack([np.ones(len(X_parsed)), X_parsed])
                else:
                    X_mat = X_parsed
                names = [n.strip() for n in feat_raw.split(",") if n.strip()]
                if names:
                    feature_names = names if len(names) == X_mat.shape[1] else ["(Intercept)"] + names
                else:
                    feature_names = [f"x{i}" for i in range(X_mat.shape[1])]
            except Exception as e:
                st.error(f"Parse error: {e}")

    reg_tab1, reg_tab2, reg_tab3, reg_tab4 = st.tabs([
        "Summary & Diagnostics",
        "Simultaneous CIs for β",
        "Hypothesis Tests H₀: Cβ = c₀",
        "Prediction Intervals",
    ])

    if st.button("Run Regression") and X_mat is not None and y_vec is not None:
        try:
            from statscore.methods.diagnostics import regression_diagnostics
            from statscore.methods.regression.least_squares import mult_lr_least_squares
            from statscore.methods.regression.summary import regression_summary

            st.session_state["_reg_X"]    = X_mat
            st.session_state["_reg_y"]    = y_vec
            st.session_state["_reg_res"]  = regression_summary(X_mat, y_vec, alpha=alpha, feature_names=feature_names)
            st.session_state["_reg_ols"]  = mult_lr_least_squares(X_mat, y_vec)
            st.session_state["_reg_diag"] = regression_diagnostics(X_mat, y_vec)
            st.session_state["_reg_feat"] = feature_names
        except Exception as e:
            st.error(f"Regression error: {e}")

    # ── Tab 1: Summary & Diagnostics ─────────────────────────────────────────
    with reg_tab1:
        result = st.session_state.get("_reg_res")
        ols    = st.session_state.get("_reg_ols")
        diag   = st.session_state.get("_reg_diag")
        feat   = st.session_state.get("_reg_feat", feature_names)

        if result is None:
            st.info("Press **Run Regression** to compute results.")
        else:
            coef_data = {
                "Variable": feat or [f"β{i}" for i in range(result.p)],
                "Estimate": [float(b) for b in result.beta_hat],
                "Std.Err":  [float(s) for s in result.std_errors],
                "t-stat":   [float(t) for t in result.t_statistics],
                "p-value":  [float(p) for p in result.p_values],
                "Sig.":     [_sig_stars(float(p)) for p in result.p_values],
                "CI lower": [ci[0] for ci in result.conf_intervals],
                "CI upper": [ci[1] for ci in result.conf_intervals],
            }
            st.subheader("Coefficient Summary")
            st.dataframe(
                pd.DataFrame(coef_data).set_index("Variable")
                  .style.format("{:.4f}", subset=["Estimate","Std.Err","t-stat","p-value","CI lower","CI upper"]),
                use_container_width=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²",     f"{result.R_squared:.4f}")
            c2.metric("Adj R²", f"{result.adj_R_squared:.4f}")
            c3.metric("Sₑ",     f"{result.Se:.4f}")
            c4.metric(f"F({result.p-1},{result.df_residual})",
                      f"{result.F_statistic:.4f}  (p={result.F_p_value:.4f})")

            st.subheader("Diagnostic Plots")
            X_s = st.session_state.get("_reg_X")
            y_s = st.session_state.get("_reg_y")
            col1, col2 = st.columns(2)
            with col1:
                if X_s is not None and X_s.shape[1] == 2:
                    from statscore.plots import plot_regression
                    _show_fig(plot_regression(X_s[:, 1], y_s, result.beta_hat,
                                             x_label=feat[1] if len(feat) > 1 else "x",
                                             y_label="y", title="Scatter + Fit"))
                else:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.scatter(result.beta_hat, np.arange(result.p), color="steelblue", s=60)
                    ax.axvline(0, color="crimson", ls="--", lw=1)
                    ax.set_yticks(np.arange(result.p))
                    ax.set_yticklabels(feat or [f"β{i}" for i in range(result.p)])
                    ax.set_title("Coefficient Estimates")
                    ax.grid(alpha=0.3, axis="x")
                    plt.tight_layout()
                    _show_fig(fig)
            with col2:
                from statscore.plots import plot_residuals
                _show_fig(plot_residuals(ols.fitted_values, ols.residuals))
            col3, col4 = st.columns(2)
            with col3:
                from statscore.plots import plot_qq
                _show_fig(plot_qq(ols.residuals, title="Q-Q Plot (Residuals)"))
            with col4:
                _show_fig(diag.plot())

    # ── Tab 2: Simultaneous CIs for β ────────────────────────────────────────
    with reg_tab2:
        result = st.session_state.get("_reg_res")
        X_s    = st.session_state.get("_reg_X")
        y_s    = st.session_state.get("_reg_y")
        feat   = st.session_state.get("_reg_feat", feature_names)

        if result is None:
            st.info("Press **Run Regression** first.")
        elif st.button("Compute Simultaneous CIs for β", key="simci_btn"):
            try:
                from statscore.methods.regression.inference import mult_norm_lr_simul_ci
                from statscore.plots import plot_simultaneous_ci
                ci_res = mult_norm_lr_simul_ci(X_s, y_s, alpha=alpha)
                st.write(f"**Method:** {ci_res.method.value}")
                rows = [
                    {
                        "Parameter":  feat[i] if i < len(feat) else f"β{i}",
                        "Estimate":   float(ci_res.beta_hat[i]),
                        "Half-width": float(ci_res.half_widths[i]),
                        "CI lower":   lo,
                        "CI upper":   hi,
                        "Contains 0": "Yes" if lo <= 0 <= hi else "No",
                    }
                    for i, (lo, hi) in enumerate(ci_res.intervals)
                ]
                st.dataframe(pd.DataFrame(rows).set_index("Parameter"), use_container_width=True)
                _show_fig(plot_simultaneous_ci(ci_res.beta_hat, ci_res.intervals,
                                               method=ci_res.method.value, labels=feat or None,
                                               title="Simultaneous CIs for β"))
            except Exception as e:
                st.error(str(e))

    # ── Tab 3: Hypothesis Tests ───────────────────────────────────────────────
    with reg_tab3:
        result = st.session_state.get("_reg_res")
        X_s    = st.session_state.get("_reg_X")
        y_s    = st.session_state.get("_reg_y")
        feat   = st.session_state.get("_reg_feat", feature_names)

        if result is None:
            st.info("Press **Run Regression** first.")
        else:
            test_kind = st.radio("Test type", [
                "Overall F (existence of linear regression)",
                "Component test  H₀: β_j = 0  for selected j",
                "General test  H₀: Cβ = c₀",
            ], key="reg_test_kind")

            if test_kind == "Overall F (existence of linear regression)":
                if st.button("Run overall F-test", key="reg_olf"):
                    try:
                        from statscore.methods.regression.inference import (
                            mult_norm_lr_test_linear_reg,
                        )
                        _show_reg_htest(mult_norm_lr_test_linear_reg(X_s, y_s, alpha=alpha),
                                        "Overall F-Test (H₀: β₁=…=βₖ=0)")
                    except Exception as e:
                        st.error(str(e))

            elif test_kind == "Component test  H₀: β_j = 0  for selected j":
                p_total = result.p
                names   = feat or [f"β{i}" for i in range(p_total)]
                sel = st.multiselect("Components to test (β_j = 0)", options=list(range(p_total)),
                                     format_func=lambda i: names[i],
                                     default=[1] if p_total > 1 else [0],
                                     key="reg_comp_sel")
                if st.button("Run component test", key="reg_comp_btn") and sel:
                    try:
                        from statscore.methods.regression.inference import mult_norm_lr_test_comp
                        _show_reg_htest(mult_norm_lr_test_comp(X_s, y_s, alpha=alpha, components=sel),
                                        f"Component Test  H₀: {[names[i] for i in sel]} = 0")
                    except Exception as e:
                        st.error(str(e))

            else:
                p_total = result.p
                st.write(f"Enter C matrix ({p_total} columns) and c₀ vector.")
                C_raw  = st.text_area("C matrix (rows by newlines)",
                                      placeholder=f"0 1 {'0 ' * (p_total - 2)}".strip(),
                                      height=80, key="reg_C_raw")
                c0_raw = st.text_input("c₀ vector", value="0", key="reg_c0_raw")
                if st.button("Run general test", key="reg_gen_btn"):
                    try:
                        from statscore.methods.regression.inference import mult_norm_lr_test_general
                        _show_reg_htest(mult_norm_lr_test_general(X_s, y_s,
                                                                   _parse_matrix(C_raw),
                                                                   _parse_numbers(c0_raw),
                                                                   alpha=alpha),
                                        "General Test  H₀: Cβ = c₀")
                    except Exception as e:
                        st.error(str(e))

    # ── Tab 4: Prediction Intervals ───────────────────────────────────────────
    with reg_tab4:
        result = st.session_state.get("_reg_res")
        X_s    = st.session_state.get("_reg_X")
        y_s    = st.session_state.get("_reg_y")

        if result is None:
            st.info("Press **Run Regression** first.")
        else:
            p_total = result.p
            st.write(f"Enter prediction-point matrix D (one row per point, {p_total} columns matching X).")
            D_raw       = st.text_area("D matrix", height=80, key="reg_D_raw",
                                       placeholder="1 0.5 1.0\n1 1.0 0.0")
            pred_method = st.selectbox("Method", ["best", "Bonferroni", "Scheffe"], key="reg_pred_meth")

            if st.button("Compute Prediction Intervals", key="reg_pred_btn"):
                try:
                    from statscore.methods.regression.prediction import mult_norm_lr_pred_ci
                    from statscore.plots import plot_simultaneous_ci
                    from statscore.utils.enums import PredictionMethod
                    pm  = {"best": PredictionMethod.BEST,
                           "Bonferroni": PredictionMethod.BONFERRONI,
                           "Scheffe":    PredictionMethod.SCHEFFE}[pred_method]
                    r   = mult_norm_lr_pred_ci(X_s, y_s, _parse_matrix(D_raw), alpha=alpha, method=pm)
                    st.write(f"**Method:** {r.method_used.value}")
                    rows = [
                        {
                            "Point":       f"d_{i+1}",
                            "Predicted ŷ": float(r.point_estimates[i]),
                            "Half-width":  float(r.half_widths[i]),
                            "PI lower":    lo,
                            "PI upper":    hi,
                        }
                        for i, (lo, hi) in enumerate(r.intervals)
                    ]
                    st.dataframe(pd.DataFrame(rows).set_index("Point"), use_container_width=True)
                    _show_fig(plot_simultaneous_ci(r.point_estimates, r.intervals,
                                                   method=r.method_used.value,
                                                   labels=[f"d_{i+1}" for i in range(len(r.intervals))],
                                                   title="Simultaneous Prediction Intervals"))
                except Exception as e:
                    st.error(str(e))


def _show_reg_htest(r, title: str) -> None:
    from statscore.plots import plot_f_test
    st.subheader(title)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"F({r.df_numerator},{r.df_denominator})", f"{r.test_statistic:.4f}")
    c2.metric("F critical", f"{r.F_critical:.4f}")
    c3.metric("p-value",    f"{r.p_value:.4f}")
    c4.metric("α",          f"{r.alpha:.4f}")
    color = "red" if r.reject_H0 else "green"
    st.markdown(f"### :{color}[{'Reject H₀' if r.reject_H0 else 'Fail to reject H₀'}]")
    _show_fig(plot_f_test(r.test_statistic, 0.0, r.F_critical,
                          r.df_numerator, r.df_denominator, "greater",
                          title=f"F-Test — {title}"))


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: BAYESIAN INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def page_bayesian():
    st.header("Bayesian Inference")
    model = st.radio("Model", [
        "Normal – known variance",
        "Normal – unknown variance (Normal-Gamma)",
        "Beta-Binomial (success probability)",
        "Gamma-Poisson (count rate)",
        "MCMC – Normal mean & variance",
        "MCMC – Linear Regression",
    ], horizontal=False)
    alpha = st.number_input("α (credible interval = 1–α)", value=0.05, min_value=0.001, max_value=0.5, step=0.01)

    if model == "Normal – known variance":
        _bayes_known_var(_sample_input("Observations x", "bayes_x", "bayes_x_col"), alpha)
    elif model == "Normal – unknown variance (Normal-Gamma)":
        _bayes_unknown_var(_sample_input("Observations x", "bayes_x2", "bayes_x2_col"), alpha)
    elif model == "Beta-Binomial (success probability)":
        _bayes_beta_binomial(alpha)
    elif model == "Gamma-Poisson (count rate)":
        _bayes_gamma_poisson(_sample_input("Count observations x", "bayes_gp", "bayes_gp_col"), alpha)
    elif model == "MCMC – Normal mean & variance":
        _mcmc_normal(_sample_input("Observations x", "mcmc_norm_x", "mcmc_norm_x_col"), alpha)
    elif model == "MCMC – Linear Regression":
        _mcmc_regression(alpha)


def _bayes_known_var(x, alpha):
    from statscore.methods.bayes.conjugate import bayes_normal_mean_known_var

    st.subheader("Prior: μ ~ N(μ₀, σ²/κ₀)")
    c1, c2, c3 = st.columns(3)
    mu0    = c1.number_input("μ₀ (prior mean)", value=0.0, key="bkv_mu0")
    kappa0 = c2.number_input("κ₀ (prior precision scaling)", value=1.0, min_value=1e-9, key="bkv_k0")
    sigma2 = c3.number_input("σ² (known variance)", value=1.0, min_value=1e-9, key="bkv_s2")

    if st.button("Run Bayesian (known var)") and x is not None:
        try:
            r = bayes_normal_mean_known_var(x, sigma_sq=sigma2, mu0=mu0, kappa0=kappa0, alpha=alpha)
            pct = int((1 - r.alpha) * 100)
            st.subheader("Posterior Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Posterior mean (μₙ)", f"{r.posterior_mean:.4f}")
            c2.metric("Posterior std",        f"{r.posterior_std:.4f}")
            c3.metric(f"{pct}% CI lower",     f"{r.credible_interval[0]:.4f}")
            c4.metric(f"{pct}% CI upper",     f"{r.credible_interval[1]:.4f}")
            c5, c6 = st.columns(2)
            c5.metric("κₙ", f"{r.kappa_n:.4f}")
            c6.metric("Predictive std", f"{r.predictive_std:.4f}")
            _show_fig(r.plot())
        except Exception as e:
            st.error(str(e))


def _bayes_unknown_var(x, alpha):
    from statscore.methods.bayes.conjugate import bayes_normal_mean_unknown_var

    st.subheader("Prior: (μ,τ) ~ Normal-Gamma(μ₀, κ₀, α₀, β₀)")
    c1, c2, c3, c4 = st.columns(4)
    mu0    = c1.number_input("μ₀", value=0.0, key="buv_mu0")
    kappa0 = c2.number_input("κ₀", value=1.0, min_value=1e-9, key="buv_k0")
    alpha0 = c3.number_input("α₀", value=1.0, min_value=1e-9, key="buv_a0")
    beta0  = c4.number_input("β₀", value=1.0, min_value=1e-9, key="buv_b0")

    if st.button("Run Bayesian (unknown var)") and x is not None:
        try:
            r   = bayes_normal_mean_unknown_var(x, mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0, alpha=alpha)
            pct = int((1 - r.alpha) * 100)
            st.subheader("Posterior Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Posterior mean μ",   f"{r.posterior_mean_mu:.4f}")
            c2.metric("E[τ] (precision)",   f"{r.posterior_mean_precision:.4f}")
            c3.metric(f"{pct}% μ CI lower", f"{r.mu_credible_interval[0]:.4f}")
            c4.metric(f"{pct}% μ CI upper", f"{r.mu_credible_interval[1]:.4f}")
            c5, c6 = st.columns(2)
            c5.metric(f"{pct}% σ² CI lower", f"{r.variance_credible_interval[0]:.4f}")
            c6.metric(f"{pct}% σ² CI upper", f"{r.variance_credible_interval[1]:.4f}")
            st.write("**Posterior hyperparameters:**",
                     "  ".join(f"{k} = {v:.4f}" for k, v in
                                {"μₙ": r.mu_n, "κₙ": r.kappa_n, "αₙ": r.alpha_n, "βₙ": r.beta_n}.items()))
            _show_fig(r.plot())
        except Exception as e:
            st.error(str(e))


def _bayes_beta_binomial(alpha: float):
    from statscore.methods.bayes.mcmc import bayes_beta_binomial

    st.subheader("Prior: p ~ Beta(α₀, β₀)   Likelihood: X ~ Binomial(n, p)")
    c1, c2, c3, c4 = st.columns(4)
    n_trials    = int(c1.number_input("n (trials)",     min_value=1, value=20, step=1, key="bb_n"))
    n_successes = int(c2.number_input("k (successes)",  min_value=0, value=7,  step=1, key="bb_k"))
    alpha0      = c3.number_input("α₀ (prior shape1)", value=1.0, min_value=1e-9, key="bb_a0")
    beta0       = c4.number_input("β₀ (prior shape2)", value=1.0, min_value=1e-9, key="bb_b0")

    if st.button("Run Beta-Binomial"):
        try:
            r   = bayes_beta_binomial(n_trials, n_successes, alpha0=alpha0, beta0=beta0, alpha=alpha)
            pct = int((1 - alpha) * 100)
            st.subheader("Posterior Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Posterior mean",   f"{r.posterior_mean:.4f}")
            c2.metric("Posterior std",    f"{r.posterior_std:.4f}")
            c3.metric(f"{pct}% CI lower", f"{r.credible_interval[0]:.4f}")
            c4.metric(f"{pct}% CI upper", f"{r.credible_interval[1]:.4f}")
            c5, c6 = st.columns(2)
            c5.metric("αₙ", f"{r.posterior_params['alpha_n']:.4f}")
            c6.metric("βₙ", f"{r.posterior_params['beta_n']:.4f}")
            _show_fig(r.plot())
        except Exception as e:
            st.error(str(e))


def _bayes_gamma_poisson(x, alpha: float):
    from statscore.methods.bayes.mcmc import bayes_gamma_poisson

    st.subheader("Prior: λ ~ Gamma(α₀, β₀)   Likelihood: Xᵢ ~ Poisson(λ)")
    c1, c2 = st.columns(2)
    alpha0 = c1.number_input("α₀ (prior shape)", value=1.0, min_value=1e-9, key="gp_a0")
    beta0  = c2.number_input("β₀ (prior rate)",  value=1.0, min_value=1e-9, key="gp_b0")

    if st.button("Run Gamma-Poisson") and x is not None:
        try:
            r   = bayes_gamma_poisson(x, alpha0=alpha0, beta0=beta0, alpha=alpha)
            pct = int((1 - alpha) * 100)
            st.subheader("Posterior Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Posterior mean λ", f"{r.posterior_mean:.4f}")
            c2.metric("Posterior std",    f"{r.posterior_std:.4f}")
            c3.metric(f"{pct}% CI lower", f"{r.credible_interval[0]:.4f}")
            c4.metric(f"{pct}% CI upper", f"{r.credible_interval[1]:.4f}")
            c5, c6 = st.columns(2)
            c5.metric("αₙ", f"{r.posterior_params['alpha_n']:.4f}")
            c6.metric("βₙ", f"{r.posterior_params['beta_n']:.4f}")
            _show_fig(r.plot())
        except Exception as e:
            st.error(str(e))


def _mcmc_normal(x, alpha: float):
    from statscore.methods.bayes.mcmc import mcmc_normal_mean_unknown_var

    st.subheader("MCMC: Normal(μ, σ²) — Metropolis-Hastings")
    st.info("Prior: μ ~ N(μ₀, σ_μ²),  σ² ~ Inverse-Gamma(α_σ, β_σ)")
    with st.expander("Prior & sampler settings"):
        c1, c2 = st.columns(2)
        mu_pm    = c1.number_input("μ prior mean", value=0.0,   key="mcmc_n_mpm")
        mu_ps    = c2.number_input("μ prior std",  value=10.0,  min_value=1e-9, key="mcmc_n_mps")
        c3, c4 = st.columns(2)
        s_pa     = c3.number_input("σ² prior α",  value=2.0,   min_value=1e-9, key="mcmc_n_spa")
        s_pb     = c4.number_input("σ² prior β",  value=1.0,   min_value=1e-9, key="mcmc_n_spb")
        c5, c6 = st.columns(2)
        n_iter   = int(c5.number_input("Iterations", value=12000, min_value=1000, step=1000, key="mcmc_n_ni"))
        n_burnin = int(c6.number_input("Burn-in",    value=2000,  min_value=100,  step=500,  key="mcmc_n_nb"))

    if st.button("Run MCMC (Normal)") and x is not None:
        with st.spinner("Running MCMC…"):
            try:
                _show_mcmc_result(mcmc_normal_mean_unknown_var(
                    x, mu_prior_mean=mu_pm, mu_prior_std=mu_ps,
                    sigma_prior_alpha=s_pa, sigma_prior_beta=s_pb,
                    n_iter=n_iter, n_burnin=n_burnin, alpha=alpha,
                ))
            except Exception as e:
                st.error(str(e))


def _mcmc_regression(alpha: float):
    from statscore.methods.bayes.mcmc import mcmc_linear_regression

    st.subheader("MCMC: Linear Regression Y = Xβ + ε — Metropolis-Hastings")
    df = st.session_state["df"]
    input_mode = st.radio("Data source", ["From loaded dataset", "Manual entry"], horizontal=True, key="mcmc_r_src")

    X_mat: np.ndarray | None = None
    y_vec: np.ndarray | None = None
    feature_names: list[str] = []

    if input_mode == "From loaded dataset" and df is not None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        y_col     = st.selectbox("Response (y)", numeric_cols, key="mcmc_r_y")
        x_cols    = st.multiselect("Predictors (X)", [c for c in numeric_cols if c != y_col],
                                   default=[c for c in numeric_cols if c != y_col][:1], key="mcmc_r_x")
        intercept = st.checkbox("Include intercept", value=True, key="mcmc_r_ic")
        if x_cols and y_col:
            sub   = df[[y_col] + x_cols].dropna()
            y_vec = sub[y_col].values.astype(float)
            X_raw = sub[x_cols].values.astype(float)
            X_mat = np.column_stack([np.ones(len(X_raw)), X_raw]) if intercept else X_raw
            feature_names = (["(Intercept)"] + x_cols) if intercept else x_cols
    else:
        intercept = st.checkbox("Prepend intercept column", value=True, key="mcmc_r_ic2")
        x_raw = st.text_area("X matrix", height=100, key="mcmc_r_xm")
        y_raw = st.text_input("y vector", key="mcmc_r_yv")
        if x_raw and y_raw:
            try:
                X_parsed = _parse_matrix(x_raw)
                y_vec    = _parse_numbers(y_raw)
                X_mat    = (np.column_stack([np.ones(len(X_parsed)), X_parsed])
                            if intercept and not np.allclose(X_parsed[:, 0], 1.0)
                            else X_parsed)
                feature_names = [f"β{i}" for i in range(X_mat.shape[1])]
            except Exception as e:
                st.error(f"Parse error: {e}")

    with st.expander("Prior & sampler settings"):
        c1, c2 = st.columns(2)
        beta_ps  = c1.number_input("β prior std (shared)", value=10.0, min_value=1e-9, key="mcmc_r_bps")
        s_pa     = c2.number_input("σ² prior α",           value=2.0,  min_value=1e-9, key="mcmc_r_spa")
        c3, c4   = st.columns(2)
        s_pb     = c3.number_input("σ² prior β",           value=1.0,  min_value=1e-9, key="mcmc_r_spb")
        n_iter   = int(c4.number_input("Iterations",        value=15000, min_value=1000, step=1000, key="mcmc_r_ni"))
        n_burnin = int(st.number_input("Burn-in",           value=3000,  min_value=100,  step=500,  key="mcmc_r_nb"))

    if st.button("Run MCMC (Regression)") and X_mat is not None and y_vec is not None:
        with st.spinner("Running MCMC…"):
            try:
                r = mcmc_linear_regression(
                    X_mat, y_vec,
                    beta_prior_std=beta_ps,
                    sigma_prior_alpha=s_pa, sigma_prior_beta=s_pb,
                    n_iter=n_iter, n_burnin=n_burnin, alpha=alpha,
                )
                if feature_names:
                    r.param_names = feature_names + ["sigma"]
                _show_mcmc_result(r)
            except Exception as e:
                st.error(str(e))


def _show_mcmc_result(r) -> None:
    pct = int((1 - r.alpha) * 100)
    st.subheader(f"MCMC Posterior — {r.model_name}")
    c1, c2 = st.columns(2)
    c1.metric("Iterations (post burn-in)", str(len(r.posterior_samples)))
    c2.metric("Acceptance rate",           f"{r.acceptance_rate:.3f}")
    rows = [
        {
            "Parameter":        name,
            "Posterior mean":   float(r.posterior_mean[i]),
            "Posterior std":    float(r.posterior_std[i]),
            f"{pct}% CI lower": r.credible_intervals[i][0],
            f"{pct}% CI upper": r.credible_intervals[i][1],
        }
        for i, name in enumerate(r.param_names)
    ]
    st.dataframe(pd.DataFrame(rows).set_index("Parameter"), use_container_width=True)
    _show_fig(r.plot())


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def page_diagnostics():
    st.header("Diagnostics")
    diag_type = st.selectbox("Test / procedure", [
        "Shapiro-Wilk normality test",
        "Levene test (homogeneity of variances)",
        "Mean confidence interval",
    ])
    alpha = st.number_input("α", value=0.05, min_value=0.001, max_value=0.5, step=0.01, key="diag_alpha")

    if diag_type == "Shapiro-Wilk normality test":
        _diag_shapiro(alpha)
    elif diag_type == "Levene test (homogeneity of variances)":
        _diag_levene(alpha)
    elif diag_type == "Mean confidence interval":
        _diag_mean_ci(alpha)


def _diag_shapiro(alpha: float):
    from statscore.methods.diagnostics import shapiro_wilk_test
    from statscore.plots import plot_qq

    x = _sample_input("Sample x (3 ≤ n ≤ 5000)", "sw_x", "sw_x_col")
    if st.button("Run Shapiro-Wilk") and x is not None:
        try:
            r = shapiro_wilk_test(x, alpha=alpha)
            st.subheader("Shapiro-Wilk Normality Test")
            c1, c2, c3 = st.columns(3)
            c1.metric("W statistic", f"{r.statistic:.4f}")
            c2.metric("p-value",     f"{r.p_value:.4f}")
            c3.metric("n",           str(r.n))
            color = "red" if r.reject_H0 else "green"
            msg   = "Reject H₀ — not normal" if r.reject_H0 else "Fail to reject H₀ — consistent with normality"
            st.markdown(f"### :{color}[{msg}]")
            _show_fig(plot_qq(x, title="Normal Q-Q Plot (Shapiro-Wilk)"))
        except Exception as e:
            st.error(str(e))


def _diag_levene(alpha: float):
    from statscore.methods.diagnostics import levene_test
    from statscore.plots import plot_anova_groups

    df = st.session_state["df"]
    input_mode = st.radio("Data source", ["From loaded dataset", "Manual entry"],
                          horizontal=True, key="lev_src")
    data: list[np.ndarray] = []

    if input_mode == "From loaded dataset" and df is not None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        selected = st.multiselect("Columns (each = one group)", numeric_cols,
                                  default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols,
                                  key="lev_cols")
        for col in selected:
            data.append(df[col].dropna().values.astype(float))
    else:
        n_groups = int(st.number_input("Number of groups", min_value=2, max_value=8, value=3, step=1, key="lev_ng"))
        for i in range(n_groups):
            raw = st.text_input(f"Group {i+1}", key=f"lev_g{i}")
            if raw:
                with contextlib.suppress(Exception):
                    data.append(_parse_numbers(raw))

    if st.button("Run Levene test") and len(data) >= 2:
        try:
            r = levene_test(data, alpha=alpha)
            st.subheader("Levene Test for Homogeneity of Variances")
            c1, c2, c3 = st.columns(3)
            c1.metric("Levene statistic", f"{r.statistic:.4f}")
            c2.metric("p-value",          f"{r.p_value:.4f}")
            c3.metric("Groups",           str(r.n_groups))
            color = "red" if r.reject_H0 else "green"
            msg   = "Reject H₀ — variances differ" if r.reject_H0 else "Fail to reject H₀ — variances homogeneous"
            st.markdown(f"### :{color}[{msg}]")
            _show_fig(plot_anova_groups(data, title="Group Distributions (Levene Check)"))
        except Exception as e:
            st.error(str(e))


def _diag_mean_ci(alpha: float):
    from statscore.methods.diagnostics import mean_confidence_interval

    x            = _sample_input("Sample x", "mci_x", "mci_x_col")
    sigma_known  = st.checkbox("σ known (use z-interval)", value=False, key="mci_sigma_known")
    sigma        = st.number_input("σ (known std)", value=1.0, min_value=1e-9, key="mci_sigma") if sigma_known else None

    if st.button("Compute CI") and x is not None:
        try:
            r = mean_confidence_interval(x, alpha=alpha, sigma=sigma)
            st.subheader(f"{'z' if sigma_known else 't'}-Interval for the Mean")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Point estimate x̄", f"{r.point_estimate:.4f}")
            c2.metric("Margin of error",   f"{r.margin_of_error:.4f}")
            c3.metric(f"{int((1-alpha)*100)}% CI lower", f"{r.lower:.4f}")
            c4.metric(f"{int((1-alpha)*100)}% CI upper", f"{r.upper:.4f}")
            st.metric("n", str(r.n))
            _show_fig(r.plot())
        except Exception as e:
            st.error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: MULTIPLE COMPARISONS
# ─────────────────────────────────────────────────────────────────────────────

def page_multiple_comparisons():
    st.header("Multiple Comparisons")
    from statscore.methods.anova.multiple_tests import (
        anova1_ci_linear_combs,
        anova1_test_linear_combs,
    )
    from statscore.utils.enums import CorrectionMethod

    alpha = st.number_input("FWER α", value=0.05, min_value=0.001, max_value=0.5, step=0.01)
    method_map = {
        "best":       CorrectionMethod.BEST,
        "Scheffe":    CorrectionMethod.SCHEFFE,
        "Bonferroni": CorrectionMethod.BONFERRONI,
        "Sidak":      CorrectionMethod.SIDAK,
        "Tukey":      CorrectionMethod.TUKEY,
    }
    method = method_map[st.selectbox("Correction method", list(method_map))]

    df = st.session_state["df"]
    input_mode = st.radio("Group data source", ["From loaded dataset", "Manual entry"], horizontal=True)
    data: list[np.ndarray] = []

    if input_mode == "From loaded dataset" and df is not None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        selected = st.multiselect("Columns (each = one group)", numeric_cols,
                                  default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols)
        for col in selected:
            data.append(df[col].dropna().values.astype(float))
    else:
        I = int(st.number_input("Number of groups", min_value=2, max_value=8, value=3, step=1))
        for i in range(I):
            raw = st.text_input(f"Group {i+1} data", key=f"mc_g{i}")
            if raw:
                with contextlib.suppress(Exception):
                    data.append(_parse_numbers(raw))

    I = len(data)
    st.subheader("Contrast / Linear Combination Matrix C")
    st.write(f"Enter one row per comparison (I={I} coefficients per row). Rows separated by newlines, values by commas/spaces.")
    if I >= 2:
        default_C = ("1 -1" + " 0" * (I - 2) + "\n" + "0 1 -1" + " 0" * (I - 3)) if I > 2 else "1 -1"
    else:
        default_C = ""
    C_raw = st.text_area("Contrast matrix C", value=default_C, height=100)

    task  = st.radio("Task", ["Simultaneous CIs", "Simultaneous Tests"], horizontal=True)
    d_raw = ""
    if task == "Simultaneous Tests":
        d_raw = st.text_input("Hypothesized values d (one per row of C)", value=" ".join(["0"] * max(I, 1)))

    if st.button("Run") and I >= 2:
        try:
            C = _parse_matrix(C_raw)
            if C.ndim == 1:
                C = C.reshape(1, -1)

            if task == "Simultaneous CIs":
                _show_simult_ci(anova1_ci_linear_combs(data, alpha=alpha, C=C, method=method), C)
            else:
                _show_simult_test(anova1_test_linear_combs(data, alpha=alpha, C=C,
                                                           d=_parse_numbers(d_raw), method=method), C)
        except Exception as e:
            st.error(f"Error: {e}")


def _show_simult_ci(result, C) -> None:
    st.subheader(f"Simultaneous CIs — Method: {result.method_used.value}")
    rows = [
        {
            "Comparison":    f"C_{i+1}",
            "Coefficients":  str(list(C[i])),
            "Point estimate":float(result.point_estimates[i]),
            "Half-width":    float(result.half_widths[i]),
            "CI lower":      lo,
            "CI upper":      hi,
            "Contains 0":    "Yes" if lo <= 0 <= hi else "No",
        }
        for i, (lo, hi) in enumerate(result.intervals)
    ]
    st.dataframe(pd.DataFrame(rows).set_index("Comparison"), use_container_width=True)
    _show_fig(result.plot())


def _show_simult_test(result, C) -> None:
    st.subheader(f"Simultaneous Tests — Method: {result.method_used.value}")
    rows = [
        {
            "Test":         f"H₀_{i+1}",
            "Coefficients": str(list(C[i])),
            "|T|":          float(result.test_statistics[i]),
            "Critical":     float(result.critical_values[i]),
            "p-value":      float(result.p_values[i]),
            "Reject?":      "Yes" if bool(result.reject[i]) else "No",
        }
        for i in range(len(result.test_statistics))
    ]
    st.dataframe(pd.DataFrame(rows).set_index("Test"), use_container_width=True)
    _show_fig(result.plot())


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path  # noqa: E402 — needed for page_data_input file reader

PAGES = {
    "Data Input":           page_data_input,
    "ANOVA":                page_anova,
    "Significance Tests":   page_significance_tests,
    "Regression":           page_regression,
    "Diagnostics":          page_diagnostics,
    "Bayesian Inference":   page_bayesian,
    "Multiple Comparisons": page_multiple_comparisons,
}

with st.sidebar:
    st.title("statscore")
    st.caption("ENS-505 Statistical Toolbox")
    page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    df = st.session_state["df"]
    if df is not None:
        st.success(f"Dataset loaded: {df.shape[0]}×{df.shape[1]}")
    else:
        st.info("No dataset loaded")

PAGES[page]()
