"""P6 analyses: normality tests, outliers, repeated-measures ANOVA / mixed model, nested designs, correlation.

Prism conventions followed here (GraphPad Statistics Guide, Prism 10/11):
  * normality: D'Agostino-Pearson K2 (n >= 8), Shapiro-Wilk (Royston, 3 <= n < 5000), Anderson-Darling
    (raw A^2; P from the D'Agostino-Stephens formula on the modified A*^2, n >= 5), Kolmogorov-Smirnov with the
    Dallal-Wilkinson (Lilliefors) P, reported as "P>0.10" above 0.10 (n >= 5); lognormality = the same tests on the logs;
  * outliers: ROUT as published (Motulsky & Brown 2006, Lorentzian robust mean, RSDR, FDR with Q) and
    Grubbs (two-sided, one outlier or iterated);
  * repeated-measures ANOVA: Geisser-Greenhouse epsilon only changes the df of the F-test P values; without
    sphericity every multiple comparison uses only the two columns compared (paired t, df n-1); with
    sphericity the pooled residual MS is used. Blank values -> mixed-effects model (REML, compound symmetry =
    one random intercept per subject); with GG on, epsilon comes from the pairwise-complete covariance matrix and
    scales the F test df, comparisons are paired differences of the subjects present in both columns; Prism's
    integer containment denominator df are used everywhere (no Satterthwaite);
  * two-way RM: within-subject comparisons use the residual MS (df = residual df), between-subject comparisons the
    ordinary two-way error (df = N - cells); the Geisser-Greenhouse epsilon corrects the within-subject main effect;
  * nested t test / one-way ANOVA: mixed model, treatment fixed, subject (subcolumn) random, the random effect
    stays in the model when its variance is estimated as zero (Prism default); integer containment df
    (subjects - groups); for balanced data identical to a t test / ANOVA on the subcolumn means;
  * correlation: Pearson (Fisher-z CI), Spearman (exact P from the null distribution of the rank statistic up to
    n = 17, otherwise t approximation; CI by Zar's approximation only for n > 10); P values are two-tailed.

Every function returns the result contract of DESIGN.md; invalid input gives status "failed".
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import optimize, stats

from . import AnalysisKind, Param
from ..data import DataSet, Table, new_id
from .stats import (_CI, _CONTROL, _MCFAM, _Bad, _col_vec, _comp, _control_index, _dunnett_pooled,
                    _mc_adjust, _mc_table, _pairs, _pctl, p_format, p_summary)
from .stats import _kind as _stats_kind


def _lists(o):
    """ndarray -> list recursively (stats._clean handles scalars / lists / dicts; Table objects pass through)."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _lists(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_lists(v) for v in o]
    return o


def _kind(name):
    def deco(f):
        def g(table, ids, params):
            return _lists(f(table, ids, params))
        g.__name__ = f.__name__
        return _stats_kind(name)(g)
    return deco


# --------------------------------------------------------------------------- shared helpers

def _masked(table, ds):
    return np.where(ds.excluded, np.nan, ds.values)


def _need_raw(table, what):
    if table.y_format != "replicates":
        raise _Bad(f"{what} needs raw values, not mean/SD/n or mean/SEM data")


def _pooled(table, ds_id):
    """All non-blank, non-excluded values of one data set (every subcolumn)."""
    v = _masked(table, table.dataset(ds_id)).ravel()
    return v[~np.isnan(v)]


def _adjust(d, se, df, method, level, k, ctl_n=None, ns=None):
    """Multiple-comparison P / CI for t-type contrasts with their own df. Returns (p, p_adj, lo, hi)."""
    d = np.asarray(d, float)
    df = np.broadcast_to(np.asarray(df, float), d.shape).copy()
    if method == "dunnett":
        out = [np.empty(len(d)) for _ in range(4)]
        for v in np.unique(df):
            sel = df == v
            res = _dunnett_pooled(d, se, ctl_n, np.asarray(ns, float), v, level)
            for o, r in zip(out, res):
                o[sel] = r[sel]
        return tuple(out)
    return _mc_adjust(d, se, df, "none" if method == "lsd" else method, level, k)


def _welch(a, b):
    """(diff = mean a - mean b, se, df) for two samples with separate variances."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        raise _Bad("zero variance in a compared pair; the comparison is undefined")
    return a.mean() - b.mean(), se, (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))


def _paired_diff(a, b):
    d = a - b
    if len(d) < 2:
        raise _Bad("a compared pair has fewer than 2 subjects")
    sd = d.std(ddof=1)
    if sd == 0:
        raise _Bad("all paired differences of a compared pair are identical; the comparison is undefined")
    return d.mean(), sd / math.sqrt(len(d)), len(d) - 1


# --------------------------------------------------------------------------- normality

def _anderson_darling(x):
    """A^2 and the modified A*^2 = A^2 (1 + 0.75/n + 2.25/n^2) with the D'Agostino-Stephens P value."""
    n = len(x)
    z = np.clip(stats.norm.cdf((np.sort(x) - x.mean()) / x.std(ddof=1)), 1e-300, 1 - 1e-16)
    i = np.arange(1, n + 1)
    a2 = float(-n - np.mean((2 * i - 1) * (np.log(z) + np.log1p(-z[::-1]))))
    a = a2 * (1 + 0.75 / n + 2.25 / n ** 2)
    return a2, a, _ad_p(a)


def _ad_p(a):
    """D'Agostino-Stephens (1986) P value of the modified statistic A*^2."""
    if a >= 0.6:
        p = math.exp(1.2937 - 5.709 * a + 0.0186 * a * a)
    elif a > 0.34:
        p = math.exp(0.9177 - 4.279 * a - 1.38 * a * a)
    elif a > 0.2:
        p = 1 - math.exp(-8.318 + 42.796 * a - 59.938 * a * a)
    else:
        p = 1 - math.exp(-13.436 + 101.14 * a - 223.73 * a * a)
    return min(1.0, p)


def _lilliefors(x):
    """Kolmogorov-Smirnov D (mean and SD estimated) and the Dallal-Wilkinson P (Stephens' form above 0.10)."""
    n = len(x)
    z = stats.norm.cdf((np.sort(x) - x.mean()) / x.std(ddof=1))
    i = np.arange(1, n + 1)
    d = float(max((i / n - z).max(), (z - (i - 1) / n).max()))
    kd, nd = (d * (n / 100) ** 0.49, 100) if n > 100 else (d, n)
    p = math.exp(-7.01256 * kd ** 2 * (nd + 2.78019) + 2.99587 * kd * math.sqrt(nd + 2.78019) - 0.122119
                 + 0.974598 / math.sqrt(nd) + 1.67997 / nd)
    if p > 0.1:
        k = (math.sqrt(n) - 0.01 + 0.85 / math.sqrt(n)) * d
        if k <= 0.302:
            p = 1.0
        elif k <= 0.5:
            p = 2.76773 - 19.828315 * k + 80.709644 * k ** 2 - 138.55152 * k ** 3 + 81.218052 * k ** 4
        elif k <= 0.9:
            p = -4.901232 + 40.662806 * k - 97.490286 * k ** 2 + 94.029866 * k ** 3 - 32.355711 * k ** 4
        elif k <= 1.31:
            p = 6.198765 - 19.558097 * k + 23.186922 * k ** 2 - 12.234627 * k ** 3 + 2.423045 * k ** 4
        else:
            p = 0.0
    return d, float(min(1.0, max(0.0, p)))


# test id -> (label, statistic label, minimum n, maximum n)
NORMALITY_TESTS = {"dagostino": ("D'Agostino-Pearson omnibus (K2)", "K2", 8, None),
                   "anderson": ("Anderson-Darling (A2)", "A2", 5, None),
                   "shapiro": ("Shapiro-Wilk (W)", "W", 3, 4999),
                   "ks": ("Kolmogorov-Smirnov (distance)", "KS distance", 5, None)}


def _run_normality(x, test):
    if test == "dagostino":
        r = stats.normaltest(x)
        return float(r.statistic), float(r.pvalue), None
    if test == "shapiro":
        r = stats.shapiro(x)
        return float(r.statistic), float(r.pvalue), None
    if test == "anderson":
        a2, _, p = _anderson_darling(x)     # Prism reports the raw A^2; P comes from the modified A*^2
        return a2, p, None
    d, p = _lilliefors(x)
    return d, p, ">0.10" if p > 0.10 else None


@_kind("normality")
def normality(table, ids, p):
    _need_raw(table, "normality tests")
    tests = [t for t in NORMALITY_TESTS if p[t]]
    if not tests:
        raise _Bad("select at least one normality test")
    alpha, logn = p["alpha"], p["lognormal"]
    msgs, res, qq, nn, cols = [], {}, {}, {}, []
    for i in ids:
        title = table.dataset(i).title
        x = _pooled(table, i)
        nn[i] = len(x)
        if logn:
            if np.any(x <= 0):
                msgs.append(f"{title}: values <= 0 cannot be tested for lognormality")
                continue
            x = np.log(x)
        if len(x) < 3:
            msgs.append(f"{title}: {len(x)} value(s); at least 3 needed")
            continue
        if np.ptp(x) == 0:
            msgs.append(f"{title}: all values identical; normality tests are undefined")
            continue
        out = {}
        for t in tests:
            label, _, lo, hi = NORMALITY_TESTS[t]
            if len(x) < lo or (hi and len(x) > hi):
                out[t] = {"statistic": None, "p": None, "p_text": None, "passed": None,
                          "note": f"needs n >= {lo}" if len(x) < lo else f"needs n <= {hi}"}
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, pv, text = _run_normality(x, t)
            out[t] = {"statistic": stat, "p": pv, "p_text": text or p_format(pv), "passed": bool(pv > alpha), "note": None}
        res[i] = out
        xs = np.sort(x)
        z = stats.norm.ppf((np.arange(1, len(xs) + 1) - 0.5) / len(xs))
        qq[i] = {"theoretical": x.mean() + x.std(ddof=1) * z, "sample": xs, "z": z, "log": bool(logn)}
        cols.append(i)
    if not res:
        raise _Bad("; ".join(msgs) or "no data set can be tested")
    rows = [["Number of values"] + [nn[i] for i in cols]]
    for t in tests:
        label, slab, *_ = NORMALITY_TESTS[t]
        rows.append([f"{label}: {slab}"] + [res[i][t]["statistic"] for i in cols])
        rows.append([f"{label}: P value"] + [res[i][t]["p_text"] or res[i][t]["note"] for i in cols])
        rows.append([f"{label}: passed normality test (alpha={alpha:g})?"] +
                    [("n/a" if res[i][t]["passed"] is None else "Yes" if res[i][t]["passed"] else "No") for i in cols])
        rows.append([f"{label}: P value summary"] + [p_summary(res[i][t]["p"]) if res[i][t]["p"] is not None else "" for i in cols])
    title = "Lognormality tests" if logn else "Normality tests"
    return {"messages": msgs, "n": nn, "tests": res, "qq": qq, "lognormal": bool(logn), "alpha": alpha,
            "display": [{"title": title, "columns": [""] + [table.dataset(i).title for i in cols], "rows": rows}]}


# --------------------------------------------------------------------------- outliers

def _grubbs_p(n, g):
    """Two-sided P of the largest |z| (Grubbs): P = n * 2 * sf_t(t, n-2) with t from G."""
    den = (n - 1) ** 2 - n * g * g
    if den <= 0:
        return 0.0
    t = math.sqrt(n * (n - 2) * g * g / den)
    return float(min(1.0, 2 * n * stats.t.sf(t, n - 2)))


def _grubbs_g_crit(n, alpha):
    t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    return (n - 1) / math.sqrt(n) * math.sqrt(t * t / (n - 2 + t * t))


def _grubbs(y, alpha, iterative):
    """Return [(index, G, P)] of outliers (indices into y), one or iterated."""
    idx, found = np.arange(len(y)), []
    while len(idx) >= 3:
        v = y[idx]
        sd = v.std(ddof=1)
        if sd == 0:
            break
        z = np.abs(v - v.mean()) / sd
        j = int(np.argmax(z))
        pv = _grubbs_p(len(v), z[j])
        if pv >= alpha:
            break
        found.append((int(idx[j]), float(z[j]), pv))
        idx = np.delete(idx, j)
        if not iterative:
            break
    return found


def _robust_mean(y):
    """Lorentzian robust mean with RSDR = P68 * N/(N-1) re-estimated until stable (ROUT with K = 1)."""
    n, m = len(y), float(np.median(y))
    grid = np.linspace(y.min(), y.max(), 201)
    for _ in range(100):
        rsdr = _pctl(np.sort(np.abs(y - m)), 0.6827) * n / (n - 1)
        if rsdr <= 0:
            raise _Bad("the robust SD of the residuals is zero (over two thirds of the values are identical)")
        merit = np.log1p(((y[None, :] - grid[:, None]) / rsdr) ** 2).sum(axis=1)
        j = int(np.argmin(merit))
        lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
        new = float(optimize.minimize_scalar(lambda M: np.log1p(((y - M) / rsdr) ** 2).sum(), bounds=(lo, hi),
                                             method="bounded", options={"xatol": 1e-13 * (1 + abs(m))}).x)
        done = abs(new - m) <= 1e-10 * (abs(m) + rsdr)
        m = new
        if done:
            break
    return m, _pctl(np.sort(np.abs(y - m)), 0.6827) * n / (n - 1)


def _rout(y, q):
    """ROUT: returns (outlier indices, mean, rsdr, {index: P}). FDR loop over the 30 % largest residuals."""
    n = len(y)
    m, rsdr = _robust_mean(y)
    res = np.abs(y - m)
    order = np.argsort(res, kind="stable")
    pv = 2 * stats.t.sf(res[order] / rsdr, n - 1)
    for i in range(max(int(0.70 * n), 1), n + 1):
        if pv[i - 1] < q * (n - (i - 1)) / n:
            return sorted(int(k) for k in order[i - 1:]), m, rsdr, {int(order[j]): float(pv[j]) for j in range(i - 1, n)}
    return [], m, rsdr, {}


def _outliers_impl(table, ids, p):
    _need_raw(table, "outlier detection")
    method, q, alpha = p["method"], p["q"], p["alpha"]
    if not 0 < q < 1 or not 0 < alpha < 1:
        raise _Bad("Q and alpha must be between 0 and 1")
    msgs, found, detail, nn = [], {}, {}, {}
    for i in ids:
        ds = table.dataset(i)
        v = _col_vec(table, i)
        ok = ~np.isnan(v)
        rows, y = np.flatnonzero(ok), v[ok]
        nn[i] = len(y)
        found[i], detail[i] = [], []
        if len(y) < 3:
            msgs.append(f"{ds.title}: {len(y)} value(s); at least 3 needed, not tested")
            continue
        if np.ptp(y) == 0:
            msgs.append(f"{ds.title}: all values identical; no outliers")
            continue
        if method == "rout":
            idx, m, rsdr, pp = _rout(y, q)
            detail[i] = [{"row": int(rows[k]), "value": float(y[k]), "p": pp[k], "ratio": float(abs(y[k] - m) / rsdr)} for k in idx]
        else:
            hits = _grubbs(y, alpha, method == "grubbs_iterative")
            idx = sorted(k for k, _, _ in hits)
            detail[i] = [{"row": int(rows[k]), "value": float(y[k]), "p": pv, "ratio": g} for k, g, pv in hits]
        found[i] = [int(rows[k]) for k in idx]
    if not any(n_ >= 3 for n_ in nn.values()):
        raise _Bad("no data set has 3 or more values")
    # cleaned copy: outliers blanked (default) or flagged as excluded (values kept)
    dss = []
    for i in ids:
        ds = table.dataset(i)
        vals, exc = ds.values.copy(), ds.excluded.copy()
        for r in found[i]:
            if p["mark"] == "exclude":
                exc[r, 0] = True
            else:
                vals[r, 0] = np.nan
        dss.append(DataSet(ds.id, ds.title, vals, exc))
    out_t = Table(new_id("tbl"), f"{table.title} (outliers {'excluded' if p['mark'] == 'exclude' else 'removed'})", table.kind,
                  table.y_format, dss, x=None if table.x is None else table.x.copy(), x_title=table.x_title,
                  row_titles=list(table.row_titles))
    titles = {i: table.dataset(i).title for i in ids}
    lab = {"rout": f"ROUT (Q = {q:g})", "grubbs": f"Grubbs (alpha = {alpha:g})",
           "grubbs_iterative": f"Iterative Grubbs (alpha = {alpha:g})"}[method]
    stat_name = "|resid| / RSDR" if method == "rout" else "G"
    return {"messages": msgs, "n": nn, "method": method, "outliers": found, "outlier_details": detail, "output_table": out_t,
            "display": [{"title": f"Outliers: {lab}", "columns": ["Data set", "Row", "Value", stat_name, "P value"],
                         "rows": [[titles[i], d["row"], d["value"], d["ratio"], p_format(d["p"])] for i in ids for d in detail[i]]},
                        {"title": "Summary", "columns": ["Data set", "Values", "Outliers"],
                         "rows": [[titles[i], nn[i], len(found[i])] for i in ids]}]}


_outliers_kind = _kind("outliers")(_outliers_impl)


def outliers(table, ids, p):
    res = _outliers_kind(table, ids, p)
    res.setdefault("output_table", None)
    return res


# --------------------------------------------------------------------------- linear mixed model (one random intercept)

def _eff(idx, k):
    """Effects (sum-to-zero) coding of integer levels idx in 0..k-1: (len(idx), k-1)."""
    z = np.zeros((len(idx), k - 1))
    for j in range(k - 1):
        z[idx == j, j] = 1.0
    z[idx == k - 1] = -1.0
    return z


def _fit_lmm(x, y, groups):
    """REML fit of y = X b + u_subject + e (compound symmetry). Returns a dict with b, Phi, variances, criteria."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, p = x.shape
    if np.linalg.matrix_rank(x) < p:
        raise _Bad("the design is rank deficient (an empty cell or a confounded factor)")
    if n - p < 1:
        raise _Bad("no residual degrees of freedom")
    _, g = np.unique(groups, return_inverse=True)
    ns = np.bincount(g).astype(float)
    if not np.any(ns >= 2):
        raise _Bad("no subject has two or more values: subject and residual variation cannot be separated")

    def parts(delta):
        c = delta / (1 + ns * delta)

        def vinv(m):
            gs = np.zeros((len(ns), m.shape[1]))
            np.add.at(gs, g, m)
            return m - c[g][:, None] * gs[g]
        a = x.T @ vinv(x)
        b = np.linalg.solve(a, x.T @ vinv(y[:, None])[:, 0])
        r = y - x @ b
        return a, b, float(r @ vinv(r[:, None])[:, 0])

    def crit(delta):
        a, _, rv = parts(delta)
        if rv <= 0:
            return np.inf
        return (n - p) * math.log(rv / (n - p)) + float(np.log1p(ns * delta).sum()) + np.linalg.slogdet(a)[1]

    grid = np.exp(np.linspace(-10, 10, 41))
    cg = [crit(d) for d in grid]
    j = int(np.argmin(cg))
    c0 = crit(0.0)
    if not np.isfinite(min(cg + [c0])):
        raise _Bad("zero residual variance; the model cannot be fitted")
    delta = float(optimize.minimize_scalar(lambda t: crit(math.exp(t)), bounds=(math.log(grid[max(j - 1, 0)]),
                                           math.log(grid[min(j + 1, len(grid) - 1)])), method="bounded",
                                           options={"xatol": 1e-10}).x)
    delta = math.exp(delta)
    if c0 <= crit(delta) + 1e-12 or delta < 1e-9:
        delta = 0.0
    a, b, rv = parts(delta)
    s2e = rv / (n - p)
    const = (n - p) * (1 + math.log(2 * math.pi))
    zs = np.zeros((n, len(ns)))
    zs[np.arange(n), g] = 1.0
    btw = np.array([np.allclose(x[:, j], (np.bincount(g, x[:, j]) / ns)[g]) for j in range(p)])   # constant within a subject
    df_res = n - np.linalg.matrix_rank(np.column_stack([x, zs]))
    df_btw = len(ns) - np.linalg.matrix_rank(x[:, btw]) if btw.any() else 0
    fit = {"x": x, "y": y, "g": g, "ns": ns, "n": n, "p": p, "beta": b, "s2e": s2e, "s2s": delta * s2e,
           "boundary": delta == 0.0, "phi": np.linalg.inv(a) * s2e, "btw": btw, "df_btw": float(df_btw), "df_res": float(df_res),
           "m2l": crit(delta) + const, "m2l_ols": c0 + const}
    return fit


def _contain_df(fit, cols):
    """Prism's integer denominator df (containment): effects constant within a subject are tested against the
    subjects (subjects - between-subject parameters), all others against the residual (N - rank[X Z])."""
    df = fit["df_btw"] if all(fit["btw"][j] for j in cols) else fit["df_res"]
    if df < 1:
        raise _Bad("no denominator degrees of freedom for this effect")
    return df


def _lmm_contrast(fit, ell):
    """(estimate, se, df) of a linear combination of the fixed effects; df by containment."""
    ell = np.asarray(ell, float)
    return (float(ell @ fit["beta"]), float(math.sqrt(ell @ fit["phi"] @ ell)),
            _contain_df(fit, np.flatnonzero(np.abs(ell) > 1e-12)))


def _lmm_f(fit, cols):
    """F test that the coefficients in ``cols`` are all zero; denominator df by containment."""
    q = len(cols)
    ll = np.zeros((q, fit["p"]))
    ll[np.arange(q), cols] = 1.0
    est = ll @ fit["beta"]
    lam = ll @ fit["phi"] @ ll.T
    f = float(est @ np.linalg.solve(lam, est) / q)
    df2 = _contain_df(fit, cols)
    return f, q, df2, float(stats.f.sf(f, q, df2))


def _random_block(fit):
    lrt = max(fit["m2l_ols"] - fit["m2l"], 0.0)
    return {"subject_variance": fit["s2s"], "subject_sd": math.sqrt(fit["s2s"]), "residual_variance": fit["s2e"],
            "residual_sd": math.sqrt(fit["s2e"]), "reml_criterion": fit["m2l"] / 2, "lrt_chi2": lrt,
            "lrt_df": 1, "lrt_p": float(stats.chi2.sf(lrt, 1)), "variance_at_zero": bool(fit["boundary"])}


def _random_display(rb, subject_label):
    return [{"title": "Random effects", "columns": ["Effect", "Variance", "SD"],
             "rows": [[subject_label, rb["subject_variance"], rb["subject_sd"]], ["Residual", rb["residual_variance"], rb["residual_sd"]]]},
            {"title": f"Test of the random effect: {subject_label}", "columns": ["Parameter", "Value"],
             "rows": [["Chi-square (1 df), full vs. model without the random effect", rb["lrt_chi2"]],
                      ["P value", p_format(rb["lrt_p"])], ["REML criterion (Prism scale: half of -2 log likelihood)", rb["reml_criterion"]]]}]


def _mc_from_contrasts(fit, xcell_rows, pairs, ids, method, level, ctl, k, ns_subj):
    """Pairwise comparisons of group means of an LMM. xcell_rows[g] = fixed-effect row of group g."""
    est = [_lmm_contrast(fit, xcell_rows[i] - xcell_rows[j]) for i, j in pairs]
    d, se, df = (np.array(v) for v in zip(*est))
    ctl_n = ns_subj.min() if ctl is not None else None     # Dunnett: smallest group size takes the control's place (Prism)
    ns = np.array([ns_subj[j] for _, j in pairs], float)
    pu, pa, lo, hi = _adjust(d, se, df, method, level, k, ctl_n, ns)
    comps = [_comp(f"{ids[i]}__{ids[j]}", ids[i], ids[j], d[q], None if lo is None else lo[q],
                   None if hi is None else hi[q], pu[q], pa[q], se=se[q], df=df[q]) for q, (i, j) in enumerate(pairs)]
    return comps


# --------------------------------------------------------------------------- repeated-measures ANOVA core

def _splitplot(y):
    """Balanced complete split-plot ANOVA. y: (B groups, n subjects, W within levels); B = 1 is a one-way RM ANOVA.
    Returns SS/df dict and the Geisser-Greenhouse epsilon of the within factor (pooled within-group covariance)."""
    b, n, w = y.shape
    gm = y.mean()
    mg, mw, mgw, ms = y.mean((1, 2)), y.mean((0, 1)), y.mean(1), y.mean(2)
    ss = {"group": w * n * ((mg - gm) ** 2).sum(), "subjects": w * ((ms - mg[:, None]) ** 2).sum(),
          "within": b * n * ((mw - gm) ** 2).sum(),
          "interaction": n * ((mgw - mg[:, None] - mw[None, :] + gm) ** 2).sum(), "total": ((y - gm) ** 2).sum()}
    ss["residual"] = ss["total"] - ss["group"] - ss["subjects"] - ss["within"] - ss["interaction"]
    df = {"group": b - 1, "subjects": b * (n - 1), "within": w - 1, "interaction": (b - 1) * (w - 1),
          "residual": b * (n - 1) * (w - 1), "total": b * n * w - 1}
    yc = (y - mgw[:, None, :]).reshape(-1, w)
    h = np.eye(w) - 1.0 / w
    sd = h @ (yc.T @ yc / (b * (n - 1))) @ h
    tr2 = float(np.trace(sd @ sd))
    eps = float(np.trace(sd) ** 2 / ((w - 1) * tr2)) if tr2 > 0 and w > 2 else 1.0
    return ss, df, min(eps, 1.0)


def _f_p(num, den, d1, d2):
    f = num / den
    return f, float(stats.f.sf(f, d1, d2))


def _gg_epsilon_pairwise(m):
    """Geisser-Greenhouse epsilon from the covariance matrix of the columns estimated with the subjects present
    in both columns of each pair (deviations from the column means over all available values); equals the usual
    epsilon for complete data. Prism's rule when blank values are present."""
    k = m.shape[1]
    ok = ~np.isnan(m)
    dev = m - np.nanmean(m, axis=0)
    s = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            both = ok[:, i] & ok[:, j]
            if both.sum() < 2:
                raise _Bad("two columns share fewer than 2 subjects; the covariance matrix is undefined")
            s[i, j] = (dev[both, i] * dev[both, j]).sum() / (both.sum() - 1)
    sd = (np.eye(k) - 1.0 / k) @ s @ (np.eye(k) - 1.0 / k)
    tr2 = float(np.trace(sd @ sd))
    # pairwise covariances need not be positive semi-definite: keep epsilon inside its theoretical range [1/(k-1), 1]
    return min(max(float(np.trace(sd) ** 2 / ((k - 1) * tr2)), 1.0 / (k - 1)), 1.0) if tr2 > 0 and k > 2 else 1.0


def _rm_comparisons_1way(m, ids, titles, p, ms_res, n, gg):
    """Pairwise comparisons after a one-way RM ANOVA (columns of m = conditions; NaN = blank, only with gg)."""
    k = m.shape[1]
    method, level = p["multiple_comparisons"], p["ci_level"]
    fam = "all_pairs" if method == "tukey" else "vs_control" if method == "dunnett" else p["mc_family"]
    ctl = _control_index(ids, p) if fam == "vs_control" else None
    pairs = _pairs(k, fam, ctl)
    if gg:  # no sphericity: paired differences of the subjects present in both columns (paired t, df n pairs - 1)
        est = []
        for i, j in pairs:
            ok = ~(np.isnan(m[:, i]) | np.isnan(m[:, j]))
            est.append(_paired_diff(m[ok, i], m[ok, j]))
        d, se, df = (np.array(v) for v in zip(*est))
        # Dunnett: contrast correlations lambda_j^2 = n_j / (n_j + n_min) with n_j the pairs of the contrast and n_min the
        # smallest column count (Prism; 0.5 for complete data), each contrast with its own df
        pu, pa, lo, hi = _adjust(d, se, df, method, level, k, float(np.min(np.sum(~np.isnan(m), axis=0))), df + 1.0)
    else:
        means = m.mean(0)
        d = np.array([means[i] - means[j] for i, j in pairs])
        se = np.full(len(pairs), math.sqrt(2 * ms_res / n))
        df = (n - 1) * (k - 1)
        pu, pa, lo, hi = _adjust(d, se, df, method, level, k, n, np.full(len(pairs), float(n)))
    comps = [_comp(f"{ids[i]}__{ids[j]}", ids[i], ids[j], d[q], None if lo is None else lo[q],
                   None if hi is None else hi[q], pu[q], pa[q], se=se[q], df=float(np.ravel(df)[q] if np.ndim(df) else df))
             for q, (i, j) in enumerate(pairs)]
    return comps


@_kind("rm_one_way_anova")
def rm_one_way_anova(table, ids, p):
    _need_raw(table, "repeated-measures ANOVA")
    k = len(ids)
    if k < 2:
        raise _Bad("select at least 2 data sets (the repeated conditions)")
    titles = {i: table.dataset(i).title for i in ids}
    mat = np.column_stack([_col_vec(table, i) for i in ids])
    mat = mat[~np.isnan(mat).all(axis=1)]
    nn = {i: int((~np.isnan(mat[:, j])).sum()) for j, i in enumerate(ids)}
    missing = bool(np.isnan(mat).any())
    model = p["model"]
    if model == "anova" and missing:
        raise _Bad("blank values: repeated-measures ANOVA needs complete rows; use the mixed-effects model")
    mixed = model == "mixed" or (model == "auto" and missing)
    msgs, method, level = [], p["multiple_comparisons"], p["ci_level"]
    res = {"n": nn, "model": "mixed" if mixed else "anova"}
    if not mixed:
        n = len(mat)
        if n < 2:
            raise _Bad(f"{n} subject row(s); at least 2 needed")
        ss, df, eps = _splitplot(mat[None])
        if ss["residual"] <= 0:
            raise _Bad("zero residual variation; F is undefined")
        gg = p["sphericity"] == "gg"
        ms = {s: ss[s] / df[s] for s in ("within", "subjects", "residual")}
        f, p_unc = _f_p(ms["within"], ms["residual"], df["within"], df["residual"])
        d1, d2 = (eps * df["within"], eps * df["residual"]) if gg else (df["within"], df["residual"])
        pval = float(stats.f.sf(f, d1, d2))
        fm, pm = _f_p(ms["subjects"], ms["residual"], df["subjects"], df["residual"])
        r2t = ss["within"] / (ss["within"] + ss["residual"])
        res.update(n_subjects=n, epsilon=eps, sphericity="gg" if gg else "assumed", f=f, df_num=d1, df_den=d2, p=pval,
                   anova={"ss_treatment": ss["within"], "df_treatment": df["within"], "ms_treatment": ms["within"],
                          "f": f, "p": pval, "p_uncorrected": p_unc, "ss_subjects": ss["subjects"], "df_subjects": df["subjects"],
                          "ms_subjects": ms["subjects"], "f_matching": fm, "p_matching": pm, "ss_residual": ss["residual"],
                          "df_residual": df["residual"], "ms_residual": ms["residual"], "ss_total": ss["total"],
                          "df_total": df["total"], "r2_treatment": r2t, "r2_matching": ss["subjects"] / ss["total"]},
                   means={i: float(mat[:, j].mean()) for j, i in enumerate(ids)})
        fl = f"F ({d1:.4g}, {d2:.4g}) = {f:.4g}" if gg else f"F ({d1}, {d2}) = {f:.4g}"
        res["display"] = [
            {"title": "Repeated measures one-way ANOVA", "columns": ["Source of variation", "SS", "DF", "MS", "F (DFn, DFd)", "P value", "Summary"],
             "rows": [["Treatment (between columns)", ss["within"], df["within"], ms["within"], fl, p_format(pval), p_summary(pval)],
                      ["Individual (between rows)", ss["subjects"], df["subjects"], ms["subjects"],
                       f"F ({df['subjects']}, {df['residual']}) = {fm:.4g}", p_format(pm), p_summary(pm)],
                      ["Residual (random)", ss["residual"], df["residual"], ms["residual"], None, None, None],
                      ["Total", ss["total"], df["total"], None, None, None, None]]},
            {"title": "Sphericity and effect sizes", "columns": ["Parameter", "Value"],
             "rows": [["Geisser-Greenhouse epsilon", eps], ["Sphericity", "not assumed (Geisser-Greenhouse)" if gg else "assumed"],
                      ["P value without correction", p_format(p_unc)], ["Treatment R squared", r2t],
                      ["Matching R squared", ss["subjects"] / ss["total"]],
                      ["Was the matching effective? (P value)", p_format(pm)]]}]
        if method != "none":
            comps = _rm_comparisons_1way(mat, ids, titles, p, ms["residual"], n, gg)
    else:
        long = [(r, j, v) for r, row in enumerate(mat) for j, v in enumerate(row) if not np.isnan(v)]
        subj, grp, y = (np.array(c) for c in zip(*long))
        x = np.column_stack([np.ones(len(y)), _eff(grp, k)])
        fit = _fit_lmm(x, y, subj)
        f, d1, d2, pval = _lmm_f(fit, list(range(1, k)))
        gg = p["sphericity"] == "gg"
        eps = _gg_epsilon_pairwise(mat) if gg else 1.0
        if gg:
            d1, d2 = eps * d1, eps * d2
            pval = float(stats.f.sf(f, d1, d2))
        rb = _random_block(fit)
        xrows = np.array([np.r_[1.0, _eff(np.array([j]), k)[0]] for j in range(k)])
        mu = xrows @ fit["beta"]
        msgs.append("blank values: mixed-effects model (REML, compound symmetry) fitted instead of repeated-measures ANOVA; "
                    + ("Geisser-Greenhouse epsilon from the pairwise-complete covariance matrix, comparisons from paired differences"
                       if gg else "sphericity assumed"))
        if fit["boundary"]:
            msgs.append("the subject variance is estimated as zero; the random effect stays in the model (integer containment df)")
        res.update(df_method="containment", n_subjects=len(np.unique(subj)), f=f, df_num=d1, df_den=d2, p=pval,
                   sphericity="gg" if gg else "assumed", epsilon=eps,
                   means={i: float(mu[j]) for j, i in enumerate(ids)},
                   mixed={"fixed": {"treatment": {"f": f, "df_num": d1, "df_den": d2, "p": pval}}, "random": rb,
                          "matching": {"chi2": rb["lrt_chi2"], "df": 1, "p": rb["lrt_p"]}})
        res["display"] = [{"title": "Mixed-effects model (REML): fixed effects (type III)", "columns": ["Source", "F (DFn, DFd)", "P value", "Summary"],
                           "rows": [["Treatment (columns)", f"F ({d1:.4g}, {d2:.4g}) = {f:.4g}", p_format(pval), p_summary(pval)]]}] + \
                         ([{"title": "Sphericity", "columns": ["Parameter", "Value"], "rows": [["Geisser-Greenhouse epsilon", eps]]}] if gg else []) + \
                         _random_display(rb, "Subject (rows)")
        if method != "none":
            fam = "all_pairs" if method == "tukey" else "vs_control" if method == "dunnett" else p["mc_family"]
            ctl = _control_index(ids, p) if fam == "vs_control" else None
            ns_subj = np.array([nn[i] for i in ids], float)
            comps = (_rm_comparisons_1way(mat, ids, titles, p, None, None, True) if gg else
                     _mc_from_contrasts(fit, xrows, _pairs(k, fam, ctl), ids, method, level, ctl, k, ns_subj))
    if method != "none":
        res["comparisons"], res["mc_method"], res["diff_label"] = comps, method, "A - B"
        res["display"].append(_mc_table(f"Multiple comparisons ({method})", comps, titles, "Mean diff."))
    res["messages"] = msgs
    return res


# --------------------------------------------------------------------------- repeated-measures two-way ANOVA

def _xrow2(r, c, nr, nc):
    a, b = _eff(np.array([r]), nr)[0] if nr > 1 else np.zeros(0), _eff(np.array([c]), nc)[0] if nc > 1 else np.zeros(0)
    return np.r_[1.0, a, b, np.outer(a, b).ravel()]


@_kind("rm_two_way_anova")
def rm_two_way_anova(table, ids, p):
    _need_raw(table, "repeated-measures ANOVA")
    C = len(ids)
    rows_design = p["design"] == "rows"
    masked = [_masked(table, table.dataset(i)) for i in ids]
    used = [r for r in range(table.n_rows) if any(not np.isnan(m[r]).all() for m in masked)]
    R = len(used)
    if R < 2 or C < 2:
        raise _Bad("two-way ANOVA needs at least 2 rows and 2 data sets (columns)")
    nsub = max(m.shape[1] for m in masked)
    rt = [table.row_titles[r] if r < len(table.row_titles) and table.row_titles[r] else f"Row {r + 1}" for r in used]
    ct = {i: table.dataset(i).title for i in ids}
    # subject -> {(row position, data set position): value}; between level of the subject
    obs, between = {}, {}
    for c in range(C):
        for rp, r in enumerate(used):
            for s in range(masked[c].shape[1]):
                v = masked[c][r, s]
                if np.isnan(v):
                    continue
                sid, b = ((c, s), c) if rows_design else ((rp, s), rp)
                obs.setdefault(sid, {})[(rp, c)] = float(v)
                between[sid] = b
    cells = {(rp, c) for o in obs.values() for (rp, c) in o}
    if len(cells) < R * C:
        e = next((rp, c) for rp in range(R) for c in range(C) if (rp, c) not in cells)
        raise _Bad(f"empty cell: {rt[e[0]]} / {ct[ids[e[1]]]}; every row x column cell needs a value")
    W = R if rows_design else C
    B = C if rows_design else R
    groups = {b: [s for s in obs if between[s] == b] for b in range(B)}
    complete = all(len(o) == W for o in obs.values()) and len({len(v) for v in groups.values()}) == 1
    if p["model"] == "anova" and not complete:
        raise _Bad("blank values or unequal subject numbers: repeated-measures ANOVA needs a complete balanced design; "
                   "use the mixed-effects model")
    mixed = p["model"] == "mixed" or (p["model"] == "auto" and not complete)
    msgs, level, mc, corr, fam = [], p["ci_level"], p["multiple_comparisons"], p["correction"], p["family"]
    gg = p["sphericity"] == "gg" and not mixed
    names = ("Interaction", "Row Factor", "Column Factor")
    nn_cells = {(rp, c): sum((rp, c) in o for o in obs.values()) for rp in range(R) for c in range(C)}
    mm = {k: float(np.mean([o[k] for o in obs.values() if k in o])) for k in nn_cells}
    res = {"n": {i: int(sum(v for (rp, c), v in nn_cells.items() if c == j)) for j, i in enumerate(ids)},
           "model": "mixed" if mixed else "anova", "design": p["design"], "n_subjects": len(obs),
           "cell_means": {rt[rp]: {i: mm[(rp, c)] for c, i in enumerate(ids)} for rp in range(R)}}
    # ---- omnibus tests
    if not mixed:
        y = np.empty((B, len(groups[0]), W))
        for b in range(B):
            for q, s in enumerate(groups[b]):
                y[b, q] = [obs[s][(w, b) if rows_design else (b, w)] for w in range(W)]
        if y.shape[1] < 2:
            raise _Bad("at least 2 subjects per group are needed")
        ss, df, eps = _splitplot(y)
        if ss["residual"] <= 0:
            raise _Bad("zero residual variation; F is undefined")
        ms = {s: ss[s] / df[s] for s in ("group", "subjects", "within", "interaction", "residual")}
        wlab, blab = ("Row Factor", "Column Factor") if rows_design else ("Column Factor", "Row Factor")
        tests = {blab: (ms["group"], ms["subjects"], df["group"], df["subjects"], ss["group"]),
                 wlab: (ms["within"], ms["residual"], df["within"], df["residual"], ss["within"]),
                 "Interaction": (ms["interaction"], ms["residual"], df["interaction"], df["residual"], ss["interaction"])}
        terms, trows = {}, []
        sst = ss["total"]
        for lab in names:
            num, den, d1, d2, s_ = tests[lab]
            f, pu = _f_p(num, den, d1, d2)
            corrected = gg and lab == wlab
            e1, e2 = (eps * d1, eps * d2) if corrected else (d1, d2)
            pv = float(stats.f.sf(f, e1, e2)) if corrected else pu
            terms[lab] = {"ss": s_, "df": d1, "ms": num, "f": f, "p": pv, "p_uncorrected": pu, "df_num": e1, "df_den": e2,
                          "pct_total": 100 * s_ / sst}
            fl = f"F ({e1:.4g}, {e2:.4g}) = {f:.4g}" if corrected else f"F ({d1}, {d2}) = {f:.4g}"
            trows.append([lab, 100 * s_ / sst, s_, d1, num, fl, p_format(pv), p_summary(pv)])
        fm, pm = _f_p(ms["subjects"], ms["residual"], df["subjects"], df["residual"])
        terms["Subjects (matching)"] = {"ss": ss["subjects"], "df": df["subjects"], "ms": ms["subjects"], "f": fm, "p": pm,
                                        "pct_total": 100 * ss["subjects"] / sst}
        terms["Residual"] = {"ss": ss["residual"], "df": df["residual"], "ms": ms["residual"], "pct_total": 100 * ss["residual"] / sst}
        trows += [["Subjects (matching)", 100 * ss["subjects"] / sst, ss["subjects"], df["subjects"], ms["subjects"],
                   f"F ({df['subjects']}, {df['residual']}) = {fm:.4g}", p_format(pm), p_summary(pm)],
                  ["Residual", 100 * ss["residual"] / sst, ss["residual"], df["residual"], ms["residual"], None, None, None],
                  ["Total", 100.0, sst, df["total"], None, None, None, None]]
        res.update(terms=terms, ss_total=sst, df_total=df["total"], ss_type="balanced", epsilon=eps,
                   sphericity="gg" if gg else "assumed")
        res["display"] = [{"title": "Two-way ANOVA with repeated measures" + (" (Geisser-Greenhouse)" if gg else ""),
                           "columns": ["Source of variation", "% of total variation", "SS", "DF", "MS", "F (DFn, DFd)", "P value", "Summary"],
                           "rows": trows},
                          {"title": "Sphericity", "columns": ["Parameter", "Value"],
                           "rows": [["Geisser-Greenhouse epsilon", eps], ["Sphericity", "not assumed (Geisser-Greenhouse)" if gg else "assumed"]]}]
    # ---- mixed-effects fit (also gives contrasts when sphericity is assumed)
    fit = None
    n_obs = sum(len(o) for o in obs.values())
    if mixed or (mc == "main_column" and not gg):
        sids = sorted(obs)
        ys, xs, gr = [], [], []
        for q, s in enumerate(sids):
            for (rp, c), v in obs[s].items():
                ys.append(v)
                xs.append(_xrow2(rp, c, R, C))
                gr.append(q)
        fit = _fit_lmm(np.array(xs), np.array(ys), np.array(gr))
        if fit["boundary"]:
            msgs.append("the subject variance is estimated as zero; denominator df = N - parameters")
    if mixed:
        a_n, b_n = R - 1, C - 1
        cols = {"Row Factor": list(range(1, 1 + a_n)), "Column Factor": list(range(1 + a_n, 1 + a_n + b_n)),
                "Interaction": list(range(1 + a_n + b_n, 1 + a_n + b_n + a_n * b_n))}
        terms, trows = {}, []
        for lab in names:
            f, d1, d2, pv = _lmm_f(fit, cols[lab])
            terms[lab] = {"f": f, "df_num": d1, "df_den": d2, "p": pv}
            trows.append([lab, f"F ({d1}, {d2:.4g}) = {f:.4g}", p_format(pv), p_summary(pv)])
        rb = _random_block(fit)
        if not complete:
            msgs.append("blank values or unequal subject numbers: mixed-effects model (REML, compound symmetry) fitted "
                        "instead of repeated-measures ANOVA; no Geisser-Greenhouse correction")
        res.update(df_method="containment", terms=terms, random=rb, matching={"chi2": rb["lrt_chi2"], "df": 1, "p": rb["lrt_p"]}, sphericity="assumed")
        res["display"] = [{"title": "Mixed-effects model (REML): fixed effects (type III)",
                           "columns": ["Source", "F (DFn, DFd)", "P value", "Summary"], "rows": trows}] + _random_display(rb, "Subject")
    if mc == "none":
        res["messages"] = msgs
        return res
    # ---- multiple comparisons (cell-mean contrasts)
    byrow = mc == "within_rows"
    if mc in ("within_rows", "within_columns"):
        outer, inner = (R, C) if byrow else (C, R)
        if inner == 2 and fam == "per_row" and corr != "tukey":    # one comparison per row: a single family (Tukey keeps its k = 2 families)
            fam = "all"
        if corr == "auto":
            corr = "bonferroni" if inner == 2 else "tukey"
        if corr == "tukey" and fam == "all":
            raise _Bad("Tukey applies to separate families (one per row or per column); "
                       "use family 'per_row' or a Sidak / Bonferroni correction")
        ip = list(combinations(range(inner), 2))
        flat = [(o, i, j) for o in range(outer) for i, j in ip]
        cell = (lambda o, i: (o, i)) if byrow else (lambda o, i: (i, o))
        defs = [([cell(o, i)], [cell(o, j)]) for o, i, j in flat]
        k_tuk = inner
        within_type = (not rows_design) if byrow else rows_design     # does the comparison contrast levels of the repeated factor?
    elif mc == "main_column":
        if corr == "auto":
            corr = "tukey"
        cp = list(combinations(range(C), 2))
        defs = [([(rp, i) for rp in range(R)], [(rp, j) for rp in range(R)]) for i, j in cp]
        k_tuk = C
        within_type = None
    else:
        raise _Bad(f"unknown multiple comparisons choice {mc!r}")
    if gg:   # separate variances: paired t when both sides come from the same subjects, Welch otherwise
        est = []
        for ca, cb in defs:
            sa = {s: np.mean([o[c] for c in ca]) for s, o in obs.items() if all(c in o for c in ca)}
            sb = {s: np.mean([o[c] for c in cb]) for s, o in obs.items() if all(c in o for c in cb)}
            if set(sa) == set(sb):
                ks = sorted(sa)
                dd, se, dfv = _paired_diff(np.array([sa[s] for s in ks]), np.array([sb[s] for s in ks]))
            else:
                dd, se, dfv = _welch(np.array(list(sa.values())), np.array(list(sb.values())))
            est.append((dd, se, dfv))
    elif not mixed and within_type is not None:
        # balanced split-plot error terms (Prism): within-subject contrasts use the residual MS; between-subject
        # contrasts the pooled error of the ordinary two-way model (subjects + residual), df = N - cells
        nb = y.shape[1]
        ms_e, df_e = (ms["residual"], df["residual"]) if within_type else \
            ((ss["subjects"] + ss["residual"]) / (df["subjects"] + df["residual"]), df["subjects"] + df["residual"])
        est = [(np.mean([mm[c] for c in ca]) - np.mean([mm[c] for c in cb]), math.sqrt(2 * ms_e / nb), float(df_e)) for ca, cb in defs]
    else:
        est = []
        for ca, cb in defs:
            ell = np.mean([_xrow2(rp, c, R, C) for rp, c in ca], axis=0) - np.mean([_xrow2(rp, c, R, C) for rp, c in cb], axis=0)
            e = _lmm_contrast(fit, ell)
            # between-subject cell contrasts: error df of the ordinary two-way model, N - number of cells
            est.append((e[0], e[1], float(n_obs - R * C)) if within_type is False else e)
    d, se, dfv = (np.array(v) for v in zip(*est))
    if mc == "main_column":
        fams = [list(range(len(defs)))]
    else:
        per = len(ip)
        fams = [list(range(len(defs)))] if fam == "all" else [list(range(o * per, (o + 1) * per)) for o in range(outer)]
    out = np.empty((len(defs), 4))
    for idx in fams:
        out[idx] = np.column_stack(_mc_adjust(d[idx], se[idx], dfv[idx], corr, level, k_tuk))
    if mc == "within_rows":
        comps = [_comp(f"r{used[o]}:{ids[i]}__{ids[j]}", ids[i], ids[j], d[q], out[q, 2], out[q, 3], out[q, 0], out[q, 1],
                       row=used[o], se=se[q], df=dfv[q]) for q, (o, i, j) in enumerate(flat)]
        title = f"Multiple comparisons: columns within each row ({corr}, {'one family' if fam == 'all' else 'one family per row'})"
    elif mc == "within_columns":
        comps = [_comp(f"c{ids[o]}:r{used[i]}__r{used[j]}", f"r{used[i]}", f"r{used[j]}", d[q], out[q, 2], out[q, 3], out[q, 0],
                       out[q, 1], col=ids[o], se=se[q], df=dfv[q]) for q, (o, i, j) in enumerate(flat)]
        title = f"Multiple comparisons: rows within each column ({corr}, {'one family' if fam == 'all' else 'one family per column'})"
    else:
        comps = [_comp(f"{ids[i]}__{ids[j]}", ids[i], ids[j], d[q], out[q, 2], out[q, 3], out[q, 0], out[q, 1], se=se[q], df=dfv[q])
                 for q, (i, j) in enumerate(cp)]
        title = f"Multiple comparisons: column means ({corr})"
    res["comparisons"], res["diff_label"] = comps, "A - B"
    rt_by_row = {used[r]: rt[r] for r in range(R)}
    if mc == "within_columns":
        res["display"].append({"title": title, "columns": ["Column", "Comparison", "Mean diff.", "CI low", "CI high", "Summary", "Adjusted P"],
                               "rows": [[ct[c["col"]], f"{rt_by_row[int(c['a'][1:])]} vs. {rt_by_row[int(c['b'][1:])]}", c["diff"],
                                         c["ci_low"], c["ci_high"], c["summary"], p_format(c["p_adj"])] for c in comps]})
    else:
        res["display"].append(_mc_table(title, comps, ct, "Mean diff.", with_row=(mc == "within_rows"), row_titles=rt_by_row))
    res["messages"] = msgs
    return res


# --------------------------------------------------------------------------- nested t test / one-way ANOVA

def _nested_long(table, ids):
    """Nested table -> (values, group index, subject index) of the non-blank, non-excluded cells."""
    _need_raw(table, "nested analyses")
    y, g, s, nsub = [], [], [], []
    sid = 0
    for gi, i in enumerate(ids):
        v = _masked(table, table.dataset(i))
        cnt = 0
        for j in range(v.shape[1]):
            col = v[:, j][~np.isnan(v[:, j])]
            if len(col):
                y += list(col)
                g += [gi] * len(col)
                s += [sid] * len(col)
                sid += 1
                cnt += 1
        nsub.append(cnt)
    return np.array(y), np.array(g), np.array(s), nsub


def _nested_core(table, ids, p, kmin):
    k = len(ids)
    if k < kmin:
        raise _Bad(f"select at least {kmin} data sets")
    y, g, s, nsub = _nested_long(table, ids)
    titles = {i: table.dataset(i).title for i in ids}
    if min(nsub) < 1:
        raise _Bad("every data set needs at least one subject (subcolumn with values)")
    if sum(nsub) - k < 1:
        raise _Bad("not enough subjects: the number of subjects must exceed the number of data sets")
    if np.ptp(y) == 0:
        raise _Bad("all values identical; the test is undefined")
    x = np.column_stack([np.ones(len(y)), _eff(g, k)])
    fit = _fit_lmm(x, y, s)
    xrows = np.array([np.r_[1.0, _eff(np.array([j]), k)[0]] for j in range(k)])
    nvals = {i: int((g == j).sum()) for j, i in enumerate(ids)}
    return fit, xrows, titles, nsub, nvals


@_kind("nested_t_test")
def nested_t_test(table, ids, p):
    if len(ids) != 2:
        raise _Bad("a nested t test compares exactly 2 data sets")
    fit, xrows, titles, nsub, nvals = _nested_core(table, ids, p, 2)
    a_id, b_id = ids
    est, se, df = _lmm_contrast(fit, xrows[1] - xrows[0])    # B - A
    t = abs(est / se)       # Prism reports t unsigned
    pval = float(2 * stats.t.sf(t, df))
    h = stats.t.ppf((1 + p["ci_level"]) / 2, df) * se
    f, _, _, _ = _lmm_f(fit, [1])
    rb = _random_block(fit)
    mu = xrows @ fit["beta"]
    level = p["ci_level"]
    msgs = ["the subject variance is estimated as zero; the random effect stays in the model (integer containment df)"] if fit["boundary"] else []
    rows = [["P value (two-tailed)", p_format(pval)], ["P value summary", p_summary(pval)], ["t", t], ["df", df], ["F (t squared)", f],
            [f"Mean of {titles[a_id]}", mu[0]], [f"Mean of {titles[b_id]}", mu[1]],
            [f"Difference ({titles[b_id]} - {titles[a_id]})", est], ["SE of difference", se],
            [f"{level:.0%} CI of difference: low", est - h], [f"{level:.0%} CI of difference: high", est + h]]
    return {"messages": msgs, "df_method": "containment", "n": nvals, "n_subjects": dict(zip(ids, nsub)), "t": t, "df": df, "f": f, "p": pval, "difference": est,
            "se": se, "ci_low": est - h, "ci_high": est + h, "means": {a_id: mu[0], b_id: mu[1]}, "random": rb,
            "diff_label": "B - A", "comparisons": [_comp(f"{a_id}__{b_id}", b_id, a_id, est, est - h, est + h, pval, pval, t=t, df=df, se=se)],
            "display": [{"title": "Nested t test", "columns": ["Parameter", "Value"], "rows": rows}] + _random_display(rb, "Subcolumns (within columns)")}


@_kind("nested_one_way_anova")
def nested_one_way_anova(table, ids, p):
    fit, xrows, titles, nsub, nvals = _nested_core(table, ids, p, 2)
    k, level, method = len(ids), p["ci_level"], p["multiple_comparisons"]
    f, d1, d2, pval = _lmm_f(fit, list(range(1, k)))
    rb = _random_block(fit)
    mu = xrows @ fit["beta"]
    msgs = ["the subject variance is estimated as zero; the random effect stays in the model (integer containment df)"] if fit["boundary"] else []
    res = {"messages": msgs, "df_method": "containment", "n": nvals, "n_subjects": dict(zip(ids, nsub)), "f": f, "df_num": d1, "df_den": d2, "p": pval,
           "means": dict(zip(ids, mu)), "random": rb,
           "display": [{"title": "Nested one-way ANOVA", "columns": ["Source", "F (DFn, DFd)", "P value", "Summary"],
                        "rows": [["Treatment (columns)", f"F ({d1}, {d2:.4g}) = {f:.4g}", p_format(pval), p_summary(pval)]]}]
                      + _random_display(rb, "Subcolumns (within columns)")}
    if method != "none":
        fam = "all_pairs" if method == "tukey" else "vs_control" if method == "dunnett" else p["mc_family"]
        ctl = _control_index(ids, p) if fam == "vs_control" else None
        # Dunnett correlation from the number of values per group (Prism), not the number of subjects
        comps = _mc_from_contrasts(fit, xrows, _pairs(k, fam, ctl), ids, method, level, ctl, k, np.array([nvals[i] for i in ids], float))
        res["comparisons"], res["mc_method"], res["diff_label"] = comps, method, "A - B"
        res["display"].append(_mc_table(f"Multiple comparisons ({method})", comps, titles, "Mean diff."))
    return res


# --------------------------------------------------------------------------- correlation

SPEARMAN_EXACT_MAX = 17      # exact P up to this n (Prism); t approximation above
_PERM_CACHE = {}


def _perm_cost_dist(cost, bound):
    """Number of permutations s of 0..n-1 for every total sum_i cost[i, s(i)] (integer costs, |partial sums| <=
    bound; result index = total + bound). Dynamic programming over the subsets of used ranks, one popcount layer at
    a time (2^n masks x (2 bound + 1) sums); float64 counts are exact below 2^53 > 17!."""
    n = len(cost)
    width = 2 * bound + 1
    size = 1 << n
    allm = np.arange(size)
    pop = sum((allm >> b) & 1 for b in range(n))
    idx = np.empty(size, np.int32)
    masks, cur = np.array([0]), np.zeros((1, width))
    cur[0, bound] = 1.0
    for i in range(n):
        nxt_masks = np.flatnonzero(pop == i + 1)
        idx[nxt_masks] = np.arange(len(nxt_masks))
        nxt = np.zeros((len(nxt_masks), width))
        for r in range(n):
            src = np.flatnonzero(((masks >> r) & 1) == 0)
            dst = idx[masks[src] | (1 << r)]
            c = int(cost[i, r])
            if c >= 0:
                nxt[dst, c:] += cur[src, :width - c]
            else:
                nxt[dst, :width + c] += cur[src, -c:]
        masks, cur = nxt_masks, nxt
    return cur[0]


SPEARMAN_NULL_FILE = Path(__file__).with_name("spearman_null.npz")  # built by validation/make_spearman_null.py


def _stored_null(n, bound):
    """Precomputed untied null distribution for n (index = T + bound), or None if absent/mismatched."""
    if not SPEARMAN_NULL_FILE.exists():
        return None
    with np.load(SPEARMAN_NULL_FILE, allow_pickle=False) as z:
        d = z[f"n{n}"] if f"n{n}" in z.files else None
    return d if d is not None and len(d) == 2 * bound + 1 else None


def untied_spearman_null(n):
    """(distribution, bound) for untied ranks of size n — used to build SPEARMAN_NULL_FILE."""
    u = np.arange(1, n + 1) - (n + 1) / 2
    c = np.rint(4 * np.outer(u, u)).astype(np.int64)
    g = int(np.gcd.reduce(np.abs(c).ravel()))
    c //= g
    bound = int(math.ceil(4 * (u * u).sum() / g)) + 1
    return _perm_cost_dist(c, bound), bound


def _spearman_exact_p(a, b):
    """Two-tailed exact P: share of the n! pairings (midranks, ties allowed) with |rs| >= observed. The null
    distribution of T = sum u_i v_s(i) (centred ranks) comes from _perm_cost_dist, cached per cost matrix (for
    data without ties that is once per n). None when the DP would need more than about 1 GB (large n with ties)."""
    n = len(a)
    u, v = stats.rankdata(a) - (n + 1) / 2, stats.rankdata(b) - (n + 1) / 2
    c = np.rint(4 * np.outer(u, v)).astype(np.int64)         # 4 u v is an integer (ranks are multiples of 1/2)
    g = int(np.gcd.reduce(np.abs(c).ravel()))
    c //= g
    bound = int(math.ceil(4 * math.sqrt((u * u).sum() * (v * v).sum()) / g)) + 1    # Cauchy-Schwarz bound of any partial sum
    if 8 * (2 * bound + 1) * (math.comb(n, n // 2) + math.comb(n, n // 2 + 1)) > 1e9:
        return None
    tied = len(np.unique(a)) < n or len(np.unique(b)) < n
    # untied: the null distribution depends only on n (row/column order does not change it) -> canonical sorted ranks
    canon = c if tied else np.rint(4 * np.outer(np.sort(u), np.sort(v))).astype(np.int64) // g
    key = (n, canon.tobytes())
    if key not in _PERM_CACHE:
        stored = None if tied else _stored_null(n, bound)
        _PERM_CACHE[key] = stored if stored is not None else _perm_cost_dist(canon, bound)
    dist = _PERM_CACHE[key]
    obs = abs(int(np.trace(c)))
    hi = dist[obs + bound:].sum() if obs + bound < len(dist) else 0.0
    lo = dist[:bound - obs + 1].sum() if bound - obs >= 0 else 0.0
    return min(1.0, float((hi + lo) / math.factorial(n)))


def _correlate(x, y, method, level):
    n = len(x)
    if n < 3:
        raise _Bad(f"{n} value pair(s); at least 3 needed")
    if np.ptp(x) == 0:
        raise _Bad("all X values are identical (vertical line); no coefficient")
    if np.ptp(y) == 0:
        raise _Bad("all Y values are identical (horizontal line); no coefficient")
    out = {"n": n, "ci_low": None, "ci_high": None, "r2": None}
    if method == "pearson":
        r = float(np.corrcoef(x, y)[0, 1])
        out.update(r=r, r2=r * r, p=0.0 if abs(r) >= 1 else float(2 * stats.t.sf(abs(r) * math.sqrt((n - 2) / (1 - r * r)), n - 2)),
                   p_method="t")
        zse = 1 / math.sqrt(n - 3) if n > 3 else None
    else:
        r = float(stats.spearmanr(x, y).statistic)
        pe = _spearman_exact_p(x, y) if n <= SPEARMAN_EXACT_MAX else None
        if pe is not None:
            out.update(p=pe, p_method="exact")
        else:
            out.update(p=0.0 if abs(r) >= 1 else float(2 * stats.t.sf(abs(r) * math.sqrt((n - 2) / (1 - r * r)), n - 2)),
                       p_method="approximate")
        out["r"] = r
        zse = math.sqrt(1.06 / (n - 3)) if n > 10 else None
    if zse is not None and abs(out["r"]) < 1:
        h = stats.norm.ppf((1 + level) / 2) * zse
        z = math.atanh(out["r"])
        out.update(ci_low=math.tanh(z - h), ci_high=math.tanh(z + h))
    return out


def _pair_xy(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return a[ok], b[ok]


@_kind("correlation")
def correlation(table, ids, p):
    _need_raw(table, "correlation")
    method, level = p["method"], p["ci_level"]
    rs = "r" if method == "pearson" else "rs"
    msgs, pairs = [], []
    if table.kind == "XY":
        for i in ids:
            xv, yv, _ = table.points(i)
            pairs.append((table.x_title, table.dataset(i).title, xv, yv, i))
        labels = None
    else:
        if len(ids) < 2:
            raise _Bad("select at least 2 data sets")
        vecs = {i: _col_vec(table, i) for i in ids}
        for a, b in combinations(ids, 2):
            xv, yv = _pair_xy(vecs[a], vecs[b])
            pairs.append((table.dataset(a).title, table.dataset(b).title, xv, yv, (a, b)))
        labels = [table.dataset(i).title for i in ids]
    results = []
    for xl, yl, xv, yv, key in pairs:
        try:
            r = _correlate(np.asarray(xv, float), np.asarray(yv, float), method, level)
        except _Bad as e:
            msgs.append(f"{xl} vs. {yl}: {e}")
            r = {"n": len(xv), "r": None, "r2": None, "p": None, "ci_low": None, "ci_high": None, "p_method": None}
        results.append({"x": xl, "y": yl, "key": key, **r})
    if all(r["r"] is None for r in results):
        raise _Bad("; ".join(msgs))
    ci_lab = f"{level:.0%} CI"
    ptab = {"title": "Correlation", "columns": ["Pair", "n", rs, f"{ci_lab} low", f"{ci_lab} high", "R squared", "P value (two-tailed)", "Summary", "P value type"],
            "rows": [[f"{r['x']} vs. {r['y']}", r["n"], r["r"], r["ci_low"], r["ci_high"], r["r2"], p_format(r["p"]), p_summary(r["p"]),
                      r["p_method"]] for r in results]}
    res = {"messages": msgs, "method": method, "pairs": [{k: v for k, v in r.items() if k != "key"} for r in results],
           "n": {i: int(np.sum(~np.isnan(_col_vec(table, i)))) if table.kind == "Column" else len(table.points(i)[1]) for i in ids},
           "display": [ptab]}
    if labels is not None:
        m = len(ids)
        mats = {k: np.full((m, m), np.nan) for k in ("r", "p", "n", "ci_low", "ci_high")}
        pos = {i: q for q, i in enumerate(ids)}
        for r in results:
            a, b = r["key"]
            for k in mats:
                v = r[k]
                mats[k][pos[a], pos[b]] = mats[k][pos[b], pos[a]] = np.nan if v is None else v
        for q, i in enumerate(ids):
            v = _col_vec(table, i)
            nv = int(np.sum(~np.isnan(v)))
            mats["n"][q, q] = nv
            mats["r"][q, q] = 1.0
            mats["p"][q, q] = np.nan
        res["matrix"] = {"labels": labels, "r": mats["r"], "p": mats["p"], "n": mats["n"],
                         "ci_low": mats["ci_low"], "ci_high": mats["ci_high"], "method": method}
        res["display"] = [{"title": f"Correlation matrix ({rs})", "columns": [""] + labels, "rows": [[labels[q]] + list(mats["r"][q]) for q in range(m)]},
                          {"title": "P values (two-tailed)", "columns": [""] + labels,
                           "rows": [[labels[q]] + [p_format(v) for v in mats["p"][q]] for q in range(m)]},
                          {"title": "Number of XY pairs", "columns": [""] + labels, "rows": [[labels[q]] + list(mats["n"][q]) for q in range(m)]}] + res["display"]
    return res


# --------------------------------------------------------------------------- registry

_MC_CHOICES = (("none", "None"), ("lsd", "No correction (Fisher's LSD)"), ("tukey", "Tukey"), ("dunnett", "Dunnett"),
               ("sidak", "Sidak"), ("bonferroni", "Bonferroni"), ("holm_sidak", "Holm-Sidak"))
_MODEL = Param("model", "Model", "choice", "auto", (("auto", "ANOVA; mixed-effects model when values are blank"),
                                                    ("anova", "Repeated-measures ANOVA"), ("mixed", "Mixed-effects model")))
_SPH = Param("sphericity", "Sphericity", "choice", "gg", (("gg", "Not assumed (Geisser-Greenhouse correction)"), ("assumed", "Assumed")))

KINDS = {
    "normality": AnalysisKind("normality", "Normality and lognormality tests", ("Column", "XY", "Grouped"), normality, (
        Param("dagostino", "D'Agostino-Pearson omnibus", "bool", True), Param("anderson", "Anderson-Darling", "bool", True),
        Param("shapiro", "Shapiro-Wilk", "bool", True), Param("ks", "Kolmogorov-Smirnov", "bool", True),
        Param("lognormal", "Test lognormality (tests on the logs)", "bool", False), Param("alpha", "Significance level", "float", 0.05))),
    "outliers": AnalysisKind("outliers", "Identify outliers", ("Column",), outliers, (
        Param("method", "Method", "choice", "rout", (("rout", "ROUT"), ("grubbs", "Grubbs (one outlier)"), ("grubbs_iterative", "Iterative Grubbs"))),
        Param("q", "ROUT Q (max. false discovery rate)", "float", 0.01), Param("alpha", "Grubbs alpha", "float", 0.05),
        Param("mark", "Output", "choice", "remove", (("remove", "Cleaned data (outliers blank)"), ("exclude", "Outliers flagged as excluded")))),
        output="table"),
    "rm_one_way_anova": AnalysisKind("rm_one_way_anova", "Repeated-measures one-way ANOVA / mixed-effects model", ("Column",), rm_one_way_anova, (
        _MODEL, _SPH, Param("multiple_comparisons", "Multiple comparisons", "choice", "none", _MC_CHOICES), _MCFAM, _CONTROL, _CI)),
    "rm_two_way_anova": AnalysisKind("rm_two_way_anova", "Repeated-measures two-way ANOVA / mixed-effects model", ("Grouped",), rm_two_way_anova, (
        Param("design", "Repeated measures", "choice", "rows", (("rows", "Each row is a time point (subjects in subcolumns)"),
                                                                ("columns", "Each column is a time point (subjects across a row)"))),
        _MODEL, _SPH,
        Param("multiple_comparisons", "Multiple comparisons", "choice", "none",
              (("none", "None"), ("within_rows", "Data sets within each row"), ("within_columns", "Rows within each data set"),
               ("main_column", "Column means (main effect)"))),
        Param("correction", "Correction", "choice", "auto",
              (("auto", "Prism default"), ("sidak", "Sidak"), ("tukey", "Tukey"), ("bonferroni", "Bonferroni"))),
        Param("family", "Family (within rows / columns)", "choice", "per_row",
              (("per_row", "Separate families (one per row, or per data set)"), ("all", "One family for all comparisons"))), _CI)),
    "nested_t_test": AnalysisKind("nested_t_test", "Nested t test", ("Nested",), nested_t_test, (_CI,)),
    "nested_one_way_anova": AnalysisKind("nested_one_way_anova", "Nested one-way ANOVA", ("Nested",), nested_one_way_anova, (
        Param("multiple_comparisons", "Multiple comparisons", "choice", "none", _MC_CHOICES), _MCFAM, _CONTROL, _CI)),
    "correlation": AnalysisKind("correlation", "Correlation", ("XY", "Column"), correlation, (
        Param("method", "Coefficient", "choice", "pearson", (("pearson", "Pearson r"), ("spearman", "Spearman rs"))), _CI)),
}
