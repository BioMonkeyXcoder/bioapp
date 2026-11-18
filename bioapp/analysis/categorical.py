"""Categorical, survival and curve analyses following GraphPad Prism's documented behaviour.

Prism conventions used here (Prism statistics guide):
  * 2x2 tables: Fisher's exact test (two-sided, sum of the probabilities of all tables no more likely than
    the observed one), chi-square with or without Yates, relative risk = (row 1 left-column fraction) /
    (row 2 left-column fraction) with the Koopman asymptotic score CI (Katz optional; Katz puts 0.5 in a zero
    left-column cell only, Woolf adds 0.5 to all cells), odds ratio with the Baptista-Pike exact CI (mid-P
    minimum-likelihood limits; Woolf optional), difference between proportions reported positive (larger proportion
    first) with Newcombe/Wilson hybrid CI (continuity corrected by default),
    sensitivity / specificity / predictive values with Wilson/Brown intervals;
  * larger tables: chi-square, Fisher-Freeman-Halton exact test (small tables), chi-square for trend (2 columns,
    >= 3 rows, scores 1..r), phi and Cramer's V;
  * survival: Kaplan-Meier, Greenwood SE, asymmetric (log-log) CI by default, an event precedes a censoring at
    the same time, logrank (Mantel-Cox) and Gehan-Breslow-Wilcoxon tests, logrank test for trend (scores 1..k),
    hazard ratio by the Mantel-Haenszel and logrank methods with reciprocals, ratio of medians (exponential CI),
    ratio CIs with z = 1.96;
  * area under the curve: trapezoid rule on [Y - baseline], total area = sum of |area| of all peaks above and below
    the baseline, peaks smaller than a percentage of max-min ignored from the peak list, SE of the area from the
    replicate variances of the points (fitted to Prism), CI = AUC +- z SE;
  * frequency distribution: bins [lo, hi), labelled by their centres (centred on multiples of the width), automatic
    width as in Prism (mean range / (1 + log2 n), rounded down to 1, 2 or 5 x 10^k), values outside the first / last
    bin are omitted;
  * three-way ANOVA: four Grouped data sets = factor A (data sets 1-2 vs 3-4) x factor B (odd vs even data
    set), rows = factor C, subcolumns = replicates; type III sums of squares.

Every function returns the result contract of DESIGN.md; invalid input gives status "failed" with a reason.
"""
from __future__ import annotations

import math
import warnings
from itertools import combinations

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln, logsumexp

from . import AnalysisKind, Param
from .stats import _Bad, _cells, _comp, _mc_adjust, p_format, p_summary
from .stats import _kind as _stats_kind


def _lists(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _lists(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_lists(v) for v in o]
    return o


def _kind(name):
    """stats._kind (failure handling, status, NaN/inf -> None) plus ndarray -> list."""
    def deco(f):
        def g(table, ids, p):
            return _lists(f(table, ids, p))
        g.__name__ = f.__name__
        return _stats_kind(name)(g)
    return deco


def _z(level):
    return float(stats.norm.ppf((1 + level) / 2))


def _check_level(level):
    if not 0 < level < 1:
        raise _Bad("confidence level must be between 0 and 1")


def _nice(w):
    """Nearest round number (1, 2, 2.5, 5 x 10^k) to w."""
    e = 10.0 ** math.floor(math.log10(w))
    return e * min((1, 2, 2.5, 5, 10), key=lambda c: abs(math.log(c / (w / e))))


def _floor_nice(w):
    """Largest 1, 2 or 5 x 10^k not above w."""
    e = 10.0 ** math.floor(math.log10(w))
    return e * max(c for c in (1, 2, 5) if c <= w / e * (1 + 1e-12))


def _title(table, i):
    return table.dataset(i).title


def _row_titles(table, rows):
    return [table.row_titles[r] if r < len(table.row_titles) and table.row_titles[r] else f"Row {r + 1}" for r in rows]


# =========================================================================== contingency

def _count_matrix(table, ids):
    if len(ids) < 2:
        raise _Bad("select at least 2 outcome columns (data sets)")
    m = np.column_stack([np.where(table.dataset(i).excluded[:, 0], np.nan, table.dataset(i).values[:, 0]) for i in ids])
    blank = np.isnan(m)
    keep = ~blank.all(axis=1)
    if blank[keep].any():
        raise _Bad("blank or excluded count in a used row; every row x column cell needs a count")
    rows = np.flatnonzero(keep)
    if len(rows) < 2:
        raise _Bad("a contingency table needs at least 2 non-empty rows")
    m = m[keep].astype(int)
    if (m.sum(axis=1) == 0).any() or (m.sum(axis=0) == 0).any():
        raise _Bad("a row or column has a zero total; the table cannot be analysed")
    return m, rows


def _wilson(x, n, z, cc=False):
    p, q, z2 = x / n, 1 - x / n, z * z
    if not cc:
        c, h = (p + z2 / (2 * n)) / (1 + z2 / n), z * math.sqrt(p * q / n + z2 / (4 * n * n)) / (1 + z2 / n)
        return max(0.0, c - h), min(1.0, c + h)
    # no special case at x = 0 or x = n: Prism's Newcombe interval uses the formula value there (0.0092 for 0 of 10)
    lo = (2 * n * p + z2 - 1 - z * math.sqrt(max(z2 - 2 - 1 / n + 4 * p * (n * q + 1), 0))) / (2 * (n + z2))
    hi = (2 * n * p + z2 + 1 + z * math.sqrt(max(z2 + 2 - 1 / n + 4 * p * (n * q - 1), 0))) / (2 * (n + z2))
    return max(0.0, lo), min(1.0, hi)


def _wilson_brown(x, n, level):
    """Prism's Wilson/Brown interval: the plain Wilson score interval, also at x = 0 or x = n (upper limit z^2 / (n + z^2)
    for x = 0, e.g. 0.390 for 0 of 6). The Brown-Cai-DasGupta edge modifications at x = 1 and x = n - 1 are not confirmed
    in Prism and not applied."""
    return _wilson(x, n, _z(level))


def _newcombe(x1, n1, x2, n2, z, cc):
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = _wilson(x1, n1, z, cc)
    l2, u2 = _wilson(x2, n2, z, cc)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def _rr_score(phi, x1, n1, x2, n2):
    """Koopman score statistic for H0: p1 / p2 = phi (vectorised over phi)."""
    phi = np.asarray(phi, float)
    N = n1 + n2
    A, B, C = N * phi, -(phi * (n1 + x2) + n2 + x1), x1 + x2
    p2 = (-B - np.sqrt(np.maximum(B * B - 4 * A * C, 0))) / (2 * A)
    p1 = phi * p2
    q1, q2 = 1 - p1, 1 - p2
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (x1 - n1 * p1) / (phi * q1)
        i11, i12 = n1 * p2 ** 2 / (p1 * q1), n1 * phi * p2 / (p1 * q1)
        i22 = n1 * phi ** 2 / (p1 * q1) + n2 / (p2 * q2)
        return u * u / (i11 - i12 ** 2 / i22)


_GRID = np.exp(np.linspace(np.log(1e-9), np.log(1e9), 4000))


def _invert(stat, level_stat, grid=_GRID):
    """Limits of {phi: stat(phi) <= level_stat} on a log grid, refined by root finding.
    Assumes a single interval; 0 / inf when the interval reaches the grid edge."""
    g = np.asarray(stat(grid)) - level_stat
    inside = np.flatnonzero(g <= 0)
    if len(inside) == 0:
        return None, None
    i0, i1 = inside[0], inside[-1]
    f = lambda v: float(stat(np.array([v]))[0]) - level_stat  # noqa: E731
    lo = 0.0 if i0 == 0 else optimize.brentq(lambda lv: f(math.exp(lv)), math.log(grid[i0 - 1]), math.log(grid[i0]), xtol=1e-12)
    hi = math.inf if i1 == len(grid) - 1 else optimize.brentq(lambda lv: f(math.exp(lv)), math.log(grid[i1]), math.log(grid[i1 + 1]), xtol=1e-12)
    return (lo if lo == 0.0 else math.exp(lo)), (hi if hi == math.inf else math.exp(hi))


def _rr_ci(a, b, c, d, level, method):
    n1, n2 = a + b, c + d
    z = _z(level)
    if a == 0 and c == 0:
        return None, None, None, "relative risk is undefined when both left-column counts are zero"
    if method == "katz":
        # Prism: 0.5 replaces a zero left-column cell only (the row totals stay); a zero numerator gives a lower limit of 0
        note, zero_a = None, a == 0
        if a == 0 or c == 0:
            a, c = a or .5, c or .5
            note = "a left-column count is zero: 0.5 was used for that cell only (Katz method)"
        rr = (a / n1) / (c / n2)
        se = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n2)
        return rr, 0.0 if zero_a else rr * math.exp(-z * se), rr * math.exp(z * se), note
    rr = (a / n1) / (c / n2) if c else math.inf
    lo, hi = _invert(lambda ph: _rr_score(ph, a, n1, c, n2), z * z)
    return rr, lo, hi, None


def _log_nchg(x, n1, n2, m1, psi):
    """log pmf of the conditional distribution of cell a (noncentral hypergeometric); psi is an array (g,)."""
    lo, hi = max(0, m1 - n2), min(n1, m1)
    xs = np.arange(lo, hi + 1)
    base = gammaln(n1 + 1) - gammaln(xs + 1) - gammaln(n1 - xs + 1) + gammaln(n2 + 1) - gammaln(m1 - xs + 1) - gammaln(n2 - m1 + xs + 1)
    lp = base[None, :] + xs[None, :] * np.log(np.atleast_1d(psi))[:, None]
    lp -= logsumexp(lp, axis=1, keepdims=True)
    return xs, lp


def _or_midp(a, n1, n2, m1, psi):
    """Mid-P version of the minimum-likelihood P: tables less likely than the observed one count fully, tables as
    likely as the observed one (the observed table itself) count half. Vectorised over psi."""
    xs, lp = _log_nchg(a, n1, n2, m1, psi)
    obs = lp[:, a - xs[0]][:, None]
    p = np.exp(lp)
    return (p * (lp < obs - 1e-7)).sum(axis=1) + 0.5 * (p * (np.abs(lp - obs) <= 1e-7)).sum(axis=1)


def _or_ci(a, b, c, d, level, method):
    z = _z(level)
    if method == "woolf":
        note = None
        if min(a, b, c, d) == 0:
            a, b, c, d = a + .5, b + .5, c + .5, d + .5
            note = "a cell is zero: 0.5 was added to all cells (Woolf method)"
        o = a * d / (b * c)
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        return o, o * math.exp(-z * se), o * math.exp(z * se), note
    o = a * d / (b * c) if b * c else math.inf
    n1, n2, m1 = a + b, c + d, a + c
    # Prism's Baptista-Pike limits (matches all 11 limits of 6 tables): the smallest and the largest psi whose
    # mid-P minimum-likelihood P value is >= alpha. The mid-P is a sawtooth in psi, so the limits sit at its jumps.
    lo, hi = _invert(lambda ps: -_or_midp(a, n1, n2, m1, ps), -(1 - level))
    return o, lo, hi, None


def _ffh_p(m, max_tables=3_000_000):
    """Fisher-Freeman-Halton exact P: sum of the probabilities of all tables with the observed margins that
    are no more likely than the observed one."""
    rows, cols = m.sum(axis=1), m.sum(axis=0)
    lf = gammaln(np.arange(0, int(m.sum()) + 2) + 1)
    const = lf[rows].sum() + lf[cols].sum() - lf[int(m.sum())]
    obs = const - lf[m].sum()
    count = [0]
    total = [0.0]

    def comps(tot, caps):
        if len(caps) == 1:
            if tot <= caps[0]:
                yield (tot,)
            return
        rest = sum(caps[1:])
        for x in range(max(0, tot - rest), min(tot, caps[0]) + 1):
            for t in comps(tot - x, caps[1:]):
                yield (x,) + t

    def rec(j, rem, acc):
        if j == len(cols) - 1:
            lp = const - acc - lf[list(rem)].sum()
            count[0] += 1
            if count[0] > max_tables:
                raise _Bad("table too large for the exact test; use the chi-square test")
            if lp <= obs + 1e-7:
                total[0] += math.exp(lp)
            return
        for v in comps(int(cols[j]), list(rem)):
            rec(j + 1, tuple(r - x for r, x in zip(rem, v)), acc + lf[list(v)].sum())

    rec(0, tuple(int(r) for r in rows), 0.0)
    return min(1.0, total[0])


def _trend(m):
    """Chi-square for trend (Cochran-Armitage) for a table with 2 columns, rows scored 1..r; returns (chi2, p)."""
    n = m.sum(axis=1).astype(float)
    a = m[:, 0].astype(float)
    s = np.arange(1, len(n) + 1, dtype=float)
    N, A = n.sum(), a.sum()
    pbar = A / N
    num = (a @ s - A * (n @ s) / N) ** 2
    den = pbar * (1 - pbar) * (n @ s ** 2 - (n @ s) ** 2 / N)
    chi2 = num / den
    return float(chi2), float(stats.chi2.sf(chi2, 1))


@_kind("contingency")
def contingency(table, ids, p):
    m, used = _count_matrix(table, ids)
    level = p["ci_level"]
    _check_level(level)
    R, C = m.shape
    N = int(m.sum())
    rt = _row_titles(table, used)
    ct = [_title(table, i) for i in ids]
    msgs = []
    res = {"n": N, "counts": m.tolist(), "row_titles": rt, "col_titles": ct}
    exp = np.outer(m.sum(axis=1), m.sum(axis=0)) / N
    if (exp < 5).any():
        msgs.append(f"{int((exp < 5).sum())} cell(s) have an expected count below 5; chi-square is approximate (Fisher's exact test is preferred)")
    chi2, pchi, dfc, _ = stats.chi2_contingency(m, correction=False)
    yates = bool(p["yates"]) and R == 2 and C == 2
    if yates:
        chi2, pchi, dfc, _ = stats.chi2_contingency(m, correction=True)
    res["chi_square"] = {"chi2": float(chi2), "df": int(dfc), "p": float(pchi), "yates": yates}
    chi2_u = float(stats.chi2_contingency(m, correction=False)[0])
    v = math.sqrt(chi2_u / (N * (min(R, C) - 1)))
    res["cramers_v"] = v
    disp = []
    pc = p["percent_of"]
    if pc != "none":
        den = {"row": m.sum(axis=1, keepdims=True), "column": m.sum(axis=0, keepdims=True), "total": N}[pc]
        pct = 100 * m / den
        res["percentages"] = {"of": pc, "values": pct.tolist()}
        disp.append({"title": f"Counts and {pc} percentages", "columns": [""] + [f"{c} (n)" for c in ct] + [f"{c} (%)" for c in ct],
                     "rows": [[rt[r]] + m[r].tolist() + pct[r].tolist() for r in range(R)]})
    else:
        disp.append({"title": "Counts", "columns": [""] + ct, "rows": [[rt[r]] + m[r].tolist() for r in range(R)]})
    test_rows = [["Chi-square" + (" (Yates' correction)" if yates else ""), float(chi2)], ["DF", int(dfc)], ["P value (two-sided)", p_format(pchi)],
                 ["P value summary", p_summary(pchi)], ["Cramer's V", v]]
    if R == 2 and C == 2:
        (a, b), (c, d) = m.tolist()
        fo, pf = stats.fisher_exact(m)
        res["fisher"] = {"p": float(pf), "odds_ratio_sample": float(fo)}
        res["chi_square_yates"] = {"chi2": float(stats.chi2_contingency(m, correction=True)[0]), "p": float(stats.chi2_contingency(m, correction=True)[1])}
        res["phi"] = v if a * d >= b * c else -v
        test_rows = [["Fisher's exact test P value (two-sided)", p_format(pf)], ["P value summary (Fisher)", p_summary(pf)]] + test_rows
        n1, n2, z = a + b, c + d, _z(level)
        rr, rlo, rhi, note = _rr_ci(a, b, c, d, level, p["rr_method"])
        orr, olo, ohi, onote = _or_ci(a, b, c, d, level, p["or_method"])
        msgs += [n_ for n_ in (note, onote) if n_]
        inv = lambda x: None if x is None else (math.inf if x == 0 else 1 / x)  # noqa: E731
        res["relative_risk"] = {"value": rr, "ci_low": rlo, "ci_high": rhi, "method": p["rr_method"],
                                "reciprocal": inv(rr), "reciprocal_ci_low": inv(rhi), "reciprocal_ci_high": inv(rlo)}
        res["odds_ratio"] = {"value": orr, "ci_low": olo, "ci_high": ohi, "method": p["or_method"],
                             "reciprocal": inv(orr), "reciprocal_ci_low": inv(ohi), "reciprocal_ci_high": inv(olo)}
        p1, p2 = a / n1, c / n2
        dm = p["diff_method"]
        if dm == "wald":
            h = z * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            dlo, dhi = p1 - p2 - h, p1 - p2 + h
        else:
            dlo, dhi = _newcombe(a, n1, c, n2, z, dm == "newcombe_cc")
        dv = p1 - p2
        if dv < 0:  # Prism reports the difference as positive (larger proportion first) and keeps the interval arms: |d| - arm_low, |d| + arm_high
            dlo, dhi, dv = dlo - 2 * dv, dhi - 2 * dv, -dv
        res["difference"] = {"value": dv, "ci_low": dlo, "ci_high": dhi, "method": dm}
        sens, spec, ppv, npv = (a, a + c), (d, b + d), (a, a + b), (d, c + d)
        diag = {}
        for name, (x, n) in (("sensitivity", sens), ("specificity", spec), ("ppv", ppv), ("npv", npv)):
            lo_, hi_ = (_wilson_brown if p["prop_ci"] == "wilson_brown" else _clopper)(x, n, level)
            diag[name] = {"value": x / n, "ci_low": lo_, "ci_high": hi_}
        se_, sp_ = sens[0] / sens[1], spec[0] / spec[1]
        diag["lr_positive"] = se_ / (1 - sp_) if sp_ < 1 else None
        diag["lr_negative"] = (1 - se_) / sp_ if sp_ > 0 else None
        res["diagnostic"] = diag
        disp.append({"title": "Test", "columns": ["Parameter", "Value"], "rows": test_rows})
        disp.append({"title": "Effect sizes (rows: first / second; first column = event)", "columns": ["Measure", "Value", f"{level:.0%} CI low", f"{level:.0%} CI high"],
                     "rows": [["Relative risk", rr, rlo, rhi], ["Reciprocal of relative risk", res["relative_risk"]["reciprocal"], res["relative_risk"]["reciprocal_ci_low"], res["relative_risk"]["reciprocal_ci_high"]],
                              ["Odds ratio", orr, olo, ohi], ["Reciprocal of odds ratio", res["odds_ratio"]["reciprocal"], res["odds_ratio"]["reciprocal_ci_low"], res["odds_ratio"]["reciprocal_ci_high"]],
                              ["Difference between proportions (larger first)", dv, dlo, dhi], ["Phi coefficient", res["phi"], None, None]]})
        disp.append({"title": "Diagnostic measures (rows: test positive / negative; columns: condition present / absent)",
                     "columns": ["Measure", "Value", f"{level:.0%} CI low", f"{level:.0%} CI high"],
                     "rows": [[k.upper() if k in ("ppv", "npv") else k.capitalize(), diag[k]["value"], diag[k]["ci_low"], diag[k]["ci_high"]] for k in ("sensitivity", "specificity", "ppv", "npv")]
                     + [["Likelihood ratio (+)", diag["lr_positive"], None, None], ["Likelihood ratio (-)", diag["lr_negative"], None, None]]})
    else:
        if p["fisher_exact"]:
            pf = _ffh_p(m)
            res["fisher_freeman_halton"] = {"p": pf}
            test_rows = [["Fisher-Freeman-Halton exact P value", p_format(pf)]] + test_rows
        if C == 2 and R >= 3:
            tr, ptr = _trend(m)
            res["trend"] = {"chi2": tr, "df": 1, "p": ptr, "chi2_departure": float(chi2_u - tr), "df_departure": R - 2,
                            "p_departure": float(stats.chi2.sf(max(chi2_u - tr, 0.0), R - 2))}
            test_rows += [["Chi-square for trend", tr], ["Trend DF", 1], ["Trend P value", p_format(ptr)],
                          ["Departure from linear trend (chi-square)", chi2_u - tr], ["Departure P value", p_format(res["trend"]["p_departure"])]]
        disp.append({"title": "Test", "columns": ["Parameter", "Value"], "rows": test_rows})
    res["messages"] = msgs
    res["display"] = disp
    return res


def _clopper(x, n, level):
    a = 1 - level
    return (0.0 if x == 0 else float(stats.beta.ppf(a / 2, x, n - x + 1)),
            1.0 if x == n else float(stats.beta.ppf(1 - a / 2, x + 1, n - x)))


# =========================================================================== survival

def _subjects(table, ids):
    """-> {id: (time, event)}; blanks / excluded cells are not subjects."""
    if len(ids) < 1:
        raise _Bad("select at least one data set (group)")
    out, seen = {}, np.zeros(table.n_rows, int)
    for i in ids:
        ds = table.dataset(i)
        v = np.where(ds.excluded[:, 0], np.nan, ds.values[:, 0])
        m = ~np.isnan(v)
        t = table.x[m]
        if np.isnan(t).any():
            raise _Bad(f"{ds.title}: a subject has a code but no time")
        if (t < 0).any():
            raise _Bad(f"{ds.title}: negative time")
        if m.sum() == 0:
            raise _Bad(f"{ds.title}: no subjects")
        seen += m
        out[i] = (t.astype(float), v[m].astype(int))
    if (seen > 1).any():
        raise _Bad("a row holds a subject in more than one group; each subject belongs to exactly one group")
    return out


def _km(t, e, z, method):
    """Kaplan-Meier with Greenwood SE; vertices start at time 0."""
    ts = np.sort(t)
    te = np.sort(t[e == 1])
    u = np.unique(te)
    n = len(t) - np.searchsorted(ts, u, "left")
    d = np.searchsorted(te, u, "right") - np.searchsorted(te, u, "left")
    surv = np.cumprod(1 - d / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.cumsum(d / (n * (n - d)))
        se = surv * np.sqrt(g)
        if method == "greenwood":
            lo, hi = np.clip(surv - z * se, 0, 1), np.clip(surv + z * se, 0, 1)
        else:
            c = np.sqrt(g) / np.abs(np.log(surv))
            lo, hi = surv ** np.exp(z * c), surv ** np.exp(-z * c)
    dead = surv <= 0
    se[dead] = lo[dead] = hi[dead] = np.nan
    return {"u": u, "n": n, "d": d, "surv": surv, "se": se, "lo": lo, "hi": hi}


def _median(time, surv, lo, hi):
    """Prism: first time survival <= 50 % (midpoint of the interval when exactly 50 %); CI from the CI curve."""
    def first(s, thr=0.5):
        k = np.flatnonzero(s <= thr + 1e-12)
        return None if len(k) == 0 else int(k[0])
    k = first(surv)
    if k is None:
        med = None
    elif abs(surv[k] - 0.5) < 1e-12 and k + 1 < len(time):
        med = (time[k] + time[k + 1]) / 2
    else:
        med = time[k]
    kl, ku = first(np.nan_to_num(lo, nan=2.0)), first(np.nan_to_num(hi, nan=2.0))  # the lower curve crosses 50 % first
    return med, (None if kl is None else float(time[kl])), (None if ku is None else float(time[ku]))


def _lr_stats(data, weight):
    """Observed, expected events (k,) and covariance (k, k) for the weighted logrank family."""
    allt = np.concatenate([t for t, _ in data])
    alle = np.concatenate([e for _, e in data])
    u = np.unique(allt[alle == 1])
    k = len(data)
    n, d = np.zeros((k, len(u))), np.zeros((k, len(u)))
    for j, (t, e) in enumerate(data):
        te = np.sort(t[e == 1])
        n[j] = len(t) - np.searchsorted(np.sort(t), u, "left")
        d[j] = np.searchsorted(te, u, "right") - np.searchsorted(te, u, "left")
    N, D = n.sum(0), d.sum(0)
    w = np.ones_like(N) if weight == "logrank" else N
    frac = n / N
    O, E = (w * d).sum(1), (w * D * frac).sum(1)
    c = np.where(N > 1, D * (N - D) / np.maximum(N - 1, 1), 0.0) * w ** 2
    V = np.diag((c * frac).sum(1)) - np.einsum("it,jt->ij", frac * c, frac)
    return O, E, V


def _chi2_k(O, E, V, method):
    """Chi-square for k groups; "conservative" is sum (O - E)^2 / E (Mantel-Cox only: Prism keeps the full
    variance for Gehan-Breslow-Wilcoxon)."""
    k = len(O)
    if method == "conservative":
        if (E <= 0).any():
            raise _Bad("a group has zero expected events")
        s = float((((O - E) ** 2) / E).sum())
    else:
        dv = (O - E)[:-1]
        try:
            s = float(dv @ np.linalg.solve(V[:-1, :-1], dv))
        except np.linalg.LinAlgError as e:
            raise _Bad("the variance matrix is singular (a group has no information)") from e
    return s, k - 1, float(stats.chi2.sf(s, k - 1))


def _at_risk_times(text, tmax):
    if text and text.strip():
        try:
            ts = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
        except ValueError as e:
            raise _Bad("number-at-risk times must be numbers separated by commas") from e
        if not ts:
            raise _Bad("number-at-risk times are empty")
        return sorted(ts)
    if tmax <= 0:
        return [0.0]
    step = _nice(tmax / 5)
    return [float(v) for v in np.arange(0, tmax + step * 1e-9, step)]


@_kind("survival")
def survival(table, ids, p):
    level = p["ci_level"]
    _check_level(level)
    z = _z(level)
    grp = _subjects(table, ids)
    ids = list(grp)
    titles = {i: _title(table, i) for i in ids}
    pct = p["units"] == "percent"
    sc = 100.0 if pct else 1.0
    msgs, curves, groups, disp = [], {}, {}, []
    tmax = max(float(t.max()) for t, _ in grp.values())
    art = _at_risk_times(p["at_risk_times"], tmax)
    at_risk = {"time": art}
    for i, (t, e) in grp.items():
        k = _km(t, e, z, p["ci_method"])
        time, surv, lo, hi = (np.concatenate([[0.0], k["u"]]), np.concatenate([[1.0], k["surv"]]),
                              np.concatenate([[1.0], k["lo"]]), np.concatenate([[1.0], k["hi"]]))
        if t.max() > time[-1]:  # the largest time is always shown when censored
            time, surv, lo, hi = (np.append(time, t.max()), np.append(surv, surv[-1]), np.append(lo, lo[-1]), np.append(hi, hi[-1]))
        ct = np.sort(t[e == 0])
        csurv = surv[np.searchsorted(k["u"], ct, "right")] if len(ct) else np.array([])
        curves[i] = {"time": time, "surv": surv, "ci_low": lo, "ci_high": hi, "censor_time": ct, "censor_surv": csurv}
        med, mlo, mhi = _median(time, surv, lo, hi)
        groups[i] = {"n": len(t), "events": int(e.sum()), "censored": int((e == 0).sum()), "median": med,
                     "median_ci_low": mlo, "median_ci_high": mhi}
        ts = np.sort(t)
        at_risk[i] = [int(len(t) - np.searchsorted(ts, a, "left")) for a in art]
        if e.sum() == 0:
            msgs.append(f"{titles[i]}: no events; the curve stays at 100 %")
        disp.append({"title": f"Survival proportions: {titles[i]}", "columns": ["Time", "Number at risk", "Events", "Survival (%)" if pct else "Survival", "SE", f"{level:.0%} CI low", f"{level:.0%} CI high"],
                     "rows": [[0.0, len(t), 0, sc, None, None, None]] + [[k["u"][q], int(k["n"][q]), int(k["d"][q]), sc * k["surv"][q], sc * k["se"][q], sc * k["lo"][q], sc * k["hi"][q]] for q in range(len(k["u"]))]})
    res = {"curves": curves, "at_risk": at_risk, "groups": groups, "ci_method": p["ci_method"], "units": p["units"],
           "n": {i: groups[i]["n"] for i in ids}}
    head = [{"title": "Survival summary", "columns": ["Group", "n", "Events", "Censored", "Median survival", f"Median {level:.0%} CI low", f"Median {level:.0%} CI high"],
             "rows": [[titles[i], g["n"], g["events"], g["censored"], g["median"] if g["median"] is not None else "undefined", g["median_ci_low"], g["median_ci_high"]] for i, g in groups.items()]}]
    k = len(ids)
    if k >= 2:
        data = [grp[i] for i in ids]
        if sum(int(e.sum()) for _, e in data) == 0:
            raise _Bad("no events in any group; the curves cannot be compared")
        cmp_, rows = {}, []
        for name, w in (("logrank", "logrank"), ("gbw", "gbw")):
            O, E, V = _lr_stats(data, w)
            lab = "Logrank (Mantel-Cox)" if w == "logrank" else "Gehan-Breslow-Wilcoxon"
            s, df, pv = _chi2_k(O, E, V, p["logrank_method"] if k >= 3 and w == "logrank" else "recommended")
            cmp_[name] = {"chi2": s, "df": df, "p": pv, "observed": O, "expected": E}
            rows += [[f"{lab} chi-square", s], [f"{lab} DF", df], [f"{lab} P value", p_format(pv)], [f"{lab} P summary", p_summary(pv)]]
        O, E, V = _lr_stats(data, "logrank")
        if k >= 3:
            c = np.arange(1, k + 1, dtype=float)
            # conservative: Poisson variance sum c^2 E - (sum c E)^2 / sum E instead of the full logrank variance
            var = float(c @ (c * E) - (c @ E) ** 2 / E.sum()) if p["logrank_method"] == "conservative" else float(c @ V @ c)
            if var <= 0:
                raise _Bad("variance of the trend statistic is zero")
            s = float(((c @ (O - E)) ** 2) / var)
            cmp_["trend"] = {"chi2": s, "df": 1, "p": float(stats.chi2.sf(s, 1))}
            rows += [["Logrank test for trend chi-square", s], ["Trend DF", 1], ["Trend P value", p_format(cmp_["trend"]["p"])], ["Trend P summary", p_summary(cmp_["trend"]["p"])]]
        if k == 2:
            v = float(V[0, 0])
            kk = (O[0] - E[0]) / v
            z = round(z, 2)  # Prism's ratio CIs use the rounded z (1.96 at 95 %)
            mh = {"hr": math.exp(kk), "ci_low": math.exp(kk - z / math.sqrt(v)), "ci_high": math.exp(kk + z / math.sqrt(v))}
            mh.update(recip=1 / mh["hr"], recip_ci_low=1 / mh["ci_high"], recip_ci_high=1 / mh["ci_low"])
            hr = {"mantel_haenszel": mh}
            hrows = [["Hazard ratio (Mantel-Haenszel), group 1 / group 2", mh["hr"], mh["ci_low"], mh["ci_high"]],
                     ["Reciprocal", mh["recip"], mh["recip_ci_low"], mh["recip_ci_high"]]]
            if min(O.min(), E.min()) > 0:
                lh = (O[0] / E[0]) / (O[1] / E[1])
                se = math.sqrt(1 / E[0] + 1 / E[1])
                lr = {"hr": lh, "ci_low": lh * math.exp(-z * se), "ci_high": lh * math.exp(z * se)}
                lr.update(recip=1 / lh, recip_ci_low=1 / lr["ci_high"], recip_ci_high=1 / lr["ci_low"])
                hr["logrank"] = lr
                hrows += [["Hazard ratio (logrank), group 1 / group 2", lr["hr"], lr["ci_low"], lr["ci_high"]], ["Reciprocal", lr["recip"], lr["recip_ci_low"], lr["recip_ci_high"]]]
            else:
                msgs.append("logrank-method hazard ratio needs events in both groups")
            cmp_["hazard_ratio"] = hr
            g1, g2 = groups[ids[0]], groups[ids[1]]
            if g1["median"] and g2["median"] and g1["events"] and g2["events"]:
                r = g1["median"] / g2["median"]
                se = math.sqrt(1 / g1["events"] + 1 / g2["events"])
                mr = {"ratio": r, "ci_low": r * math.exp(-z * se), "ci_high": r * math.exp(z * se)}
                mr.update(recip=1 / r, recip_ci_low=1 / mr["ci_high"], recip_ci_high=1 / mr["ci_low"])
                cmp_["median_ratio"] = mr
                hrows.append(["Ratio of median survivals (assumes exponential decay)", r, cmp_["median_ratio"]["ci_low"], cmp_["median_ratio"]["ci_high"]])
            head.append({"title": "Hazard ratio", "columns": ["Measure", "Value", f"{level:.0%} CI low", f"{level:.0%} CI high"], "rows": hrows})
        head.insert(1, {"title": "Comparison of survival curves", "columns": ["Parameter", "Value"], "rows": rows})
        head.insert(2, {"title": "Observed and expected events (logrank)", "columns": ["Group", "Observed", "Expected"],
                        "rows": [[titles[i], O[q], E[q]] for q, i in enumerate(ids)]})
        res["comparison"] = cmp_
    head.append({"title": "Number at risk", "columns": ["Time"] + [titles[i] for i in ids], "rows": [[a] + [at_risk[i][q] for i in ids] for q, a in enumerate(art)]})
    res["messages"], res["display"] = msgs, head + disp
    return res


# =========================================================================== Cox regression

def _design(cols):
    """cols: [(name, values, categories or None)] -> (X, term names, notes). Categorical: dummies vs first present level."""
    xs, names, notes = [], [], []
    for name, v, cats in cols:
        if cats is None:
            xs.append(v)
            names.append(name)
        else:
            lv = np.unique(v)
            if len(lv) < 2:
                raise _Bad(f"{name}: a categorical covariate needs at least 2 levels among the rows used")
            notes.append(f"{name}: reference level {cats[int(lv[0])]}")
            for q in lv[1:]:
                xs.append((v == q).astype(float))
                names.append(f"{name}: {cats[int(q)]}")
    return np.column_stack(xs), names, notes


def _cox(time, event, X, names, ties, level):
    from statsmodels.duration.hazard_regression import PHReg
    n, k = X.shape
    if event.sum() == 0:
        raise _Bad("no events")
    if event.sum() <= k:
        raise _Bad(f"{int(event.sum())} events for {k} covariate term(s); more events than terms are needed")
    if (X.std(axis=0) == 0).any():
        raise _Bad("a covariate is constant among the rows used")
    msgs = []
    mod = PHReg(time, X, status=event, ties=ties)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = mod.fit(disp=False)
        except np.linalg.LinAlgError as e:
            raise _Bad("the Cox model is singular (collinear covariates)") from e
    if any("onverge" in str(x.message) for x in w):
        msgs.append("the optimizer reported a convergence warning")
    b, se = np.asarray(r.params), np.asarray(r.bse)
    if not np.isfinite(b).all() or not np.isfinite(se).all() or (np.abs(b) > 25).any():
        raise _Bad("the Cox fit diverged (perfect separation or a collinear covariate)")
    z = _z(level)
    ll, ll0 = float(mod.loglike(b)), float(mod.loglike(np.zeros(k)))
    cov = np.linalg.inv(-mod.hessian(b))
    s0, h0 = np.asarray(mod.score(np.zeros(k))), -np.asarray(mod.hessian(np.zeros(k)))
    lrt, wald, score = 2 * (ll - ll0), float(b @ np.linalg.solve(cov, b)), float(s0 @ np.linalg.solve(h0, s0))
    tests = {nm: {"chi2": s, "df": k, "p": float(stats.chi2.sf(s, k))} for nm, s in (("likelihood_ratio", lrt), ("wald", wald), ("score", score))}
    coef = [{"term": nm, "estimate": float(b[j]), "se": float(se[j]), "ci_low": float(b[j] - z * se[j]), "ci_high": float(b[j] + z * se[j]),
             "stat": float(b[j] / se[j]), "p": float(2 * stats.norm.sf(abs(b[j] / se[j]))), "hr": float(math.exp(b[j])),
             "hr_ci_low": float(math.exp(b[j] - z * se[j])), "hr_ci_high": float(math.exp(b[j] + z * se[j]))} for j, nm in enumerate(names)]
    return coef, tests, ll, ll0, msgs


@_kind("cox_regression")
def cox_regression(table, ids, p):
    level, ties = p["ci_level"], p["ties"]
    _check_level(level)
    msgs = []
    if table.kind == "Survival":
        grp = _subjects(table, ids)
        if len(grp) < 2:
            raise _Bad("Cox regression on a Survival table needs at least 2 groups")
        gids = list(grp)
        time = np.concatenate([grp[i][0] for i in gids])
        event = np.concatenate([grp[i][1] for i in gids])
        label = np.concatenate([np.full(len(grp[i][0]), q) for q, i in enumerate(gids)])
        X, names, notes = _design([("Group", label.astype(float), [_title(table, i) for i in gids])])
    else:
        if table.kind != "MultipleVariables":
            raise _Bad("Cox regression needs a Survival or Multiple variables table")
        tv, ev, covs = p["time_var"], p["event_var"], list(p["covariates"] or [])
        if not tv or not ev or not covs:
            raise _Bad("choose a time variable, an event variable and at least one covariate")
        for nm in (tv, ev, *covs):
            table.dataset(nm)
        if len({tv, ev, *covs}) != 2 + len(covs):
            raise _Bad("time, event and covariate variables must all differ")
        col = lambda i: np.where(table.dataset(i).excluded[:, 0], np.nan, table.dataset(i).values[:, 0])  # noqa: E731
        t, e = col(tv), col(ev)
        if table.dataset(tv).categories is not None:
            raise _Bad("the time variable must be numeric")
        evds = table.dataset(ev)
        if evds.categories is not None:
            lab = p["event_label"]
            if lab not in evds.categories:
                raise _Bad("choose which category of the event variable is the event")
            e = np.where(np.isnan(e), np.nan, (e == evds.categories.index(lab)).astype(float))
        elif not np.all(np.isin(e[~np.isnan(e)], (0, 1))):
            raise _Bad("the event variable must be coded 1 (event) / 0 (censored)")
        cv = [col(i) for i in covs]
        ok = ~np.isnan(t) & ~np.isnan(e) & ~np.any([np.isnan(c) for c in cv], axis=0)
        if (t[ok] < 0).any():
            raise _Bad("negative time")
        if ok.sum() < table.n_rows:
            msgs.append(f"{int(table.n_rows - ok.sum())} row(s) with a blank in a used variable were dropped")
        time, event = t[ok], e[ok].astype(int)
        X, names, notes = _design([(_title(table, i), c[ok], table.dataset(i).categories) for i, c in zip(covs, cv)])
    coef, tests, ll, ll0, m2 = _cox(time, event, X, names, ties, level)
    msgs += m2
    pv = lambda x: p_format(x)  # noqa: E731
    disp = [{"title": f"Cox proportional hazards regression ({ties} ties)", "columns": ["Term", "Coefficient", "SE", "Z", "P value", "Hazard ratio", f"HR {level:.0%} CI low", f"HR {level:.0%} CI high"],
             "rows": [[c["term"], c["estimate"], c["se"], c["stat"], pv(c["p"]), c["hr"], c["hr_ci_low"], c["hr_ci_high"]] for c in coef]},
            {"title": "Model tests", "columns": ["Test", "Chi-square", "DF", "P value"], "rows": [[k.replace("_", " "), v["chi2"], v["df"], pv(v["p"])] for k, v in tests.items()]},
            {"title": "Data", "columns": ["Parameter", "Value"], "rows": [["Subjects", len(time)], ["Events", int(event.sum())], ["Log-likelihood", ll], ["Null log-likelihood", ll0]] + [["Reference", n_] for n_ in notes]}]
    return {"messages": msgs, "coefficients": coef, "tests": tests, "loglik": ll, "loglik_null": ll0, "ties": ties,
            "n": len(time), "events": int(event.sum()), "display": disp}


# =========================================================================== area under the curve

def _auc_series(table, ds_id):
    s = table.summary(ds_id)
    x = table.x
    ok = ~np.isnan(x) & ~np.isnan(s["mean"])
    o = np.argsort(x[ok], kind="stable")
    return x[ok][o], s["mean"][ok][o], s["sd"][ok][o], s["n"][ok][o]


def _trap(x, y):
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def _cross(x, y, i, j, b):
    """X where the segment i -> j crosses the baseline (y_i, y_j on opposite sides or one on it)."""
    return x[i] if y[i] == b else x[j] if y[j] == b else x[i] + (b - y[i]) * (x[j] - x[i]) / (y[j] - y[i])


_AUC_K = 0.75  # Prism's variance of an area: K * sum over trapezoids of width^2 * (var_a + var_b), var = SD^2 / n; K fitted at n = 3 (all evidence)


def _auc_one(x, y, sd, n, b, pct, direction, min_pts, z):
    msgs = []
    yr = float(y.max() - y.min())
    with np.errstate(divide="ignore", invalid="ignore"):
        vm = sd ** 2 / n  # variance of each mean
    # Prism: total area = sum of |area| over the whole curve (positive peaks + |negative peaks| + peaks too small to count);
    # trapezoids are split where the curve crosses the baseline, the crossing point itself carries no variance
    total, var = 0.0, 0.0
    for i in range(len(x) - 1):
        da, db, w = y[i] - b, y[i + 1] - b, x[i + 1] - x[i]
        if da * db < 0:
            f = da / (da - db)
            parts = [(w * f, abs(da), vm[i], 0.0), (w * (1 - f), abs(db), 0.0, vm[i + 1])]
        else:
            parts = [(w, abs(da) + abs(db), vm[i], vm[i + 1])]
        for pw, h, va, vb in parts:
            total += pw * h / 2
            var += pw ** 2 * (va + vb)
    # peaks: runs of points strictly above (or below) the baseline
    peaks, sign = [], np.sign(y - b)
    q = 0
    while q < len(x):
        s = sign[q]
        if s == 0 or (s < 0 and direction == "above"):
            q += 1
            continue
        r = q
        while r + 1 < len(x) and sign[r + 1] == s:
            r += 1
        px = np.concatenate([[x[q] if q == 0 else _cross(x, y, q - 1, q, b)], x[q:r + 1], [x[r] if r == len(x) - 1 else _cross(x, y, r, r + 1, b)]])
        py = np.concatenate([[b if q > 0 else y[q]], y[q:r + 1], [b if r < len(x) - 1 else y[r]]])
        k = q + int(np.argmax(np.abs(y[q:r + 1] - b)))
        if abs(y[k] - b) >= pct / 100 * yr and r - q + 1 >= min_pts:
            v0 = vm[q] if q == 0 else (vm[q - 1] if y[q - 1] == b else 0.0)  # a baseline crossing carries no variance
            v1 = vm[r] if r == len(x) - 1 else (vm[r + 1] if y[r + 1] == b else 0.0)
            vp = np.concatenate([[v0], vm[q:r + 1], [v1]])
            peaks.append({"x_start": float(px[0]), "x_end": float(px[-1]), "x_peak": float(x[k]), "y_peak": float(y[k]), "area": _trap(px, py - b),
                          "se": math.sqrt(_AUC_K * float(np.sum(np.diff(px) ** 2 * (vp[:-1] + vp[1:])))) if np.isfinite(vp).all() else None})
        q = r + 1
    tot_peak = sum(abs(pk["area"]) for pk in peaks)
    for pk in peaks:
        pk["fraction"] = abs(pk["area"]) / tot_peak if tot_peak else None
    out = {"total": total, "baseline": float(b), "peaks": peaks, "total_peak_area": tot_peak, "net_area": sum(pk["area"] for pk in peaks),
           "x_start": float(x[0]), "x_end": float(x[-1]), "se": None, "ci_low": None, "ci_high": None}
    if np.all(n >= 2) and np.isfinite(vm).all():
        se = math.sqrt(_AUC_K * var)
        out.update(se=se, ci_low=total - z * se, ci_high=total + z * se)
    else:
        msgs.append("SE and CI need at least 2 replicates (or SD and N) at every point")
    return out, msgs


@_kind("auc")
def auc(table, ids, p):
    level = p["ci_level"]
    _check_level(level)
    if not 0 <= p["min_peak_height_pct"] < 100:
        raise _Bad("minimum peak height must be a percentage between 0 and 100")
    z = _z(level)  # Prism's AUC CI is AUC +- z SE with the exact normal z (1.959964), not t
    res, msgs, disp = {}, [], []
    for i in ids:
        x, y, sd, n = _auc_series(table, i)
        title = _title(table, i)
        if len(x) < 2:
            raise _Bad(f"{title}: at least 2 points are needed")
        if np.ptp(x) == 0:
            raise _Bad(f"{title}: all X values are identical")
        bm = p["baseline_source"]
        b = {"value": p["baseline"], "first_y": y[0], "mean_y": float(y.mean()), "min_y": float(y.min())}[bm]
        if b is None:
            raise _Bad("enter the baseline value")
        r, m = _auc_one(x, y, sd, n, float(b), p["min_peak_height_pct"], p["peak_direction"], int(p["min_peak_points"]), z)
        res[i] = r
        msgs += [f"{title}: {t}" for t in m]
        disp.append({"title": f"Area under the curve: {title}", "columns": ["Parameter", "Value"],
                     "rows": [["Baseline", r["baseline"]], ["Total area", r["total"]], ["SE", r["se"]], [f"{level:.0%} CI low", r["ci_low"]], [f"{level:.0%} CI high", r["ci_high"]],
                              ["Total peak area", r["total_peak_area"]]] + ([["Net area", r["net_area"]]] if p["peak_direction"] == "both" else [])})
        disp.append({"title": f"Peaks: {title}", "columns": ["Peak", "Start X", "End X", "Peak X", "Peak Y", "Area", "Fraction of total peak area"],
                     "rows": [[q + 1, pk["x_start"], pk["x_end"], pk["x_peak"], pk["y_peak"], pk["area"], pk["fraction"]] for q, pk in enumerate(r["peaks"])]})
    return {"messages": msgs, "auc": res, "n": {i: int(len(_auc_series(table, i)[0])) for i in ids}, "display": disp}


# =========================================================================== frequency distribution

@_kind("frequency_distribution")
def frequency_distribution(table, ids, p):
    vals = {}
    for i in ids:
        if table.y_format != "replicates":
            raise _Bad("a frequency distribution needs raw values, not mean/SD/n")
        if table.kind == "XY" and p["replicates"] == "mean":
            v = table.summary(i)["mean"]
            v = v[~np.isnan(v)]
        else:
            v = table.points(i)[1]
        if len(v) == 0:
            raise _Bad(f"{_title(table, i)}: no values")
        vals[i] = v
    allv = np.concatenate(list(vals.values()))
    lo_v, hi_v = float(allv.min()), float(allv.max())
    w = p["bin_width"]
    if w is None:
        # Prism FAQ 1660: mean over the data sets of range / (1 + log2 n), rounded down to 1, 2 or 5 x 10^k, to an integer
        # (at least 1) when every value is an integer
        w = float(np.mean([(v.max() - v.min()) / (1 + math.log2(len(v))) for v in vals.values()]))
        w = _floor_nice(w) if w > 0 else 1.0
        if np.all(allv == np.floor(allv)):
            w = max(1.0, math.floor(w))
    if not w > 0:
        raise _Bad("bin width must be positive")
    # Prism: bins are centred on multiples of the width, the first one holds the minimum (centre = round(min / w) * w)
    first = (p["first_bin_center"] if p["first_bin_center"] is not None else math.floor(lo_v / w + 0.5) * w) - w / 2
    if p["last_bin_center"] is not None:
        nb = int(round((p["last_bin_center"] - first - w / 2) / w)) + 1
    else:
        nb = int(math.floor((hi_v - first) / w + 1e-9)) + 1
    if nb < 1:
        raise _Bad("the last bin centre lies below the first bin")
    if nb > 10000:
        raise _Bad("more than 10000 bins; increase the bin width")
    edges = first + w * np.arange(nb + 1)
    res, msgs, disp = {}, [], []
    rel = p["relative"]
    for i, v in vals.items():
        k = np.searchsorted(edges, v, "right") - 1
        inb = (k >= 0) & (k < nb)
        counts = np.bincount(k[inb], minlength=nb)
        tot = counts.sum()
        omitted = int((~inb).sum())
        if omitted:
            msgs.append(f"{_title(table, i)}: {omitted} value(s) outside the bins were omitted")
        if tot == 0:
            raise _Bad(f"{_title(table, i)}: no value falls in the bins")
        relative = counts / tot * (100.0 if rel == "percent" else 1.0)
        cum = np.cumsum(relative if rel != "none" else counts)
        res[i] = {"edges": edges, "counts": counts, "relative": relative, "cumulative": cum, "bin_width": w, "n": int(tot), "omitted": omitted}
        disp.append({"title": f"Frequency distribution: {_title(table, i)}", "columns": ["Bin centre", "Bin start", "Bin end", "Count", "Relative frequency (%)" if rel == "percent" else "Relative frequency", "Cumulative"],
                     "rows": [[(edges[q] + edges[q + 1]) / 2, edges[q], edges[q + 1], int(counts[q]), relative[q], cum[q]] for q in range(nb)]})
    return {"messages": msgs, "histograms": res, "bin_width": w, "n": {i: res[i]["n"] for i in res}, "display": disp}


# =========================================================================== three-way ANOVA

def _type3(long):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    return sm.stats.anova_lm(ols("y ~ C(a, Sum) * C(b, Sum) * C(c, Sum)", long).fit(), typ=3)


@_kind("three_way_anova")
def three_way_anova(table, ids, p):
    import pandas as pd
    msgs = []
    if len(ids) < 4:
        raise _Bad("three-way ANOVA needs four data sets: A1B1, A1B2, A2B1, A2B2")
    if len(ids) > 4:
        msgs.append("only the first four data sets are used (A1B1, A1B2, A2B1, A2B2)")
        ids = ids[:4]
    cells, used = _cells(table, ids)
    R = len(cells)
    if R < 2:
        raise _Bad("factor C (rows) needs at least 2 levels")
    rt = _row_titles(table, used)
    for r in range(R):
        for j in range(4):
            if len(cells[r][j]) == 0:
                raise _Bad(f"empty cell: {rt[r]} / {_title(table, ids[j])}; every combination needs values")
    nn = np.array([[len(cells[r][j]) for j in range(4)] for r in range(R)], float)
    mm = np.array([[cells[r][j].mean() for j in range(4)] for r in range(R)])
    N = int(nn.sum())
    dfe = N - 4 * R
    if dfe < 1:
        raise _Bad("no replicates: the residual has zero degrees of freedom")
    sse = float(sum(((x - x.mean()) ** 2).sum() for row in cells for x in row))
    if sse <= 0:
        raise _Bad("zero variance within every cell; F is undefined")
    long = pd.DataFrame([(j // 2, j % 2, r, v) for r in range(R) for j in range(4) for v in cells[r][j]], columns=["a", "b", "c", "y"])
    tab = _type3(long)
    A, B, Cn = "C(a, Sum)", "C(b, Sum)", "C(c, Sum)"
    fa, fb, fc = p["factor_a"], p["factor_b"], p["factor_c"]
    names = [(fa, A), (fb, B), (fc, Cn), (f"{fa} x {fb}", f"{A}:{B}"), (f"{fa} x {fc}", f"{A}:{Cn}"), (f"{fb} x {fc}", f"{B}:{Cn}"), (f"{fa} x {fb} x {fc}", f"{A}:{B}:{Cn}")]
    sst = float(((long["y"] - long["y"].mean()) ** 2).sum())
    mse = sse / dfe
    terms, rows = {}, []
    for lab, key in names:
        ss, df_, f_, pv = (float(tab.loc[key, "sum_sq"]), int(tab.loc[key, "df"]), float(tab.loc[key, "F"]), float(tab.loc[key, "PR(>F)"]))
        terms[lab] = {"ss": ss, "df": df_, "ms": ss / df_, "f": f_, "p": pv, "pct_total": 100 * ss / sst}
        rows.append([lab, 100 * ss / sst, ss, df_, ss / df_, f"F ({df_}, {dfe}) = {f_:.4g}", p_format(pv), p_summary(pv)])
    terms["Residual"] = {"ss": sse, "df": dfe, "ms": mse, "pct_total": 100 * sse / sst}
    rows += [["Residual", 100 * sse / sst, sse, dfe, mse, None, None, None], ["Total", 100.0, sst, N - 1, None, None, None, None]]
    if np.ptp(nn) > 0:
        msgs.append("unbalanced design: type III sums of squares; components need not add up to the total")
    res = {"messages": msgs, "n": {i: int(nn[:, j].sum()) for j, i in enumerate(ids)}, "terms": terms, "ss_total": sst, "df_total": N - 1,
           "ss_type": "III", "mse": mse, "df_residual": dfe,
           "display": [{"title": "Three-way ANOVA (ordinary)", "columns": ["Source of variation", "% of total variation", "SS", "DF", "MS", "F (DFn, DFd)", "P value", "Summary"], "rows": rows}]}
    if p["multiple_comparisons"] == "none":
        return res
    level, corr = p["ci_level"], p["correction"]
    _check_level(level)
    comps, mrows = [], []
    # marginal means = unweighted means of the cell means; contrast weights give each cell 1 / (cells per level)
    cellmap = [(r, j) for r in range(R) for j in range(4)]
    for fac, name, lv in (("a", fa, [[(r, j) for (r, j) in cellmap if j // 2 == q] for q in range(2)]),
                          ("b", fb, [[(r, j) for (r, j) in cellmap if j % 2 == q] for q in range(2)]),
                          ("c", fc, [[(r, j) for (r, j) in cellmap if r == q] for q in range(R)])):
        K = len(lv)
        mu = [np.mean([mm[c] for c in cs]) for cs in lv]
        pairs = list(combinations(range(K), 2))
        d = np.array([mu[i] - mu[j] for i, j in pairs])
        se = np.sqrt(mse * np.array([sum(1 / nn[c] for c in lv[i]) / len(lv[i]) ** 2 + sum(1 / nn[c] for c in lv[j]) / len(lv[j]) ** 2 for i, j in pairs]))
        cr = ("tukey" if K > 2 else "bonferroni") if corr == "auto" else corr
        pu, pa, lo, hi = _mc_adjust(d, se, dfe, cr, level, K)
        lab = (lambda q: rt[q]) if fac == "c" else (lambda q: f"{name} level {q + 1}")
        for q, (i, j) in enumerate(pairs):
            cid = f"{fac}:{i}__{j}"
            comps.append(_comp(cid, f"{fac}{i}", f"{fac}{j}", d[q], lo[q], hi[q], pu[q], pa[q], se=se[q], factor=name))
            mrows.append([name, f"{lab(i)} vs. {lab(j)}", d[q], lo[q], hi[q], p_summary(pa[q]), p_format(pa[q])])
    res["comparisons"] = comps
    res["diff_label"] = "level i - level j"
    res["display"].append({"title": "Multiple comparisons: main effects", "columns": ["Factor", "Comparison", "Mean diff.", "CI low", "CI high", "Summary", "Adjusted P"], "rows": mrows})
    return res


# =========================================================================== registry

_CI = Param("ci_level", "Confidence level", "float", 0.95)

KINDS = {
    "contingency": AnalysisKind("contingency", "Contingency table analysis", ("Contingency",), contingency, (
        Param("yates", "Yates' continuity correction for chi-square (2x2)", "bool", False),
        Param("fisher_exact", "Fisher-Freeman-Halton exact test (larger tables)", "bool", False),
        Param("rr_method", "Relative risk CI", "choice", "koopman", (("koopman", "Koopman asymptotic score"), ("katz", "Katz (log)"))),
        Param("or_method", "Odds ratio CI", "choice", "baptista_pike", (("baptista_pike", "Baptista-Pike (exact)"), ("woolf", "Woolf (logit)"))),
        Param("diff_method", "Difference between proportions CI", "choice", "newcombe_cc",
              (("newcombe_cc", "Newcombe/Wilson with continuity correction"), ("newcombe", "Newcombe/Wilson"), ("wald", "Asymptotic (Wald)"))),
        Param("prop_ci", "Sensitivity / specificity CI", "choice", "wilson_brown", (("wilson_brown", "Hybrid Wilson/Brown"), ("exact", "Clopper-Pearson (exact)"))),
        Param("percent_of", "Show percentages of", "choice", "row", (("row", "Row totals"), ("column", "Column totals"), ("total", "Grand total"), ("none", "None"))),
        _CI)),
    "survival": AnalysisKind("survival", "Survival analysis (Kaplan-Meier)", ("Survival",), survival, (
        Param("ci_method", "Survival curve CI", "choice", "transformation", (("transformation", "Asymmetrical (log-log transformation)"), ("greenwood", "Symmetrical (Greenwood)"))),
        Param("units", "Survival reported as", "choice", "fraction", (("fraction", "Fraction"), ("percent", "Percent"))),
        Param("logrank_method", "P value for 3 or more curves", "choice", "recommended", (("recommended", "Recommended (matches SPSS / SAS)"), ("conservative", "Conservative (Prism 5 and earlier)"))),
        Param("at_risk_times", "Number-at-risk times (comma separated; blank = automatic)", "str", ""),
        _CI)),
    "cox_regression": AnalysisKind("cox_regression", "Cox proportional hazards regression", ("Survival", "MultipleVariables"), cox_regression, (
        Param("ties", "Ties", "choice", "efron", (("efron", "Efron"), ("breslow", "Breslow"))),
        Param("time_var", "Time variable (Multiple variables table)", "dataset", None),
        Param("event_var", "Event variable (1 = event, 0 = censored)", "dataset", None),
        Param("event_label", "Event category (categorical event variable)", "str", ""),
        Param("covariates", "Covariates (Multiple variables table)", "datasets", None),
        _CI)),
    "auc": AnalysisKind("auc", "Area under the curve", ("XY",), auc, (
        Param("baseline_source", "Baseline", "choice", "value", (("value", "Y = value"), ("first_y", "Y of the first point"), ("mean_y", "Mean of all Y"), ("min_y", "Minimum Y"))),
        Param("baseline", "Baseline value", "float", 0.0),
        Param("min_peak_height_pct", "Ignore peaks smaller than (% of max - min)", "float", 10.0),
        Param("min_peak_points", "Ignore peaks defined by fewer than N points", "int", 1),
        Param("peak_direction", "Peaks", "choice", "above", (("above", "Above baseline only"), ("both", "Above and below baseline"))),
        _CI)),
    "frequency_distribution": AnalysisKind("frequency_distribution", "Frequency distribution", ("Column", "XY"), frequency_distribution, (
        Param("bin_width", "Bin width (blank = automatic)", "float?", None),
        Param("first_bin_center", "Center of the first bin (blank = automatic)", "float?", None),
        Param("last_bin_center", "Center of the last bin (blank = automatic)", "float?", None),
        Param("relative", "Relative frequencies", "choice", "none", (("none", "Counts"), ("fraction", "Fractions"), ("percent", "Percentages"))),
        Param("replicates", "XY tables: replicates", "choice", "each", (("each", "Each replicate"), ("mean", "Mean of replicates")))), ),
    "three_way_anova": AnalysisKind("three_way_anova", "Three-way ANOVA", ("Grouped",), three_way_anova, (
        Param("factor_a", "Name of factor A (data sets 1-2 vs 3-4)", "str", "A"),
        Param("factor_b", "Name of factor B (odd vs even data sets)", "str", "B"),
        Param("factor_c", "Name of factor C (rows)", "str", "C"),
        Param("multiple_comparisons", "Multiple comparisons", "choice", "none", (("none", "None"), ("main_effects", "Levels of each factor (main effects)"))),
        Param("correction", "Correction", "choice", "auto", (("auto", "Prism default"), ("tukey", "Tukey"), ("sidak", "Sidak"), ("bonferroni", "Bonferroni"))),
        _CI)),
}
