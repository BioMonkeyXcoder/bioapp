"""Significance tests on Column and Grouped tables, following Prism's definitions.

Prism conventions used here (from the Prism statistics guide):
  * percentiles: rank = p*(n+1), linear interpolation, clamped to min / max;
  * skewness G1 and excess kurtosis G2 (bias-corrected, as scipy ``bias=False``);
  * t test difference = mean(B) - mean(A); paired pairing r reported with a one-tailed P;
  * Mann-Whitney / Wilcoxon: exact P (also with ties) when the smaller group has <= 100 values /
    fewer than 200 pairs, otherwise a normal approximation;
  * multiple comparisons use the ANOVA pooled MSE and its df; Holm-Sidak has no CI;
  * two-way ANOVA uses type III sums of squares;
  * Dunn: mean rank differences, tie-corrected SE, P multiplied by the number of comparisons.

Every function returns the result contract of DESIGN.md; invalid input gives status "failed".
Mean/SD/n tables are handled by rebuilding synthetic data with the same n, mean and SD, which
leaves every ANOVA / unpaired t quantity unchanged.
"""
from __future__ import annotations

import math
from itertools import combinations, permutations

import numpy as np
from scipy import optimize, stats
from scipy.integrate import quad
from scipy.special import comb

from . import AnalysisKind, Param


# --------------------------------------------------------------------------- formatting

def p_summary(p) -> str:
    """Prism GP style: ns P>0.05, * <=0.05, ** <=0.01, *** <=0.001, **** <=0.0001."""
    if p is None or not np.isfinite(p):
        return ""
    return "ns" if p > 0.05 else "*" if p > 0.01 else "**" if p > 0.001 else "***" if p > 0.0001 else "****"


def p_format(p) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return "<0.0001" if p < 0.0001 else ">0.9999" if p > 0.9999 else f"{p:.4f}"


# --------------------------------------------------------------------------- plumbing

class _Bad(Exception):
    """Invalid input; the message becomes the failure reason."""


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    return o


def _safe_n(table, ids) -> dict:
    try:
        return {i: int(np.sum(~np.isnan(np.where(table.dataset(i).excluded, np.nan, table.dataset(i).values)))) for i in ids}
    except Exception:  # noqa: BLE001 - only used to decorate a failure report
        return {}


def _kind(name):
    def deco(f):
        def run(table, ids, params):
            ids = list(ids) or [d.id for d in table.datasets]
            try:
                res = f(table, ids, params)
            except _Bad as e:
                return {"kind": name, "status": "failed", "messages": [str(e)], "display": [], "n": _safe_n(table, ids)}
            res["kind"] = name
            res["status"] = "warning" if res.get("messages") else "ok"
            return _clean(res)
        run.__name__ = f.__name__
        return run
    return deco


def _synth(mean, sd, n):
    """n values with exactly this sample mean and SD (n=1: the mean)."""
    if n == 1:
        return np.array([mean])
    z = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    return mean + sd * (z - z.mean()) / z.std(ddof=1)


def _col_vec(table, ds_id):
    ds = table.dataset(ds_id)
    return np.where(ds.excluded[:, 0], np.nan, ds.values[:, 0])


def _summary_arrays(table, ds_id):
    s = table.summary(ds_id)
    if np.any(np.isnan(s["n"][~np.isnan(s["mean"])])):
        raise _Bad("mean/SD or mean/SEM input needs N; use a mean, SD, N layout")
    return s


def _groups(table, ids, summary_ok=False):
    """Column table -> [(id, title, values)]; blanks and excluded cells dropped."""
    out = []
    if table.y_format == "replicates":
        for i in ids:
            v = _col_vec(table, i)
            out.append((i, table.dataset(i).title, v[~np.isnan(v)]))
        return out
    if not summary_ok:
        raise _Bad("this test needs raw values; it cannot run on mean/SD/n data")
    for i in ids:
        title = table.dataset(i).title
        s = _summary_arrays(table, i)
        ok = ~np.isnan(s["mean"])
        if ok.sum() != 1:
            raise _Bad(f"{title}: mean/SD/n input needs exactly one filled row per data set")
        m, sd, n = s["mean"][ok][0], s["sd"][ok][0], int(round(s["n"][ok][0]))
        if n < 1 or (n >= 2 and not sd >= 0):
            raise _Bad(f"{title}: invalid N or SD")
        out.append((i, title, _synth(m, 0.0 if n == 1 else sd, n)))
    return out


def _need(groups, kmin, nmin, what=""):
    if len(groups) < kmin:
        raise _Bad(f"select at least {kmin} data sets{what}")
    for _, title, x in groups:
        if len(x) < nmin:
            raise _Bad(f"{title}: {len(x)} value(s); at least {nmin} needed")


def _paired(table, ids, kmin, kmax=None):
    """Row-aligned complete rows of raw Column data -> (matrix, titles, n per data set, rows dropped)."""
    if table.y_format != "replicates":
        raise _Bad("paired / repeated-measures tests need raw values, not mean/SD/n")
    if len(ids) < kmin or (kmax and len(ids) > kmax):
        raise _Bad(f"select {kmin if kmax == kmin else f'{kmin} or more'} data sets")
    m = np.column_stack([_col_vec(table, i) for i in ids])
    blank = np.isnan(m)
    keep = ~blank.any(axis=1)
    dropped = int((blank.any(axis=1) & ~blank.all(axis=1)).sum())
    n = {i: int((~blank[:, j]).sum()) for j, i in enumerate(ids)}
    return m[keep], [table.dataset(i).title for i in ids], n, dropped


def _comp(id_, a, b, diff, lo, hi, p, padj, row=None, **extra):
    return {"id": id_, "row": row, "a": a, "b": b, "diff": diff, "ci_low": lo, "ci_high": hi,
            "p": p, "p_adj": padj, "summary": p_summary(padj), **extra}


def _mc_table(title, comps, titles, diff_label, with_row=False, row_titles=None):
    cols = (["Row"] if with_row else []) + ["Comparison", diff_label, "CI low", "CI high", "Summary", "Adjusted P"]
    rows = []
    for c in comps:
        r = [f"{titles[c['a']]} vs. {titles[c['b']]}", c["diff"], c["ci_low"], c["ci_high"], c["summary"], p_format(c["p_adj"])]
        rows.append(([row_titles[c["row"]]] if with_row else []) + r)
    return {"title": title, "columns": cols, "rows": rows}


# --------------------------------------------------------------------------- distributions

def _perm_sum_dist(vals, n1):
    """Null distribution of the sum of a random n1-subset of integer ``vals`` (ties allowed).
    Returns probabilities for sums 0..sum(vals)."""
    vals = np.asarray(vals, dtype=int)
    u, cnt = np.unique(vals, return_counts=True)
    S = int(vals.sum())
    dp = np.zeros((n1 + 1, S + 1))
    dp[0, 0] = 1.0
    s_done = k_done = 0   # only the reachable corner (values ascending) is updated
    for v, c in zip(u, cnt):
        c = int(c)
        s_new, k_new = s_done + int(v) * c, min(n1, k_done + c)
        new = dp[:k_new + 1, :s_new + 1].copy()
        for j in range(1, min(c, n1) + 1):
            new[j:, j * v:] += comb(c, j) * dp[:k_new + 1 - j, :s_new + 1 - j * v]
        dp[:k_new + 1, :s_new + 1] = new
        s_done, k_done = s_new, k_new
    return dp[n1] / dp[n1].sum()


def _subset_sum_dist(vals):
    """Null distribution of the sum of a random subset (each item in with prob 1/2)."""
    vals = np.asarray(vals, dtype=int)
    S = int(vals.sum())
    dp = np.zeros(S + 1)
    dp[0] = 1.0
    for v in vals:
        new = dp.copy()
        new[v:] += dp[:S + 1 - v]
        dp = new / 2.0
    return dp


def _two_sided_exact(dist, s_obs, mean):
    s = np.arange(len(dist))
    return float(min(1.0, dist[np.abs(s - mean) >= abs(s_obs - mean) - 1e-9].sum()))


def _pick_k(cdf_km1, level):
    """Narrowest interval [D(k), D(m+1-k)] (largest k, 1-based) with confidence 1-2*cdf(k-1) >= level;
    k = 1 (the widest) when none reaches the level."""
    conf = 1 - 2 * cdf_km1
    k = max(int((conf >= level - 1e-12).sum()), 1)
    return k, float(conf[k - 1])


def _u_cdf(nA, nB):
    """P(U <= u), u = 0..nA*nB/2-ish, for the Mann-Whitney U without ties."""
    m = nA * nB
    if min(nA, nB) <= 100:
        n1, N = min(nA, nB), nA + nB
        d = _perm_sum_dist(np.arange(1, N + 1), n1)
        off = n1 * (n1 + 1) // 2
        return np.cumsum(d[off:off + m + 1])
    u = np.arange(m + 1)
    return stats.norm.cdf((u + 0.5 - m / 2) / math.sqrt(m * (nA + nB + 1) / 12))


def _spearman_p1(a, b, exact_max=10):
    """Spearman rs and its one-tailed P (H1: rs > 0). Exact permutation P (midranks, ties allowed) for
    n <= exact_max, otherwise SciPy's t approximation. The n! pairings are enumerated by splitting the
    positions in two halves (meet in the middle)."""
    n = len(a)
    r = float(stats.spearmanr(a, b).statistic)
    if n > exact_max:
        return r, float(stats.spearmanr(a, b, alternative="greater").pvalue)
    ra, rb = stats.rankdata(a), stats.rankdata(b)
    h = n // 2
    obs = ra @ rb - 1e-9
    base1, base2 = (np.array(list(permutations(range(k)))) for k in (h, n - h))
    hits = 0
    for sub in map(np.array, combinations(range(n), h)):
        rest = np.setdiff1d(np.arange(n), sub)
        s1 = rb[sub[base1]] @ ra[:h]            # first h positions take the values in sub
        s2 = rb[rest[base2]] @ ra[h:]
        hits += int((s1[:, None] + s2[None, :] >= obs).sum())
    return r, hits / math.factorial(n)


def _tie_term(x):
    _, c = np.unique(x, return_counts=True)
    return float((c ** 3 - c).sum())


# --------------------------------------------------------------------------- descriptive

def _pctl(xs, p):
    n = len(xs)
    r = p * (n + 1)
    if r < 1:
        return xs[0]
    if r >= n:
        return xs[-1]
    lo = int(r)
    return xs[lo - 1] + (r - lo) * (xs[lo] - xs[lo - 1])


DESC_ROWS = (("n", "Number of values"), ("mean", "Mean"), ("sd", "Std. deviation"), ("sem", "Std. error of mean"),
             ("median", "Median"), ("p25", "25% percentile"), ("p75", "75% percentile"), ("min", "Minimum"),
             ("max", "Maximum"), ("range", "Range"), ("ci_low", "Lower CI of mean"), ("ci_high", "Upper CI of mean"),
             ("cv", "Coefficient of variation (%)"), ("geo_mean", "Geometric mean"),
             ("geo_sd_factor", "Geometric SD factor"), ("skewness", "Skewness"), ("kurtosis", "Kurtosis"))


def _describe(x, level):
    n = len(x)
    d = {k: None for k, _ in DESC_ROWS}
    d["n"] = n
    if n == 0:
        return d
    xs = np.sort(x)
    d.update(mean=x.mean(), median=float(np.median(x)), p25=_pctl(xs, 0.25), p75=_pctl(xs, 0.75),
             min=xs[0], max=xs[-1], range=xs[-1] - xs[0])
    if n > 1:
        sd = x.std(ddof=1)
        sem = sd / math.sqrt(n)
        h = stats.t.ppf((1 + level) / 2, n - 1) * sem
        d.update(sd=sd, sem=sem, ci_low=d["mean"] - h, ci_high=d["mean"] + h)
        if d["mean"] != 0:
            d["cv"] = 100 * sd / abs(d["mean"])
        if sd > 0 and n >= 3:
            d["skewness"] = stats.skew(x, bias=False)
        if sd > 0 and n >= 4:
            d["kurtosis"] = stats.kurtosis(x, bias=False)
    if xs[0] > 0:
        lg = np.log(x)
        d["geo_mean"] = math.exp(lg.mean())
        if n > 1:
            d["geo_sd_factor"] = math.exp(lg.std(ddof=1))
    return d


@_kind("descriptive")
def descriptive(table, ids, p):
    g = _groups(table, ids)
    if all(len(x) == 0 for _, _, x in g):
        raise _Bad("all selected data sets are empty")
    level = p["ci_level"]
    res = {i: _describe(x, level) for i, _, x in g}
    msgs = [f"{t}: no values" for _, t, x in g if len(x) == 0]
    msgs += [f"{t}: only one value; SD, SEM and CI undefined" for _, t, x in g if len(x) == 1]
    rows = [[lab] + [res[i][k] for i, _, _ in g] for k, lab in DESC_ROWS]
    return {"messages": msgs, "datasets": res, "n": {i: res[i]["n"] for i in res},
            "display": [{"title": "Descriptive statistics", "columns": [""] + [t for _, t, _ in g], "rows": rows}]}


# --------------------------------------------------------------------------- two groups

@_kind("t_test")
def t_test(table, ids, p):
    level, paired = p["ci_level"], p["paired"]
    if len(ids) != 2:
        raise _Bad("a t test compares exactly two data sets")
    a_id, b_id = ids
    msgs = []
    if paired:
        m, titles, n, dropped = _paired(table, ids, 2, 2)
        a, b = m[:, 0], m[:, 1]
        npairs = len(a)
        if npairs < 2:
            raise _Bad(f"{npairs} complete pair(s); at least 2 needed")
        d = b - a
        sd = d.std(ddof=1)
        if sd == 0:
            raise _Bad("all differences are identical (zero variance); t is undefined")
        diff, se, df = d.mean(), sd / math.sqrt(npairs), npairs - 1
        if dropped:
            msgs.append(f"{dropped} row(s) with a blank in either data set were dropped")
        extra = [("Pairs", npairs), ("Rows dropped (blank cell)", dropped), ("SD of differences", sd)]
        pairing = {"n_pairs": npairs, "rows_dropped": dropped, "r": None, "p_one_tailed": None}
        if npairs >= 3 and a.std() > 0 and b.std() > 0:
            r = float(np.corrcoef(a, b)[0, 1])
            tr = r * math.sqrt((npairs - 2) / max(1 - r * r, 1e-300))
            pairing.update(r=r, p_one_tailed=float(stats.t.sf(tr, npairs - 2)))
        extra += [("Pairing correlation r", pairing["r"]), ("Pairing P (one-tailed)", p_format(pairing["p_one_tailed"]))]
        label, var_test = "Paired t test", None
    else:
        g = _groups(table, ids, summary_ok=True)
        _need(g, 2, 2)
        (_, ta, a), (_, tb, b) = g
        titles, n = [ta, tb], {a_id: len(a), b_id: len(b)}
        na, nb, va, vb = len(a), len(b), a.var(ddof=1), b.var(ddof=1)
        diff = b.mean() - a.mean()
        if p["welch"]:
            sea2, seb2 = va / na, vb / nb
            se = math.sqrt(sea2 + seb2)
            df = (sea2 + seb2) ** 2 / (sea2 ** 2 / (na - 1) + seb2 ** 2 / (nb - 1)) if se > 0 else None
        else:
            df = na + nb - 2
            sp2 = ((na - 1) * va + (nb - 1) * vb) / df
            se = math.sqrt(sp2 * (1 / na + 1 / nb))
        if se == 0:
            raise _Bad("zero variance in both data sets; t is undefined")
        hi_v, lo_v = (va, vb) if va >= vb else (vb, va)
        n_hi, n_lo = (na, nb) if va >= vb else (nb, na)
        ftest = {"f": None, "df_num": n_hi - 1, "df_den": n_lo - 1, "p": None}
        if lo_v > 0:
            f = hi_v / lo_v
            ftest.update(f=f, p=min(1.0, 2 * stats.f.sf(f, n_hi - 1, n_lo - 1)))
        extra = [("F (variance ratio)", ftest["f"]), ("F DFn, DFd", f"{n_hi - 1}, {n_lo - 1}"), ("F test P", p_format(ftest["p"]))]
        label = "Welch's unpaired t test" if p["welch"] else "Unpaired t test"
        pairing = None
    t = diff / se
    pval = float(2 * stats.t.sf(abs(t), df))
    h = stats.t.ppf((1 + level) / 2, df) * se
    r2 = t * t / (t * t + df)
    cmp_ = _comp(f"{a_id}__{b_id}", b_id, a_id, diff, diff - h, diff + h, pval, pval, t=t, df=df, se=se)
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    rows = [["P value (two-tailed)", p_format(pval)], ["P value summary", p_summary(pval)],
            ["Significantly different (P<0.05)?", "Yes" if pval <= 0.05 else "No"],
            ["t", t], ["df", df], [f"Mean of {titles[0]}", mean_a], [f"Mean of {titles[1]}", mean_b],
            [f"Difference ({titles[1]} - {titles[0]})", diff], ["SE of difference", se],
            [f"{level:.0%} CI of difference: low", diff - h], [f"{level:.0%} CI of difference: high", diff + h],
            ["R squared", r2]] + [list(e) for e in extra]
    res = {"messages": msgs, "n": n, "t": t, "df": df, "p": pval, "difference": diff, "se": se,
           "ci_low": diff - h, "ci_high": diff + h, "r2": r2, "means": {a_id: mean_a, b_id: mean_b},
           "diff_label": "B - A", "comparisons": [cmp_],
           "display": [{"title": label, "columns": ["Parameter", "Value"], "rows": rows}]}
    if paired:
        res["pairing"] = pairing
    else:
        res["variance_test"] = ftest
    return res


def _mw_p_exact_or_approx(x, y):
    nA, nB = len(x), len(y)
    allv = np.concatenate([x, y])
    N = nA + nB
    r = stats.rankdata(allv)
    uA = r[:nA].sum() - nA * (nA + 1) / 2
    uB = nA * nB - uA
    if min(nA, nB) <= 100:
        r2 = np.rint(2 * r).astype(int)
        small = slice(0, nA) if nA <= nB else slice(nA, N)
        n1 = min(nA, nB)
        dist = _perm_sum_dist(r2, n1)
        pval = _two_sided_exact(dist, r2[small].sum(), n1 * r2.sum() / N)
        return min(uA, uB), uA, pval, "exact"
    sd = math.sqrt(nA * nB / 12 * ((N + 1) - _tie_term(allv) / (N * (N - 1))))
    if sd == 0:
        return min(uA, uB), uA, 1.0, "approximate"
    z = (abs(uA - nA * nB / 2) - 0.5) / sd
    return min(uA, uB), uA, float(min(1.0, 2 * stats.norm.sf(max(z, 0)))), "approximate"


@_kind("mann_whitney")
def mann_whitney(table, ids, p):
    if len(ids) != 2:
        raise _Bad("the Mann-Whitney test compares exactly two data sets")
    g = _groups(table, ids)
    _need(g, 2, 1)
    (a_id, ta, x), (b_id, tb, y) = g
    nA, nB, level = len(x), len(y), p["ci_level"]
    msgs = []
    if np.ptp(np.concatenate([x, y])) == 0:
        msgs.append("all values are identical; P = 1")
    if nA + nB <= 7:
        msgs.append("total n <= 7: P cannot fall below 0.05 whatever the data")
    u, uA, pval, method = _mw_p_exact_or_approx(x, y)
    diffs = np.sort((y[:, None] - x[None, :]).ravel())
    m = len(diffs)
    hl = float(np.median(diffs))
    cdf = _u_cdf(nA, nB)
    ks = np.arange(1, (m + 1) // 2 + 1)
    k, conf = _pick_k(cdf[ks - 1], level)
    lo, hi = diffs[k - 1], diffs[m - k]
    if conf < level - 1e-12:
        msgs.append(f"requested {level:.0%} CI not attainable; reported interval has {conf:.1%} confidence")
    med ={a_id: float(np.median(x)), b_id: float(np.median(y))}
    rows = [["P value (two-tailed)", p_format(pval)], ["P value summary", p_summary(pval)], ["Method", method],
            ["Mann-Whitney U", u], [f"Median of {ta}", med[a_id]], [f"Median of {tb}", med[b_id]],
            [f"Difference between medians ({tb} - {ta})", med[b_id] - med[a_id]],
            [f"Hodges-Lehmann difference ({tb} - {ta})", hl],
            [f"CI of HL difference ({conf:.1%}): low", lo], ["CI of HL difference: high", hi]]
    return {"messages": msgs, "n": {a_id: nA, b_id: nB}, "u": u, "p": pval, "method": method, "medians": med,
            "median_difference": med[b_id] - med[a_id], "hodges_lehmann": hl, "hl_ci_low": lo, "hl_ci_high": hi,
            "hl_ci_confidence": conf, "diff_label": "B - A",
            "comparisons": [_comp(f"{a_id}__{b_id}", b_id, a_id, hl, lo, hi, pval, pval)],
            "display": [{"title": "Mann-Whitney test", "columns": ["Parameter", "Value"], "rows": rows}]}


@_kind("wilcoxon_paired")
def wilcoxon_paired(table, ids, p):
    m, titles, n, dropped = _paired(table, ids, 2, 2)
    a_id, b_id = ids
    level, pratt = p["ci_level"], p["zero_method"] == "pratt"
    msgs = []
    if dropped:
        msgs.append(f"{dropped} row(s) with a blank in either data set were dropped")
    d = m[:, 1] - m[:, 0]
    nz = d != 0
    if not nz.any():
        raise _Bad("all paired differences are zero")
    ranks = stats.rankdata(np.abs(d) if pratt else np.abs(d[nz]))
    r_nz = ranks[nz] if pratt else ranks
    sgn = np.sign(d[nz])
    w_pos, w_neg = r_nz[sgn > 0].sum(), r_nz[sgn < 0].sum()
    nn = int(nz.sum())
    if nn < 6:
        msgs.append("fewer than 6 non-zero pairs: P cannot fall below 0.05 whatever the data")
    if nz.sum() < len(d):
        msgs.append(f"{int((~nz).sum())} pair(s) with zero difference ({'Pratt' if pratt else 'Wilcoxon'} method)")
    r2 = np.rint(2 * r_nz).astype(int)
    if len(d) < 200:
        method = "exact"
        pval = _two_sided_exact(_subset_sum_dist(r2), 2 * w_pos, r2.sum() / 2)
    else:
        method = "approximate"
        z = (abs(w_pos - r_nz.sum() / 2) - 0.5) / math.sqrt((r_nz ** 2).sum() / 4)
        pval = float(min(1.0, 2 * stats.norm.sf(max(z, 0))))
    # Prism: median of all paired differences (zeros included) with the distribution-free (sign test)
    # CI [D(k), D(n+1-k)]; the result key stays "hodges_lehmann" for the validation harness.
    dd = np.sort(d)
    nt = len(dd)
    hl = float(np.median(dd))
    ks = np.arange(1, (nt + 1) // 2 + 1)
    k, conf = _pick_k(stats.binom.cdf(ks - 1, nt, 0.5), level)
    lo, hi = dd[k - 1], dd[nt - k]
    if conf < level - 1e-12:
        msgs.append(f"requested {level:.0%} CI not attainable; reported interval has {conf:.1%} confidence")
    rho = None
    if len(m) >= 3 and m[:, 0].std() > 0 and m[:, 1].std() > 0:
        rs, p1 = _spearman_p1(m[:, 0], m[:, 1])
        rho = {"rs": rs, "p_one_tailed": p1}
    rows = [["P value (two-tailed)", p_format(pval)], ["P value summary", p_summary(pval)], ["Method", method],
            ["Sum of positive ranks (B > A)", w_pos], ["Sum of negative ranks (B < A)", -w_neg],
            ["Sum of signed ranks (W)", w_pos - w_neg], ["Number of pairs", len(m)], ["Non-zero pairs", nn],
            ["Rows dropped (blank cell)", dropped],
            [f"Median of {titles[0]}", float(np.median(m[:, 0]))], [f"Median of {titles[1]}", float(np.median(m[:, 1]))],
            ["Median of differences (B - A)", hl],
            [f"CI ({conf:.1%}): low", lo], ["CI: high", hi],
            ["Spearman rs (pairing)", rho and rho["rs"]], ["Spearman P (one-tailed)", p_format(rho and rho["p_one_tailed"])]]
    return {"messages": msgs, "n": n, "n_pairs": len(m), "rows_dropped": dropped, "w": w_pos - w_neg,
            "sum_pos": w_pos, "sum_neg": w_neg, "p": pval, "method": method,
            "medians": {a_id: float(np.median(m[:, 0])), b_id: float(np.median(m[:, 1]))},
            "hodges_lehmann": hl, "hl_ci_low": lo, "hl_ci_high": hi, "hl_ci_confidence": conf,
            "pairing_spearman": rho, "diff_label": "B - A",
            "comparisons": [_comp(f"{a_id}__{b_id}", b_id, a_id, hl, lo, hi, pval, pval)],
            "display": [{"title": "Wilcoxon matched-pairs signed rank test", "columns": ["Parameter", "Value"], "rows": rows}]}


# --------------------------------------------------------------------------- multiple comparisons

def _mc_adjust(d, se, df, method, level, k=None):
    """Pooled-variance multiple comparisons. Returns (p, p_adj, lo, hi) arrays (lo/hi None for Holm-Sidak)."""
    d, se = np.asarray(d, float), np.asarray(se, float)
    m = len(d)
    t = np.abs(d) / se
    p = 2 * stats.t.sf(t, df)
    a = 1 - level
    if method == "none":
        padj, crit = p, stats.t.ppf(1 - a / 2, df)
    elif method == "bonferroni":
        padj, crit = np.minimum(1, m * p), stats.t.ppf(1 - a / (2 * m), df)
    elif method == "sidak":
        padj, crit = 1 - (1 - p) ** m, stats.t.ppf(1 - (1 - level ** (1 / m)) / 2, df)
    elif method == "holm_sidak":
        order = np.argsort(p)
        step = np.maximum.accumulate(1 - (1 - p[order]) ** (m - np.arange(m)))
        padj = np.empty(m)
        padj[order] = np.minimum(1, step)
        return p, padj, None, None
    elif method == "tukey":
        padj = stats.studentized_range.sf(t * math.sqrt(2), k, df)
        crit = stats.studentized_range.ppf(level, k, df) / math.sqrt(2)
    else:
        raise _Bad(f"unknown multiple comparisons method {method!r}")
    return p, padj, d - crit * se, d + crit * se


_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_E = np.linspace(-10, 10, 81)   # composite Gauss-Legendre over u in [-10, 10]: 80 panels x 8 nodes
_U = ((_E[:-1] + _E[1:])[:, None] / 2 + (_E[1:] - _E[:-1])[:, None] / 2 * _GL_X).ravel()
_UW = ((_E[1:] - _E[:-1])[:, None] / 2 * _GL_W).ravel() * stats.norm.pdf(_U)

def _maxt_sf(c, lam, df):
    """P(max_i |T_i| > c) for T_i = Z_i / (chi_df / sqrt(df)) with corr(Z_i, Z_j) = lam_i * lam_j.
    Z_i = sqrt(1-lam_i^2) V_i + lam_i U with V_i, U iid N(0, 1); lam = 0 gives the studentized maximum
    modulus. Deterministic: adaptive quadrature over the chi variable, Gauss-Legendre over U; the
    complement is summed in log space, so P stays accurate down to ~1e-14."""
    lam = np.asarray(lam, float)[:, None]
    sig = np.sqrt(1 - lam ** 2)
    u, w = (_U[None, :], _UW) if lam.any() else (np.zeros((1, 1)), np.ones(1))
    chi = stats.chi(df)

    def f(x):
        cs = c * x / math.sqrt(df)
        q = np.minimum(stats.norm.sf((cs - lam * u) / sig) + stats.norm.sf((cs + lam * u) / sig), 1 - 1e-16)
        return chi.pdf(x) * float(w @ -np.expm1(np.log1p(-q).sum(axis=0)))

    return min(1.0, quad(f, chi.ppf(1e-15), chi.ppf(1 - 1e-15), points=[chi.mean()], epsabs=0, epsrel=1e-11, limit=400)[0])


def _maxt_crit(lam, df, level):
    return optimize.brentq(lambda c: _maxt_sf(c, lam, df) - (1 - level), 1e-3, 60, xtol=1e-10)


def _dunnett_pooled(d, se, n_ctl, ns, df, level):
    """Dunnett vs control from the pooled MSE: exact adjusted P and simultaneous CI (rho_ij = lam_i lam_j)."""
    t = np.abs(d) / se
    p = 2 * stats.t.sf(t, df)
    lam = np.sqrt(ns / (ns + n_ctl))
    crit = _maxt_crit(lam, df, level)
    padj = np.array([_maxt_sf(c, lam, df) for c in t])
    return p, np.clip(padj, 0, 1), d - crit * se, d + crit * se


def _smm_cdf(t, m, df):
    """P(max of m independent |t| ratios sharing one chi <= t): studentized maximum modulus."""
    return 1 - _maxt_sf(t, np.zeros(m), df)


def _dunnett_t3(d, se, df, m, level):
    """Dunnett T3: Welch-Satterthwaite df rounded to the nearest integer (as Prism) for the SMM step."""
    d, se, df = np.asarray(d), np.asarray(se), np.asarray(df)
    p = 2 * stats.t.sf(np.abs(d) / se, df)
    dfr = np.maximum(np.floor(df + 0.5), 1)
    lam0 = np.zeros(m)
    padj = np.array([_maxt_sf(abs(x) / s, lam0, v) for x, s, v in zip(d, se, dfr)])
    crit = np.array([_maxt_crit(lam0, v, level) for v in dfr])
    return p, np.clip(padj, 0, 1), d - crit * se, d + crit * se

# --------------------------------------------------------------------------- one-way ANOVA

def _welch_anova(means, vars_, ns):
    k = len(ns)
    w = ns / vars_
    W = w.sum()
    mw = (w * means).sum() / W
    s = ((1 - w / W) ** 2 / (ns - 1)).sum()
    f = ((w * (means - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k * k - 1) * s)
    df2 = (k * k - 1) / (3 * s)
    return f, k - 1, df2, float(stats.f.sf(f, k - 1, df2))


def _bf_anova(means, vars_, ns):
    k, N = len(ns), ns.sum()
    gm = (ns * means).sum() / N
    c = (1 - ns / N) * vars_
    f = (ns * (means - gm) ** 2).sum() / c.sum()
    df2 = 1 / ((c / c.sum()) ** 2 / (ns - 1)).sum()
    return f, k - 1, df2, float(stats.f.sf(f, k - 1, df2))


def _pairs(k, family, ctl):
    return [(ctl, j) for j in range(k) if j != ctl] if family == "vs_control" else list(combinations(range(k), 2))


def _control_index(ids, p):
    ctl = p.get("control") or ids[0]
    if ctl not in ids:
        raise _Bad("the control data set is not among the selected data sets")
    return ids.index(ctl)


@_kind("one_way_anova")
def one_way_anova(table, ids, p):
    g = _groups(table, ids, summary_ok=True)
    _need(g, 2, 1)
    gid = [i for i, _, _ in g]
    titles = {i: t for i, t, _ in g}
    xs = [x for _, _, x in g]
    ns = np.array([len(x) for x in xs], float)
    k, N = len(xs), ns.sum()
    means = np.array([x.mean() for x in xs])
    vars_ = np.array([x.var(ddof=1) if len(x) > 1 else np.nan for x in xs])
    gm = np.concatenate(xs).mean()
    ssb = float((ns * (means - gm) ** 2).sum())
    sse = float(sum(((x - x.mean()) ** 2).sum() for x in xs))
    dfb, dfe = k - 1, int(N - k)
    if dfe < 1:
        raise _Bad("no residual degrees of freedom: every data set has a single value")
    if sse <= 0:
        raise _Bad("zero variance within every data set; F is undefined")
    mse, msb = sse / dfe, ssb / dfb
    f = msb / mse
    pval = float(stats.f.sf(f, dfb, dfe))
    sst = ssb + sse
    anova = {"ss_between": ssb, "df_between": dfb, "ms_between": msb, "f": f, "p": pval,
             "ss_within": sse, "df_within": dfe, "ms_within": mse, "ss_total": sst, "df_total": int(N - 1), "r2": ssb / sst}
    msgs, res = [], {"n": {i: len(x) for i, _, x in g}, "means": dict(zip(gid, means)), "anova": anova}
    level, method, summ = p["ci_level"], p["multiple_comparisons"], table.y_format != "replicates"
    disp = [{"title": "Ordinary one-way ANOVA", "columns": ["Source", "SS", "DF", "MS", "F (DFn, DFd)", "P value"],
             "rows": [["Treatment (between columns)", ssb, dfb, msb, f"F ({dfb}, {dfe}) = {f:.4g}", p_format(pval)],
                      ["Residual (within columns)", sse, dfe, mse, None, None], ["Total", sst, int(N - 1), None, None, None]]},
            {"title": "R squared", "columns": ["R squared"], "rows": [[anova["r2"]]]}]
    # equal-SD tests
    var_ok = bool(np.all(ns >= 2) and np.all(vars_ > 0))
    vt = {"bartlett": None, "brown_forsythe": None}
    if var_ok:
        bt = stats.bartlett(*xs)
        vt["bartlett"] = {"statistic": float(bt.statistic), "df": dfb, "p": float(bt.pvalue)}
        if not summ:
            lv = stats.levene(*xs, center="median")
            vt["brown_forsythe"] = {"f": float(lv.statistic), "df_num": dfb, "df_den": dfe, "p": float(lv.pvalue)}
    else:
        msgs.append("equal-SD tests need at least 2 values and non-zero SD in every data set")
    res["variance_tests"] = vt
    rows = [[n_, v and v.get("f", v.get("statistic")), v and p_format(v["p"])] for n_, v in
            (("Brown-Forsythe test F", vt["brown_forsythe"]), ("Bartlett's test statistic", vt["bartlett"]))]
    disp.append({"title": "Tests for equal SDs", "columns": ["Test", "Statistic", "P value"], "rows": rows})
    # Welch / Brown-Forsythe ANOVA
    wb = p["welch_bf"]
    if wb != "ordinary":
        if not var_ok:
            raise _Bad(f"{wb} ANOVA needs at least 2 values and non-zero SD in every data set")
        fw, d1, d2, pw = (_welch_anova if wb == "welch" else _bf_anova)(means, vars_, ns)
        res["welch" if wb == "welch" else "brown_forsythe_anova"] = {"f": fw, "df_num": d1, "df_den": d2, "p": pw}
        disp.insert(1, {"title": "Welch's ANOVA" if wb == "welch" else "Brown-Forsythe ANOVA",
                        "columns": ["Statistic", "DFn", "DFd", "P value"], "rows": [[fw, d1, d2, p_format(pw)]]})
    # multiple comparisons
    if method != "none":
        fam = "all_pairs" if method == "tukey" else "vs_control" if method == "dunnett" else p["mc_family"]
        ctl = _control_index(gid, p) if fam == "vs_control" else None
        pairs = _pairs(k, fam, ctl)
        d = np.array([means[i] - means[j] for i, j in pairs])
        if method == "dunnett_t3":
            if not var_ok:
                raise _Bad("Dunnett T3 needs at least 2 values and non-zero SD in every data set")
            v = np.array([vars_[i] / ns[i] + vars_[j] / ns[j] for i, j in pairs])
            se = np.sqrt(v)
            df = np.array([v[q] ** 2 / ((vars_[i] / ns[i]) ** 2 / (ns[i] - 1) + (vars_[j] / ns[j]) ** 2 / (ns[j] - 1))
                           for q, (i, j) in enumerate(pairs)])
            pu, pa, lo, hi = _dunnett_t3(d, se, df, len(pairs), level)
        else:
            se = np.sqrt(mse * np.array([1 / ns[i] + 1 / ns[j] for i, j in pairs]))
            if method == "dunnett":
                pu, pa, lo, hi = _dunnett_pooled(d, se, ns[ctl], ns[[j for _, j in pairs]], dfe, level)
            else:
                pu, pa, lo, hi = _mc_adjust(d, se, dfe, "none" if method == "lsd" else method, level, k)
        comps = [_comp(f"{gid[i]}__{gid[j]}", gid[i], gid[j], d[q], None if lo is None else lo[q],
                       None if hi is None else hi[q], pu[q], pa[q], se=se[q]) for q, (i, j) in enumerate(pairs)]
        res["comparisons"] = comps
        res["mc_method"] = method
        res["diff_label"] = "A - B"
        disp.append(_mc_table(f"Multiple comparisons ({method})", comps, titles, "Mean diff."))
    res["display"] = disp
    res["messages"] = msgs
    return res


# --------------------------------------------------------------------------- nonparametric k groups

def _dunn(mean_ranks, ns, N, ties, pairs):
    """Mean rank differences and their tie-corrected Dunn standard errors."""
    var = N * (N + 1) / 12 - ties / (12 * (N - 1))
    d = np.array([mean_ranks[i] - mean_ranks[j] for i, j in pairs])
    se = np.sqrt(np.array([var * (1 / ns[i] + 1 / ns[j]) for i, j in pairs]))
    return d, se


@_kind("kruskal_wallis")
def kruskal_wallis(table, ids, p):
    g = _groups(table, ids)
    _need(g, 2, 1)
    gid = [i for i, _, _ in g]
    titles = {i: t for i, t, _ in g}
    xs = [x for _, _, x in g]
    ns = np.array([len(x) for x in xs], float)
    k, N = len(xs), int(ns.sum())
    allv = np.concatenate(xs)
    ties = _tie_term(allv)
    if ties >= N ** 3 - N:
        raise _Bad("all values are identical; H is undefined")
    r = stats.rankdata(allv)
    cuts = np.cumsum(ns).astype(int)[:-1]
    rsum = np.array([c.sum() for c in np.split(r, cuts)])
    h = (12 / (N * (N + 1)) * (rsum ** 2 / ns).sum() - 3 * (N + 1)) / (1 - ties / (N ** 3 - N))
    pval = float(stats.chi2.sf(h, k - 1))
    mr = rsum / ns
    res = {"n": dict(zip(gid, ns.astype(int))), "h": h, "df": k - 1, "p": pval, "method": "chi-square approximation",
           "rank_sums": dict(zip(gid, rsum)), "mean_ranks": dict(zip(gid, mr)), "messages": []}
    disp = [{"title": "Kruskal-Wallis test", "columns": ["Parameter", "Value"],
             "rows": [["P value", p_format(pval)], ["P value summary", p_summary(pval)], ["Kruskal-Wallis statistic H", h],
                      ["DF", k - 1], ["Method", "chi-square approximation"]]},
            {"title": "Rank sums", "columns": ["Data set", "n", "Rank sum", "Mean rank"],
             "rows": [[titles[i], int(ns[q]), rsum[q], mr[q]] for q, i in enumerate(gid)]}]
    if p["multiple_comparisons"] == "dunn":
        ctl = _control_index(gid, p) if p["mc_family"] == "vs_control" else None
        pairs = _pairs(k, p["mc_family"], ctl)
        d, se = _dunn(mr, ns, N, ties, pairs)
        pu = 2 * stats.norm.sf(np.abs(d) / se)
        pa = np.minimum(1, len(pairs) * pu)
        comps = [_comp(f"{gid[i]}__{gid[j]}", gid[i], gid[j], d[q], None, None, pu[q], pa[q], z=d[q] / se[q])
                 for q, (i, j) in enumerate(pairs)]
        res["comparisons"] = comps
        res["diff_label"] = "Mean rank diff. (A - B)"
        disp.append(_mc_table("Dunn's multiple comparisons", comps, titles, "Mean rank diff."))
    res["display"] = disp
    return res


@_kind("friedman")
def friedman(table, ids, p):
    m, titles_l, n, dropped = _paired(table, ids, 3)
    gid, k, nr = list(ids), len(ids), len(m)
    titles = dict(zip(gid, titles_l))
    if nr < 2:
        raise _Bad(f"{nr} complete row(s); at least 2 needed")
    rk = np.apply_along_axis(stats.rankdata, 1, m)
    tie_rows = sum(_tie_term(r_) for r_ in m)
    if tie_rows >= nr * k * (k * k - 1):
        raise _Bad("all values are tied within every row; Friedman statistic is undefined")
    rsum = rk.sum(axis=0)
    q = (12 / (nr * k * (k + 1)) * (rsum ** 2).sum() - 3 * nr * (k + 1)) / (1 - tie_rows / (nr * k * (k * k - 1)))
    pval = float(stats.chi2.sf(q, k - 1))
    msgs = [f"{dropped} row(s) with a blank were dropped"] if dropped else []
    res = {"messages": msgs, "n": n, "n_rows": nr, "rows_dropped": dropped, "statistic": q, "df": k - 1, "p": pval,
           "rank_sums": dict(zip(gid, rsum)), "method": "chi-square approximation"}
    disp = [{"title": "Friedman test", "columns": ["Parameter", "Value"],
             "rows": [["P value", p_format(pval)], ["P value summary", p_summary(pval)], ["Friedman statistic", q], ["DF", k - 1],
                      ["Rows used", nr], ["Rows dropped (blank cell)", dropped], ["Method", "chi-square approximation"]]},
            {"title": "Rank sums", "columns": ["Data set", "Rank sum", "Mean rank"],
             "rows": [[titles[i], rsum[j], rsum[j] / nr] for j, i in enumerate(gid)]}]
    if p["multiple_comparisons"] == "dunn":
        ctl = _control_index(gid, p) if p["mc_family"] == "vs_control" else None
        pairs = _pairs(k, p["mc_family"], ctl)
        mr = rsum / nr
        d = np.array([mr[i] - mr[j] for i, j in pairs])
        se = math.sqrt(k * (k + 1) / (6 * nr))
        pu = 2 * stats.norm.sf(np.abs(d) / se)
        pa = np.minimum(1, len(pairs) * pu)
        comps = [_comp(f"{gid[i]}__{gid[j]}", gid[i], gid[j], d[q_], None, None, pu[q_], pa[q_], z=d[q_] / se)
                 for q_, (i, j) in enumerate(pairs)]
        res["comparisons"] = comps
        res["diff_label"] = "Mean rank diff. (A - B)"
        disp.append(_mc_table("Dunn's multiple comparisons", comps, titles, "Mean rank diff."))
    res["display"] = disp
    return res


# --------------------------------------------------------------------------- two-way ANOVA

def _cells(table, ids):
    """Grouped table -> (cells[r][c] arrays, row indices used). Summary layouts become synthetic cells."""
    if table.y_format == "replicates":
        masked = [np.where(table.dataset(i).excluded, np.nan, table.dataset(i).values) for i in ids]
        cells = [[m[r][~np.isnan(m[r])] for m in masked] for r in range(table.n_rows)]
    else:
        sm = [_summary_arrays(table, i) for i in ids]
        cells = [[np.empty(0) if np.isnan(s["mean"][r]) else _synth(s["mean"][r], 0.0 if s["n"][r] < 2 else s["sd"][r], int(round(s["n"][r])))
                  for s in sm] for r in range(table.n_rows)]
    used = [r for r, row in enumerate(cells) if any(len(c) for c in row)]
    return [cells[r] for r in used], used


def _type3_table(long):
    # separate scope: patsy's C() must not be shadowed by a local variable named C
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    return sm.stats.anova_lm(ols("y ~ C(r, Sum) * C(c, Sum)", long).fit(), typ=3)


@_kind("two_way_anova")
def two_way_anova(table, ids, p):
    import pandas as pd

    cells, used = _cells(table, ids)
    R, C = len(cells), len(ids)
    if R < 2 or C < 2:
        raise _Bad("two-way ANOVA needs at least 2 rows and 2 data sets (columns)")
    rt = [table.row_titles[r] if r < len(table.row_titles) and table.row_titles[r] else f"Row {r + 1}" for r in used]
    ct = {i: table.dataset(i).title for i in ids}
    for r in range(R):
        for c in range(C):
            if len(cells[r][c]) == 0:
                raise _Bad(f"empty cell: {rt[r]} / {ct[ids[c]]}; every row x column cell needs a value")
    nn = np.array([[len(cells[r][c]) for c in range(C)] for r in range(R)], float)
    mm = np.array([[cells[r][c].mean() for c in range(C)] for r in range(R)])
    N = int(nn.sum())
    dfe = N - R * C
    if dfe < 1:
        raise _Bad("no replicates: the residual has zero degrees of freedom")
    long = pd.DataFrame([(r, c, v) for r in range(R) for c in range(C) for v in cells[r][c]], columns=["r", "c", "y"])
    sse = float(sum(((x - x.mean()) ** 2).sum() for row in cells for x in row))
    if sse <= 0:
        raise _Bad("zero variance within every cell; F is undefined")
    tab = _type3_table(long)
    names = [("Interaction", "C(r, Sum):C(c, Sum)"), ("Row Factor", "C(r, Sum)"), ("Column Factor", "C(c, Sum)")]
    sst = float(((long["y"] - long["y"].mean()) ** 2).sum())
    mse = sse / dfe
    terms, rows = {}, []
    for lab, key in names:
        ss, df_ = float(tab.loc[key, "sum_sq"]), int(tab.loc[key, "df"])
        f_, pv = float(tab.loc[key, "F"]), float(tab.loc[key, "PR(>F)"])
        terms[lab] = {"ss": ss, "df": df_, "ms": ss / df_, "f": f_, "p": pv, "pct_total": 100 * ss / sst}
        rows.append([lab, 100 * ss / sst, ss, df_, ss / df_, f"F ({df_}, {dfe}) = {f_:.4g}", p_format(pv), p_summary(pv)])
    terms["Residual"] = {"ss": sse, "df": dfe, "ms": mse, "pct_total": 100 * sse / sst}
    rows.append(["Residual", 100 * sse / sst, sse, dfe, mse, None, None, None])
    rows.append(["Total", 100.0, sst, N - 1, None, None, None, None])
    msgs = []
    if np.ptp(nn) > 0:
        msgs.append("unbalanced design: type III sums of squares; components need not add up to the total")
    res = {"messages": msgs, "n": {i: int(nn[:, c].sum()) for c, i in enumerate(ids)},
           "n_cells": {rt[r]: {i: int(nn[r, c]) for c, i in enumerate(ids)} for r in range(R)},
           "terms": terms, "ss_total": sst, "df_total": N - 1, "ss_type": "III",
           "cell_means": {rt[r]: {i: float(mm[r, c]) for c, i in enumerate(ids)} for r in range(R)},
           "display": [{"title": "Two-way ANOVA (ordinary)",
                        "columns": ["Source of variation", "% of total variation", "SS", "DF", "MS", "F (DFn, DFd)", "P value", "Summary"],
                        "rows": rows}]}
    mc = p["multiple_comparisons"]
    if mc == "none":
        return res
    level, corr = p["ci_level"], p["correction"]
    fam = p["family"]
    if mc in ("within_rows", "within_columns"):
        # within_rows compares the C data sets inside each row; within_columns compares the R rows inside each data set
        byrow = mc == "within_rows"
        outer, inner = (R, C) if byrow else (C, R)    # one separate family per outer level; inner levels are compared
        if inner == 2 and fam == "per_row":
            fam = "all"  # a separate family would hold a single comparison
        if corr == "auto":
            corr = "bonferroni" if inner == 2 else "tukey"
        if corr == "tukey" and fam == "all":
            raise _Bad("Tukey applies to separate families (one per row or per column); "
                       "use family 'per_row' or a Sidak / Bonferroni correction")
    elif corr == "auto":
        corr = "tukey"
    cp = list(combinations(range(C), 2))
    comps = []
    if mc in ("within_rows", "within_columns"):
        ip = list(combinations(range(inner), 2))
        flat = [(o, i, j) for o in range(outer) for i, j in ip]
        cell = (lambda o, i: (o, i)) if byrow else (lambda o, i: (i, o))
        d = np.array([mm[cell(o, i)] - mm[cell(o, j)] for o, i, j in flat])
        se = np.sqrt(mse * np.array([1 / nn[cell(o, i)] + 1 / nn[cell(o, j)] for o, i, j in flat]))
        fams = [list(range(len(flat)))] if fam == "all" else [list(range(o * len(ip), (o + 1) * len(ip))) for o in range(outer)]
        out = np.empty((len(flat), 4))
        for idx in fams:
            out[idx] = np.column_stack(_mc_adjust(d[idx], se[idx], dfe, corr, level, inner))
        if byrow:
            comps = [_comp(f"r{used[o]}:{ids[i]}__{ids[j]}", ids[i], ids[j], d[q], out[q, 2], out[q, 3], out[q, 0], out[q, 1],
                           row=used[o], se=se[q]) for q, (o, i, j) in enumerate(flat)]
            sep = "one family per row"
        else:  # a / b are row ids ("r<k>"); "col" names the data set the rows are compared in
            comps = [_comp(f"c{ids[o]}:r{used[i]}__r{used[j]}", f"r{used[i]}", f"r{used[j]}", d[q], out[q, 2], out[q, 3], out[q, 0],
                           out[q, 1], col=ids[o], se=se[q]) for q, (o, i, j) in enumerate(flat)]
            sep = "one family per column"
        title = (f"Multiple comparisons: {'columns within each row' if byrow else 'rows within each column'} "
                 f"({corr}, {sep if fam == 'per_row' else 'one family'})")
    elif mc == "main_column":
        mu = mm.mean(axis=0)
        d = np.array([mu[i] - mu[j] for i, j in cp])
        se = np.sqrt(mse * np.array([(1 / nn[:, i] + 1 / nn[:, j]).sum() for i, j in cp]) / R ** 2)
        pu, pa, lo, hi = _mc_adjust(d, se, dfe, corr, level, C)
        comps = [_comp(f"{ids[i]}__{ids[j]}", ids[i], ids[j], d[q], lo[q], hi[q], pu[q], pa[q], se=se[q]) for q, (i, j) in enumerate(cp)]
        title = f"Multiple comparisons: column means ({corr})"
    else:
        raise _Bad(f"unknown multiple comparisons choice {mc!r}")
    res["comparisons"] = comps
    res["diff_label"] = "A - B"
    # row index -> title for the table
    rt_by_row = {used[r]: rt[r] for r in range(R)}
    if mc == "within_columns":
        rows_ = [[ct[c["col"]], f"{rt_by_row[int(c['a'][1:])]} vs. {rt_by_row[int(c['b'][1:])]}", c["diff"], c["ci_low"], c["ci_high"],
                  c["summary"], p_format(c["p_adj"])] for c in comps]
        res["display"].append({"title": title, "columns": ["Column", "Comparison", "Mean diff.", "CI low", "CI high", "Summary", "Adjusted P"],
                               "rows": rows_})
    else:
        res["display"].append(_mc_table(title, comps, ct, "Mean diff.", with_row=(mc == "within_rows"), row_titles=rt_by_row))
    return res


# --------------------------------------------------------------------------- registry

_CI = Param("ci_level", "Confidence level", "float", 0.95)
_MCFAM = Param("mc_family", "Comparisons", "choice", "all_pairs", (("all_pairs", "Every data set with every other"), ("vs_control", "Each data set with the control")))
_CONTROL = Param("control", "Control data set", "dataset", None)

KINDS = {
    "descriptive": AnalysisKind("descriptive", "Descriptive statistics", ("Column",), descriptive, (_CI,)),
    "t_test": AnalysisKind("t_test", "t test", ("Column",), t_test, (
        Param("paired", "Paired (matched rows)", "bool", False),
        Param("welch", "Welch's correction (unpaired)", "bool", False), _CI)),
    "mann_whitney": AnalysisKind("mann_whitney", "Mann-Whitney test", ("Column",), mann_whitney, (_CI,)),
    "wilcoxon_paired": AnalysisKind("wilcoxon_paired", "Wilcoxon matched-pairs test", ("Column",), wilcoxon_paired, (
        Param("zero_method", "Pairs with zero difference", "choice", "wilcoxon",
              (("wilcoxon", "Ignore (Wilcoxon)"), ("pratt", "Rank with the rest (Pratt)"))), _CI)),
    "one_way_anova": AnalysisKind("one_way_anova", "One-way ANOVA", ("Column",), one_way_anova, (
        Param("welch_bf", "Omnibus test", "choice", "ordinary",
              (("ordinary", "Ordinary"), ("welch", "Welch ANOVA"), ("brown_forsythe", "Brown-Forsythe ANOVA"))),
        Param("multiple_comparisons", "Multiple comparisons", "choice", "none",
              (("none", "None"), ("lsd", "No correction (Fisher's LSD)"), ("tukey", "Tukey"), ("dunnett", "Dunnett"), ("sidak", "Sidak"),
               ("bonferroni", "Bonferroni"), ("holm_sidak", "Holm-Sidak"), ("dunnett_t3", "Dunnett T3 (unequal SDs)"))),
        _MCFAM, _CONTROL, _CI)),
    "kruskal_wallis": AnalysisKind("kruskal_wallis", "Kruskal-Wallis test", ("Column",), kruskal_wallis, (
        Param("multiple_comparisons", "Multiple comparisons", "choice", "none", (("none", "None"), ("dunn", "Dunn"))),
        _MCFAM, _CONTROL)),
    "friedman": AnalysisKind("friedman", "Friedman test", ("Column",), friedman, (
        Param("multiple_comparisons", "Multiple comparisons", "choice", "none", (("none", "None"), ("dunn", "Dunn"))),
        _MCFAM, _CONTROL)),
    "two_way_anova": AnalysisKind("two_way_anova", "Two-way ANOVA", ("Grouped",), two_way_anova, (
        Param("multiple_comparisons", "Multiple comparisons", "choice", "none",
              (("none", "None"), ("within_rows", "Data sets within each row"), ("within_columns", "Rows within each data set"),
               ("main_column", "Column means (main effect)"))),
        Param("correction", "Correction", "choice", "auto",
              (("auto", "Prism default"), ("sidak", "Sidak"), ("tukey", "Tukey"), ("bonferroni", "Bonferroni"))),
        Param("family", "Family (within rows / columns)", "choice", "per_row",
              (("per_row", "Separate families (one per row, or per data set)"), ("all", "One family for all comparisons"))),
        _CI)),
}
