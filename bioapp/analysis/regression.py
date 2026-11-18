"""Regression analyses following Prism: simple linear (+ comparison of lines), Deming, simple logistic,
and multiple linear / logistic / Poisson regression.

Prism conventions used here:
  * linear regression: t-based CIs with df = n-2 (n-1 when the line is forced through a point); F test of
    slope != 0; Sy.x = sqrt(SSres/df); X-intercept CI by Fieller's theorem; line comparison tests slopes first
    (extra sum-of-squares F) and, only when the slopes do not differ, the elevations (intercepts);
  * logistic / Poisson regression: maximum likelihood (IRLS), Wald z tests, profile-likelihood CIs by default
    (checked against Prism for simple logistic regression: coefficients, SE, CI, odds ratios, X at 50 %, Tjur and
    Cox-Snell R2, deviance, likelihood-ratio test, ROC area and P);
  * AICc counts K = parameters (+1 for the variance of least squares), as in fit.py;
  * categorical predictors are dummy coded against a reference level (default: the level that appears first in the
    column, as Prism's category id 0);
  * coefficient t / Z statistics are reported unsigned; the log-likelihood rows are -LL; ROC SE is Bamber's variance;
    Hosmer-Lemeshow groups are cut at the k/g quantiles with position k (n + 1) / g.
Every function returns the result contract of DESIGN.md; invalid input gives status "failed" with a reason.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, special, stats

from . import AnalysisKind, Param
from .stats import _Bad, p_summary, _kind as _stats_kind

def _tolist(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _tolist(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_tolist(v) for v in o]
    return o


def _kind(name):
    """stats._kind plus ndarray -> list conversion (results must be JSON-serializable)."""
    wrap = _stats_kind(name)
    return lambda f: wrap(lambda table, ids, params: _tolist(f(table, ids, params)))


_LEVEL = Param("ci_level", "Confidence level (%)", "float", 95.0)


# --------------------------------------------------------------------------- helpers

def _tbl(title, columns, rows):
    return {"title": title, "columns": columns, "rows": rows}


def _level(params):
    lv = float(params.get("ci_level", 95.0))
    if not 50 <= lv < 100:
        raise _Bad("confidence level must be in [50, 100)")
    return lv / 100


def _xy(table, ds_id, fit_to):
    """x, y of one data set: every replicate, or the mean of the replicates of each row."""
    if fit_to == "replicates":
        if table.y_format != "replicates":
            raise _Bad("fitting every replicate needs raw values; choose 'means' for mean/SD/SEM tables")
        x, y, _ = table.points(ds_id)
    else:
        s = table.summary(ds_id)
        ok = ~np.isnan(s["mean"]) & ~np.isnan(table.x)
        x, y = table.x[ok], s["mean"][ok]
    return np.asarray(x, float), np.asarray(y, float)


def _fieller(a, b, va, vb, cab, t):
    """CI of -a/b (Fieller). None when the interval is unbounded."""
    qa = b * b - t * t * vb
    if qa <= 0:
        return None
    qb = a * b - t * t * cab
    disc = qb * qb - qa * (a * a - t * t * va)
    if disc < 0:
        return None
    r = math.sqrt(disc)
    return sorted([(-qb - r) / qa, (-qb + r) / qa])


# --------------------------------------------------------------------------- linear regression

def _line(x, y, level, fixed=None):
    """Least-squares line; ``fixed`` = (x0, y0) forces it through that point (origin: (0, 0))."""
    n = len(x)
    if fixed is not None:
        x0, y0 = fixed
        u, v = x - x0, y - y0
        suu = float(u @ u)
        if n < 2 or suu == 0:
            raise _Bad("need at least 2 points with X different from the fixed point")
        b = float(u @ v) / suu
        a = y0 - b * x0
        df = n - 1
        ss = float(np.sum((y - a - b * x) ** 2))
        ss0 = float(v @ v)
        xbar, sxx = x0, suu
        h = lambda g: (g - x0) ** 2 / suu  # noqa: E731
        cab_f = -x0
    else:
        if n < 3 or len(np.unique(x)) < 2:
            raise _Bad("need at least 3 points with at least 2 different X values")
        xbar, ybar = float(x.mean()), float(y.mean())
        sxx = float(np.sum((x - xbar) ** 2))
        b = float(np.sum((x - xbar) * (y - ybar))) / sxx
        a = ybar - b * xbar
        df = n - 2
        ss = float(np.sum((y - a - b * x) ** 2))
        ss0 = float(np.sum((y - ybar) ** 2))
        h = lambda g: 1 / n + (g - xbar) ** 2 / sxx  # noqa: E731
        cab_f = -xbar
    syx = math.sqrt(ss / df)
    seb = syx / math.sqrt(sxx)
    t = float(stats.t.ppf(0.5 + level / 2, df))
    x0f = fixed[0] if fixed is not None else None
    if fixed is None:
        sea = syx * math.sqrt(1 / n + xbar ** 2 / sxx)
        va, cab = sea ** 2, cab_f * seb ** 2
    elif x0f == 0:  # through the origin: the intercept is fixed, not estimated
        sea, va, cab = None, 0.0, 0.0
    else:
        sea = abs(x0f) * seb
        va, cab = sea ** 2, cab_f * seb ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (ss0 - ss) / (ss / df) if ss > 0 else math.inf
    p = float(stats.f.sf(f, 1, df)) if np.isfinite(f) else 0.0
    sst = float(np.sum((y - y.mean()) ** 2))
    if fixed is not None and x0f == 0 and fixed[1] == 0:
        xi = None  # the line passes through the origin
    elif b != 0:
        xi = {"value": -a / b, **dict(zip(("ci_low", "ci_high"), _fieller(a, b, va, seb ** 2, cab, t) or (None, None)))}
    else:
        xi = None
    inv = None
    if b != 0:
        lo, hi = b - t * seb, b + t * seb
        inv = {"value": 1 / b, "ci_low": 1 / hi if lo * hi > 0 else None, "ci_high": 1 / lo if lo * hi > 0 else None}
        if inv["ci_low"] is not None and inv["ci_low"] > inv["ci_high"]:
            inv["ci_low"], inv["ci_high"] = inv["ci_high"], inv["ci_low"]
    return {"n": n, "df": df, "b": b, "a": a, "seb": seb, "sea": sea, "t": t, "ss": ss, "syx": syx,
            "r2": 1 - ss / sst if sst > 0 else float("nan"), "f": f, "p": p, "h": h, "xi": xi, "inv": inv,
            "forced": fixed is not None, "sxx": sxx, "xbar": xbar}


def _runs_exact(npos, nneg, r):
    """P(runs <= r) for the Wald-Wolfowitz runs test (exact, no ties in signs)."""
    n = npos + nneg
    total = special.comb(n, npos)
    s = 0.0
    for k in range(2, r + 1):
        if k % 2 == 0:
            c = 2 * special.comb(npos - 1, k // 2 - 1) * special.comb(nneg - 1, k // 2 - 1)
        else:
            m = (k - 1) // 2
            c = special.comb(npos - 1, m) * special.comb(nneg - 1, m - 1) + special.comb(npos - 1, m - 1) * special.comb(nneg - 1, m)
        s += c
    return float(s / total)


def _runs_test(x, resid):
    """Runs test on residual signs ordered by X (Prism: too few runs = deviation from the line)."""
    order = np.argsort(x, kind="stable")
    sg = np.sign(resid[order])
    sg = sg[sg != 0]
    npos, nneg = int((sg > 0).sum()), int((sg < 0).sum())
    if npos == 0 or nneg == 0:
        return {"above": npos, "below": nneg, "runs": 1, "p": None, "note": "all residuals on one side"}
    runs = 1 + int(np.sum(sg[1:] != sg[:-1]))
    return {"above": npos, "below": nneg, "runs": runs, "p": _runs_exact(npos, nneg, runs)}


def _normality(resid):
    out = {}
    if len(resid) >= 8:
        k2, p = stats.normaltest(resid)
        out["dagostino_k2"] = {"stat": float(k2), "p": float(p)}
    if len(resid) >= 3:
        w, p = stats.shapiro(resid)
        out["shapiro_wilk"] = {"stat": float(w), "p": float(p)}
    return out


def _ss_line(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    return float(np.sum((yc - xc * (xc @ yc) / (xc @ xc)) ** 2))


def _compare_lines(fits, data, alpha):
    """Slopes (separate vs common) then, if they do not differ, elevations (common slope vs one line)."""
    K = len(fits)
    N = sum(f["n"] for f in fits)
    if N - 2 * K < 1:
        raise _Bad("too few points to compare the lines")
    ss_sep = sum(f["ss"] for f in fits)
    df_sep = N - 2 * K
    sxx = sum(f["sxx"] for f in fits)
    sxy = sum(float(np.sum((x - x.mean()) * (y - y.mean()))) for x, y in data)
    bc = sxy / sxx
    ss_c = sum(float(np.sum((y - y.mean() - bc * (x - x.mean())) ** 2)) for x, y in data)
    df_c = N - K - 1
    out = {"common_slope": bc, "ss_separate": ss_sep, "df_separate": df_sep, "ss_common_slope": ss_c, "df_common_slope": df_c}
    fs = ((ss_c - ss_sep) / (K - 1)) / (ss_sep / df_sep) if ss_sep > 0 else math.inf
    ps = float(stats.f.sf(fs, K - 1, df_sep))
    out["slopes"] = {"f": fs, "dfn": K - 1, "dfd": df_sep, "p": ps, "differ": ps < alpha}
    out["intercepts"] = None
    if ps >= alpha:
        xa, ya = np.concatenate([d[0] for d in data]), np.concatenate([d[1] for d in data])
        ss_all = _ss_line(xa, ya)
        fi = ((ss_all - ss_c) / (K - 1)) / (ss_c / df_c) if ss_c > 0 else math.inf
        pi = float(stats.f.sf(fi, K - 1, df_c))
        out["intercepts"] = {"f": fi, "dfn": K - 1, "dfd": df_c, "p": pi, "differ": pi < alpha, "ss_single_line": ss_all}
        # the common slope with per-set intercepts, and the single line, are the two fitted alternatives
        out["common_intercepts"] = [float(y.mean() - bc * x.mean()) for x, y in data]
    return out


def _band(fit, x):
    g = np.linspace(float(x.min()), float(x.max()), 100)
    y = fit["a"] + fit["b"] * g
    hw = fit["t"] * fit["syx"] * np.sqrt(fit["h"](g))
    hp = fit["t"] * fit["syx"] * np.sqrt(1 + fit["h"](g))
    return {"x": g, "y": y, "ci_low": y - hw, "ci_high": y + hw, "pi_low": y - hp, "pi_high": y + hp}


def _ci_row(name, est, se, t):
    return [name, est, se, None if se is None else est - t * se, None if se is None else est + t * se]


@_kind("linear_regression")
def linear_regression(table, ids, params):
    level = _level(params)
    through = params.get("through", "none")
    fixed = None if through == "none" else (0.0, 0.0) if through == "origin" else (float(params["x0"]), float(params["y0"]))
    fit_to = params.get("fit_to", "replicates")
    msgs, res, display, resid_out = [], {}, [], {}
    fits, data, titles = [], [], []
    for i in ids:
        title = table.dataset(i).title
        try:
            x, y = _xy(table, i, fit_to)
            f = _line(x, y, level, fixed)
        except _Bad as e:
            res[i] = {"status": "failed", "messages": [f"{title}: {e}"]}
            msgs.append(f"{title}: {e}")
            continue
        fits.append(f), data.append((x, y)), titles.append(title)
        r = y - f["a"] - f["b"] * x
        resid_out[i] = {"x": x, "r": r}
        t = f["t"]
        d = {"status": "ok",
             "params": {"slope": {"value": f["b"], "se": f["seb"], "ci_low": f["b"] - t * f["seb"], "ci_high": f["b"] + t * f["seb"]},
                        "intercept": {"value": f["a"], "se": f["sea"],
                                      "ci_low": None if f["sea"] is None else f["a"] - t * f["sea"],
                                      "ci_high": None if f["sea"] is None else f["a"] + t * f["sea"]}},
             "derived": {k: v for k, v in (("x_intercept", f["xi"]), ("inv_slope", f["inv"])) if v is not None},
             "stats": {"n": f["n"], "df": f["df"], "ss": f["ss"], "sy_x": f["syx"], "r2": f["r2"],
                       "f": f["f"], "f_dfn": 1, "f_dfd": f["df"], "p": f["p"]},
             "flags": [], "curve": _band(f, x)}
        if params.get("runs_test"):
            d["runs_test"] = _runs_test(x, r)
        if params.get("normality"):
            d["residual_normality"] = _normality(r)
        res[i] = d
        pct = f"{level * 100:g}% CI"
        rows = [_ci_row("Slope", f["b"], f["seb"], t), _ci_row("Y-intercept", f["a"], f["sea"], t)]
        if f["xi"]:
            rows.append(["X-intercept", f["xi"]["value"], None, f["xi"]["ci_low"], f["xi"]["ci_high"]])
        if f["inv"]:
            rows.append(["1/slope", f["inv"]["value"], None, f["inv"]["ci_low"], f["inv"]["ci_high"]])
        display.append(_tbl(f"{title}: best-fit values", ["Parameter", "Value", "Std. error", f"{pct} low", f"{pct} high"], rows))
        gof = [["R squared", f["r2"]], ["Sy.x", f["syx"]], ["F", f["f"]], ["DFn, DFd", f"1, {f['df']}"],
               ["P value (slope different from zero)", f["p"]], ["Summary", p_summary(f["p"])], ["n (points analyzed)", f["n"]]]
        if "runs_test" in d:
            rt = d["runs_test"]
            gof += [["Runs test: points above / below", f"{rt['above']} / {rt['below']}"], ["Number of runs", rt["runs"]],
                    ["P value (runs test)", rt["p"]]]
        for k, v in d.get("residual_normality", {}).items():
            gof.append([f"Residual normality, {k}: P", v["p"]])
        display.append(_tbl(f"{title}: goodness of fit", ["Statistic", "Value"], gof))
    if not fits:
        raise _Bad("; ".join(msgs) or "no data")
    out = {"datasets": res, "residuals": resid_out, "messages": msgs, "display": display}
    if len(fits) >= 2 and params.get("compare", True):
        if fixed is not None:
            msgs.append("slopes and intercepts are not compared for lines forced through a point")
        else:
            c = _compare_lines(fits, data, float(params.get("alpha", 0.05)))
            out["comparison"] = c
            rows = [["Slopes: F", c["slopes"]["f"]], ["DFn, DFd", f"{c['slopes']['dfn']}, {c['slopes']['dfd']}"],
                    ["P value", c["slopes"]["p"]], ["Slopes significantly different?", "Yes" if c["slopes"]["differ"] else "No"],
                    ["Common slope", c["common_slope"]]]
            if c["intercepts"]:
                ci = c["intercepts"]
                rows += [["Intercepts (elevations, common slope): F", ci["f"]], ["DFn, DFd", f"{ci['dfn']}, {ci['dfd']}"],
                         ["P value", ci["p"]], ["Intercepts significantly different?", "Yes" if ci["differ"] else "No"]]
            else:
                rows.append(["Intercepts", "not compared: the slopes differ"])
            display.append(_tbl("Comparison of lines", ["Test", "Result"], rows))
    return out


# --------------------------------------------------------------------------- Deming regression

def _deming(x, y, delta):
    sxx, syy = float(np.sum((x - x.mean()) ** 2)), float(np.sum((y - y.mean()) ** 2))
    sxy = float(np.sum((x - x.mean()) * (y - y.mean())))
    if sxy == 0:
        raise _Bad("slope is undefined: X and Y are uncorrelated")
    b = (syy - delta * sxx + math.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy ** 2)) / (2 * sxy)
    return b, float(y.mean() - b * x.mean())


@_kind("deming_regression")
def deming_regression(table, ids, params):
    level = _level(params)
    ratio = float(params.get("error_ratio", 1.0))
    if not ratio > 0:
        raise _Bad("the SD ratio of Y errors to X errors must be positive")
    delta = ratio ** 2
    res, display, msgs = {}, [], []
    for i in ids:
        title = table.dataset(i).title
        try:
            x, y = _xy(table, i, params.get("fit_to", "replicates"))
            n = len(x)
            if n < 4 or len(np.unique(x)) < 2:
                raise _Bad("need at least 4 points with at least 2 different X values")
            b, a = _deming(x, y, delta)
        except _Bad as e:
            res[i] = {"status": "failed", "messages": [f"{title}: {e}"]}
            msgs.append(f"{title}: {e}")
            continue
        jk = np.array([_deming(np.delete(x, k), np.delete(y, k), delta) for k in range(n)])
        se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean(0)) ** 2, axis=0))  # jackknife SE
        t = float(stats.t.ppf(0.5 + level / 2, n - 2))
        g = np.linspace(x.min(), x.max(), 100)
        res[i] = {"status": "ok",
                  "params": {"slope": {"value": b, "se": float(se[0]), "ci_low": b - t * se[0], "ci_high": b + t * se[0]},
                             "intercept": {"value": a, "se": float(se[1]), "ci_low": a - t * se[1], "ci_high": a + t * se[1]}},
                  "derived": {}, "stats": {"n": n, "df": n - 2, "error_ratio": ratio}, "flags": [],
                  "curve": {"x": g, "y": a + b * g}}
        pct = f"{level * 100:g}% CI"
        display.append(_tbl(f"{title}: Deming regression (SD Y-error / SD X-error = {ratio:g})",
                            ["Parameter", "Value", "Std. error (jackknife)", f"{pct} low", f"{pct} high"],
                            [["Slope", b, se[0], b - t * se[0], b + t * se[0]], ["Y-intercept", a, se[1], a - t * se[1], a + t * se[1]]]))
    if not any(r["status"] == "ok" for r in res.values()):
        raise _Bad("; ".join(msgs))
    return {"datasets": res, "messages": msgs, "display": display}


# --------------------------------------------------------------------------- generalized linear models (logit, Poisson)

def _glm(X, y, fam, offset=None, b0=None):
    """Maximum likelihood by Newton / IRLS with step halving. Returns beta, loglik, cov, mu, eta."""
    n, k = X.shape
    off = np.zeros(n) if offset is None else offset

    def ll_mu(eta):
        if fam == "logit":
            return float(np.sum(y * eta - np.logaddexp(0, eta))), special.expit(eta)
        mu = np.exp(np.clip(eta, -700, 700))
        return float(np.sum(y * eta - mu - special.gammaln(y + 1))), mu

    b = np.zeros(k) if b0 is None else np.array(b0, float)
    ll, mu = ll_mu(X @ b + off)
    for _ in range(300):
        w = mu * (1 - mu) if fam == "logit" else mu
        try:
            step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-12 * np.eye(k), X.T @ (y - mu))
        except np.linalg.LinAlgError:
            raise _Bad("the model matrix is singular: predictors are collinear or the data are completely separated") from None
        s = 1.0
        while True:
            ll_new, mu_new = ll_mu(X @ (b + s * step) + off)
            if ll_new >= ll - 1e-12 or s < 1e-10:
                break
            s /= 2
        b, done = b + s * step, abs(ll_new - ll) < 1e-12 and np.max(np.abs(s * step)) < 1e-9
        ll, mu = ll_new, mu_new
        if done:
            break
    if np.max(np.abs(b)) > 25 and fam == "logit" and np.any(mu * (1 - mu) < 1e-9):
        raise _Bad("complete or quasi-complete separation: the maximum likelihood estimates do not exist")
    w = mu * (1 - mu) if fam == "logit" else mu
    try:
        cov = np.linalg.inv(X.T @ (X * w[:, None]))
    except np.linalg.LinAlgError:
        raise _Bad("the model matrix is singular: predictors are collinear") from None
    return {"b": b, "ll": ll, "cov": cov, "mu": mu, "eta": X @ b + off}


def _profile_ci(X, y, fam, fit, j, level, theta=None):
    """Profile-likelihood CI of coefficient j: where 2(ll_hat - ll_profile) reaches chi2(level, 1)."""
    crit = float(stats.chi2.ppf(level, 1))
    oth = [i for i in range(X.shape[1]) if i != j]
    se = math.sqrt(fit["cov"][j, j])
    b0 = fit["b"]

    def g(v, warm=[None]):
        if oth:
            f = _glm(X[:, oth], y, fam, offset=X[:, j] * v, b0=warm[0] if warm[0] is not None else b0[oth])
            warm[0] = f["b"]
            ll = f["ll"]
        else:
            eta = X[:, j] * v
            ll = float(np.sum(y * eta - np.logaddexp(0, eta))) if fam == "logit" else float(np.sum(y * eta - np.exp(eta) - special.gammaln(y + 1)))
        return 2 * (fit["ll"] - ll) - crit

    out = []
    for sg in (-1, 1):
        step, lo = se, b0[j]
        hi = None
        try:
            for _ in range(30):
                if g(lo + sg * step) > 0:
                    hi = lo + sg * step
                    break
                step *= 2
            out.append(None if hi is None else optimize.brentq(g, *sorted((lo, hi)), xtol=1e-10, rtol=1e-12))
        except (_Bad, ValueError, FloatingPointError):
            out.append(None)
    return out


def _roc(y, p, level):
    """ROC curve and area. SE and CI as Prism: Bamber / Hanley-McNeil variance with empirical Q1, Q2 (ties broken at random,
    which gives the e^2/3 terms; reproduces Prism's SE to 1e-15 with and without ties), CI = area +/- normal z * SE;
    P tests area = 0.5 with the null SE of the Mann-Whitney U."""
    pos, neg = p[y == 1], p[y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        raise _Bad("ROC analysis needs both outcomes (0 and 1)")
    allr = stats.rankdata(np.concatenate([pos, neg]))
    auc = float(((allr[:n1] - stats.rankdata(pos)) / n0).mean())
    sp, sn = np.sort(pos), np.sort(neg)
    lo, hi = np.searchsorted(sp, neg, "left"), np.searchsorted(sp, neg, "right")  # per negative: positives below / not above
    g, e = (n1 - hi) / n1, (hi - lo) / n1  # share of positives above / tied with each negative
    lo, hi = np.searchsorted(sn, pos, "left"), np.searchsorted(sn, pos, "right")
    b, e2 = lo / n0, (hi - lo) / n0  # per positive: share of negatives below / tied
    q1, q2 = float(np.mean(g * g + g * e + e * e / 3)), float(np.mean(b * b + b * e2 + e2 * e2 / 3))
    se = math.sqrt(max((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0), 0.0))
    z = float(stats.norm.ppf(0.5 + level / 2))
    se0 = math.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
    thr = np.unique(p)[::-1]
    tpr = np.concatenate([[0.0], [(pos >= t).mean() for t in thr]])
    fpr = np.concatenate([[0.0], [(neg >= t).mean() for t in thr]])
    return {"fpr": fpr, "tpr": tpr, "thresholds": np.concatenate([[math.inf], thr]), "auc": auc, "se": se,
            "ci_low": max(0.0, auc - z * se), "ci_high": min(1.0, auc + z * se),
            "p": float(2 * stats.norm.sf(abs(auc - 0.5) / se0))}


def _classify(y, p, cutoff):
    pred = p >= cutoff
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    tn, fn = int(np.sum(~pred & (y == 0))), int(np.sum(~pred & (y == 1)))
    d = lambda a, b: a / b if b else None  # noqa: E731
    return {"cutoff": cutoff, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "sensitivity": d(tp, tp + fn),
            "specificity": d(tn, tn + fp), "accuracy": (tp + tn) / len(y), "predicted": pred.astype(int)}


def _hosmer_lemeshow(y, p, groups=10):
    """Risk groups as Prism: cut points at the k/g quantiles with position k (n + 1) / g (linear interpolation between
    order statistics, i.e. 3, 4, ..., 4, 3 members for n = 38, g = 10); an observation joins the next group only when its
    probability exceeds the cut point, so tied probabilities always share a group."""
    n = len(y)
    g = min(groups, n // 2)
    if g < 3:
        return None
    cuts = np.interp(np.arange(1, g) * (n + 1) / g, np.arange(1, n + 1), np.sort(p))
    grp = np.searchsorted(cuts, p, side="left")
    chi, used = 0.0, 0
    for k in range(g):
        idx = np.flatnonzero(grp == k)
        if not len(idx):
            continue
        used += 1
        e = float(p[idx].sum())
        pm = e / len(idx)
        if 0 < pm < 1:
            chi += (float(y[idx].sum()) - e) ** 2 / (len(idx) * pm * (1 - pm))
    if used < 3:
        return None
    return {"chi2": chi, "df": used - 2, "p": float(stats.chi2.sf(chi, used - 2)), "groups": used,
            "sizes": [int(np.sum(grp == k)) for k in range(g) if np.any(grp == k)]}


def _aicc(neg2ll, k, n):
    return neg2ll + 2 * k + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else None


def _glm_report(X, y, names, fam, level, ci_method):
    """Fit a logistic / Poisson model and gather everything Prism reports."""
    n, k = X.shape
    fit = _glm(X, y, fam)
    null = _glm(np.ones((n, 1)), y, fam)
    se = np.sqrt(np.diag(fit["cov"]))
    z = fit["b"] / se
    zc = float(stats.norm.ppf(0.5 + level / 2))
    coefs = []
    for j in range(k):
        if ci_method == "profile":
            lo, hi = _profile_ci(X, y, fam, fit, j, level)
        else:
            lo, hi = fit["b"][j] - zc * se[j], fit["b"][j] + zc * se[j]
        e = float(fit["b"][j])
        coefs.append({"term": names[j], "estimate": e, "se": float(se[j]), "ci_low": lo, "ci_high": hi,
                      "stat": abs(float(z[j])), "p": float(2 * stats.norm.sf(abs(z[j]))),  # Prism reports |Z|; the sign is the estimate's
                      "ratio": math.exp(e) if abs(e) < 700 else None,
                      "ratio_ci_low": None if lo is None else math.exp(lo), "ratio_ci_high": None if hi is None else math.exp(hi)})
    g2 = 2 * (fit["ll"] - null["ll"])
    # Prism's "log likelihood" rows are -LL (positive): "ll" / "ll_null" hold -LL (displayed as "-Log likelihood")
    gof = {"deviance": -2 * fit["ll"], "ll": -fit["ll"], "ll_null": -null["ll"], "aicc": _aicc(-2 * fit["ll"], k, n),
           "lr_chi2": g2, "lr_df": k - 1, "lr_p": float(stats.chi2.sf(g2, k - 1)) if k > 1 else None,
           "cox_snell_r2": 1 - math.exp(-g2 / n)}
    if fam == "logit":
        mu = fit["mu"]
        gof["tjur_r2"] = float(mu[y == 1].mean() - mu[y == 0].mean())
        gof["nagelkerke_r2"] = gof["cox_snell_r2"] / (1 - math.exp(2 * null["ll"] / n))
    else:
        mu = fit["mu"]
        dev = 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))
        pearson = float(np.sum((y - mu) ** 2 / mu))
        dfr = n - k
        gof.update({"model_deviance": float(dev), "pearson_chi2": pearson, "df_residual": dfr,
                    "deviance_p": float(stats.chi2.sf(dev, dfr)) if dfr > 0 else None,
                    "pearson_p": float(stats.chi2.sf(pearson, dfr)) if dfr > 0 else None,
                    "dispersion": pearson / dfr if dfr > 0 else None, "mcfadden_r2": 1 - fit["ll"] / null["ll"]})
    return fit, coefs, gof


def _vif(X):
    """VIF of each non-constant column (1 / (1 - R2) from the other columns)."""
    if X.shape[1] < 2:
        return [1.0] * X.shape[1]
    try:
        return [float(v) for v in np.diag(np.linalg.inv(np.corrcoef(X.T)))]
    except np.linalg.LinAlgError:
        return [math.inf] * X.shape[1]


# --------------------------------------------------------------------------- simple logistic regression

@_kind("simple_logistic_regression")
def simple_logistic_regression(table, ids, params):
    level = _level(params)
    method = params.get("ci_method", "profile")
    cutoff = float(params.get("cutoff", 0.5))
    res, display, msgs = {}, [], []
    for i in ids:
        title = table.dataset(i).title
        try:
            x, y = _xy(table, i, "replicates")
            if not np.all(np.isin(y, (0, 1))):
                raise _Bad("Y must contain only 0 and 1")
            if y.min() == y.max():
                raise _Bad("Y has only one outcome; both 0 and 1 are needed")
            if len(np.unique(x)) < 2:
                raise _Bad("X needs at least 2 different values")
            X = np.column_stack([np.ones(len(x)), x])
            fit, coefs, gof = _glm_report(X, y, ["β0", "β1"], "logit", level, method)
            b0, b1 = fit["b"]
            if b1 == 0:
                raise _Bad("slope is exactly zero")
            c = fit["cov"]
            x50 = -b0 / b1
            gr = np.array([-1 / b1, b0 / b1 ** 2])
            se50 = float(math.sqrt(gr @ c @ gr))
            # profile likelihood of X at 50 %: logit = beta * (X - theta), beta free
            crit = float(stats.chi2.ppf(level, 1))
            if method == "profile":
                def gth(th):
                    f = _glm((x - th)[:, None], y, "logit", b0=[b1])
                    return 2 * (fit["ll"] - f["ll"]) - crit
                ci50 = []
                for sg in (-1, 1):
                    step, hit = se50, None
                    for _ in range(30):
                        if gth(x50 + sg * step) > 0:
                            hit = x50 + sg * step
                            break
                        step *= 2
                    ci50.append(None if hit is None else float(optimize.brentq(gth, *sorted((x50, hit)), xtol=1e-10, rtol=1e-12)))
            else:
                zc = float(stats.norm.ppf(0.5 + level / 2))
                ci50 = [x50 - zc * se50, x50 + zc * se50]
            p = fit["mu"]
            roc = _roc(y, p, level)
            cls = _classify(y, p, cutoff)
            g = np.linspace(float(x.min()), float(x.max()), 200)
            G = np.column_stack([np.ones(200), g])
            eta = G @ fit["b"]
            sef = np.sqrt(np.einsum("ij,jk,ik->i", G, c, G))
            zc = float(stats.norm.ppf(0.5 + level / 2))
        except _Bad as e:
            res[i] = {"status": "failed", "messages": [f"{title}: {e}"]}
            msgs.append(f"{title}: {e}")
            continue
        flags = []
        if len(y) < 10 * 2:
            flags.append("fewer than 10 observations per parameter")
        res[i] = {"status": "ok", "model": "log odds = β0 + β1*X",
                  "params": {"β0": coefs[0], "β1": coefs[1]},
                  "derived": {"x_at_50pct": {"value": x50, "se": se50, "ci_low": ci50[0], "ci_high": ci50[1]}},
                  "coefficients": coefs, "gof": gof,
                  "stats": {"n": len(y), "n1": int(y.sum()), "n0": int(len(y) - y.sum()), "df": len(y) - 2},
                  "roc": roc, "classification": cls, "flags": flags,
                  "predicted": {"x": x, "observed": y, "predicted": p},
                  "curve": {"x": g, "y": special.expit(eta), "ci_low": special.expit(eta - zc * sef), "ci_high": special.expit(eta + zc * sef)}}
        pct = f"{level * 100:g}% CI ({method})"
        rows = [[c_["term"], c_["estimate"], c_["se"], c_["ci_low"], c_["ci_high"], c_["ratio"], c_["ratio_ci_low"], c_["ratio_ci_high"]] for c_ in coefs]
        rows.append(["X when Y=50%", x50, se50, ci50[0], ci50[1], None, None, None])
        display += [
            _tbl(f"{title}: parameters", ["Parameter", "Best-fit", "Std. error", f"{pct} low", f"{pct} high", "Odds ratio", "OR low", "OR high"], rows),
            _tbl(f"{title}: tests", ["Test", "Statistic", "P value", "Summary"],
                 [["Slope different from zero (|Z|)", abs(coefs[1]["stat"]), coefs[1]["p"], p_summary(coefs[1]["p"])],
                  ["Likelihood ratio test (G squared)", gof["lr_chi2"], gof["lr_p"], p_summary(gof["lr_p"])],
                  ["ROC area (SE, CI in table below)", roc["auc"], roc["p"], p_summary(roc["p"])]]),
            _tbl(f"{title}: ROC and goodness of fit", ["Quantity", "Value"],
                 [["ROC area", roc["auc"]], ["ROC std. error", roc["se"]], [f"ROC {level * 100:g}% CI low", roc["ci_low"]],
                  [f"ROC {level * 100:g}% CI high", roc["ci_high"]], ["Tjur's R squared", gof["tjur_r2"]],
                  ["Cox-Snell's R squared", gof["cox_snell_r2"]], ["Model deviance, G squared", gof["deviance"]], ["AICc", gof["aicc"]]]),
            _tbl(f"{title}: classification at cutoff {cutoff:g}", ["Quantity", "Value"],
                 [["True positives", cls["tp"]], ["False positives", cls["fp"]], ["True negatives", cls["tn"]],
                  ["False negatives", cls["fn"]], ["Sensitivity", cls["sensitivity"]], ["Specificity", cls["specificity"]],
                  ["Accuracy", cls["accuracy"]]]),
            _tbl(f"{title}: data summary", ["Quantity", "Value"],
                 [["Rows analyzed", len(y)], ["Number of 1", int(y.sum())], ["Number of 0", int(len(y) - y.sum())],
                  ["Number of parameter estimates", 2], ["#observations/#parameters", len(y) / 2]])]
    if not any(r["status"] == "ok" for r in res.values()):
        raise _Bad("; ".join(msgs))
    return {"datasets": res, "messages": msgs, "display": display}


# --------------------------------------------------------------------------- multiple regression

def _vars(table, ids, params):
    dep = params.get("dependent")
    if not dep:
        raise _Bad("choose the dependent variable")
    known = {d.id for d in table.datasets}
    if dep not in known:
        raise _Bad(f"dependent variable {dep!r} is not in the table")
    preds = list(params.get("predictors") or [i for i in ids if i != dep])
    if dep in preds:
        raise _Bad("the dependent variable cannot also be a predictor")
    if not preds:
        raise _Bad("choose at least one predictor variable")
    if any(p not in known for p in preds):
        raise _Bad("a predictor is not in the table")
    return dep, list(dict.fromkeys(preds))


def _parse_ref(text):
    out = {}
    for part in filter(None, (s.strip() for s in str(text or "").split(";"))):
        if "=" not in part:
            raise _Bad(f"reference levels are 'Variable=Level; ...'; cannot read {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _design(table, ids, params, fam):
    """Model matrix with dummy coding and interactions; rows with any blank / excluded cell are skipped."""
    dep, preds = _vars(table, ids, params)
    cols = {}
    for i in [dep] + preds:
        d = table.dataset(i)
        cols[i] = np.where(d.excluded[:, 0], np.nan, d.values[:, 0])
    keep = ~np.any(np.isnan(np.column_stack([cols[i] for i in [dep] + preds])), axis=1)
    n_skip = int((~keep).sum())
    rows = np.flatnonzero(keep)
    msgs = [f"{n_skip} row(s) with blank or excluded values were skipped"] if n_skip else []
    dd = table.dataset(dep)
    y = cols[dep][keep]
    if fam == "linear":
        if dd.categories is not None:
            raise _Bad("multiple linear regression needs a numeric dependent variable")
    elif fam == "poisson":
        if dd.categories is not None or np.any(y < 0) or np.any(y != np.round(y)):
            raise _Bad("Poisson regression needs a dependent variable of non-negative integer counts")
    else:
        if dd.categories is not None:
            present = [k for k in range(len(dd.categories)) if np.any(y == k)]
            if len(present) != 2:
                raise _Bad(f"the dependent variable needs exactly 2 observed categories, found {len(present)}")
            ev = str(params.get("event") or "").strip() or dd.categories[present[1]]
            if ev not in dd.categories or dd.categories.index(ev) not in present:
                raise _Bad(f"event level {ev!r} is not an observed category of the dependent variable")
            y = (y == dd.categories.index(ev)).astype(float)
            msgs.append(f"outcome 1 = {ev!r}")
        elif not np.all(np.isin(y, (0, 1))):
            raise _Bad("the dependent variable must be 0/1 (or a categorical variable with 2 levels)")
        elif y.min() == y.max():
            raise _Bad("the dependent variable has only one outcome")
    refs = _parse_ref(params.get("reference"))
    blocks = {}  # variable id -> [(name, column)]
    for i in preds:
        d, v = table.dataset(i), cols[i][keep]
        if d.categories is None:
            if np.ptp(v) == 0:
                raise _Bad(f"{d.title}: the predictor is constant")
            blocks[i] = [(d.title, v)]
            continue
        present = [k for k in range(len(d.categories)) if np.any(v == k)]
        if len(present) < 2:
            raise _Bad(f"{d.title}: a categorical predictor needs at least 2 observed levels")
        if len(present) < len(d.categories):
            msgs.append(f"{d.title}: unobserved levels were dropped")
        ref = refs.get(d.title)
        if ref is None:  # Prism: the level that appears first in the column (category id 0)
            first = int(cols[i][~np.isnan(cols[i])][0])
            r = first if first in present else int(v[0])
        elif ref in d.categories and d.categories.index(ref) in present:
            r = d.categories.index(ref)
        else:
            raise _Bad(f"{d.title}: reference level {ref!r} is not an observed level")
        blocks[i] = [(f"{d.title} [{d.categories[k]}]", (v == k).astype(float)) for k in present if k != r]
    names, vecs = ["Intercept"], [np.ones(len(y))]
    for i in preds:
        for nm, v in blocks[i]:
            names.append(nm), vecs.append(v)
    spec = str(params.get("interactions") or "").strip()
    pairs = []
    if spec.lower() == "all":
        pairs = [(a, b) for k, a in enumerate(preds) for b in preds[k + 1:]]
    elif spec:
        by_title = {table.dataset(i).title: i for i in preds}
        for part in filter(None, (s.strip() for s in spec.split(";"))):
            ab = [s.strip() for s in part.replace("*", ":").split(":")]
            if len(ab) != 2 or ab[0] not in by_title or ab[1] not in by_title or ab[0] == ab[1]:
                raise _Bad(f"interaction {part!r}: use 'Variable1:Variable2' with two different predictors")
            pairs.append((by_title[ab[0]], by_title[ab[1]]))
    for a, b in pairs:
        for na, va in blocks[a]:
            for nb, vb in blocks[b]:
                names.append(f"{na} × {nb}"), vecs.append(va * vb)
    X = np.column_stack(vecs)
    n, k = X.shape
    if n <= k:
        raise _Bad(f"{n} complete rows for {k} parameters: need more rows than parameters")
    if np.linalg.matrix_rank(X) < k:
        raise _Bad("the predictors are perfectly collinear")
    return X, y, names, rows, msgs


def _terms_table(title, coefs, level, ratio_label=None, vif=None):
    pct = f"{level * 100:g}% CI"
    cols = ["Term", "Estimate", "Std. error", f"{pct} low", f"{pct} high", "|t| or |Z|", "P value", "Summary"]
    if ratio_label:
        cols += [ratio_label, "low", "high"]
    cols.append("VIF")
    rows = []
    for j, c in enumerate(coefs):
        r = [c["term"], c["estimate"], c["se"], c["ci_low"], c["ci_high"], c["stat"], c["p"], p_summary(c["p"])]
        if ratio_label:
            r += [c["ratio"], c["ratio_ci_low"], c["ratio_ci_high"]]
        r.append(None if j == 0 or vif is None else vif[j - 1])
        rows.append(r)
    return _tbl(title, cols, rows)


@_kind("multiple_linear_regression")
def multiple_linear_regression(table, ids, params):
    level = _level(params)
    X, y, names, rows, msgs = _design(table, ids, params, "linear")
    n, k = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ b
    ss = float(np.sum((y - pred) ** 2))
    df = n - k
    if df < 1:
        raise _Bad("no residual degrees of freedom")
    s2 = ss / df
    se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
    t = b / se
    tc = float(stats.t.ppf(0.5 + level / 2, df))
    coefs = [{"term": names[j], "estimate": float(b[j]), "se": float(se[j]), "ci_low": float(b[j] - tc * se[j]),
              "ci_high": float(b[j] + tc * se[j]), "stat": abs(float(t[j])), "p": float(2 * stats.t.sf(abs(t[j]), df))} for j in range(k)]  # Prism reports |t|
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss / sst
    f = ((sst - ss) / (k - 1)) / s2 if k > 1 and ss > 0 else float("nan")
    gof = {"r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / df, "sy_x": math.sqrt(s2), "ss": ss, "df": df, "f": f, "f_dfn": k - 1,
           "f_p": float(stats.f.sf(f, k - 1, df)) if np.isfinite(f) else None,
           "aicc": _aicc(n * math.log(ss / n), k + 1, n) if ss > 0 else None}
    vif = _vif(X[:, 1:])
    out = {"coefficients": coefs, "gof": gof, "vif": dict(zip(names[1:], vif)), "rows": rows,
           "predicted": {"observed": y, "predicted": pred, "residual": y - pred},
           "stats": {"n": n, "df": df, "n_parameters": k}, "messages": msgs,
           "display": [_terms_table("Parameter estimates", coefs, level, vif=vif),
                       _tbl("Goodness of fit", ["Statistic", "Value"],
                            [["R squared", r2], ["Adjusted R squared", gof["adj_r2"]], ["Sy.x", gof["sy_x"]], ["SSres", ss],
                             ["F (overall)", f], ["DFn, DFd", f"{k - 1}, {df}"], ["P value (overall)", gof["f_p"]], ["AICc", gof["aicc"]]]),
                       _tbl("Data summary", ["Quantity", "Value"], [["Rows analyzed", n], ["Number of parameters", k]])]}
    return out


def _glm_kind(fam, ratio_label):
    def run(table, ids, params):
        level = _level(params)
        method = params.get("ci_method", "profile")
        X, y, names, rows, msgs = _design(table, ids, params, fam)
        n, k = X.shape
        fit, coefs, gof = _glm_report(X, y, names, fam, level, method)
        vif = _vif(X[:, 1:])
        if fam == "logit" and n < 10 * k:
            msgs.append("fewer than 10 observations per parameter: estimates may be unreliable")
        mu = fit["mu"]
        out = {"coefficients": coefs, "gof": gof, "vif": dict(zip(names[1:], vif)), "rows": rows,
               "predicted": {"observed": y, "predicted": mu},
               "stats": {"n": n, "n_parameters": k}, "messages": msgs}
        gofrows = [["-Log likelihood (model)", gof["ll"]], ["-Log likelihood (intercept only)", gof["ll_null"]],
                   ["Deviance (-2 log likelihood)", gof["deviance"]], ["Likelihood ratio chi-square", gof["lr_chi2"]],
                   ["DF", gof["lr_df"]], ["P value (model vs intercept only)", gof["lr_p"]], ["AICc", gof["aicc"]],
                   ["Cox-Snell's R squared", gof["cox_snell_r2"]]]
        if fam == "logit":
            roc = _roc(y, mu, level)
            out["roc"] = roc
            out["classification"] = _classify(y, mu, float(params.get("cutoff", 0.5)))
            hl = _hosmer_lemeshow(y, mu)
            out["hosmer_lemeshow"] = hl
            out["stats"].update({"n1": int(y.sum()), "n0": int(n - y.sum())})
            gofrows += [["Tjur's R squared", gof["tjur_r2"]], ["Nagelkerke's R squared", gof["nagelkerke_r2"]],
                        ["ROC area", roc["auc"]], ["ROC area CI low", roc["ci_low"]], ["ROC area CI high", roc["ci_high"]],
                        ["Hosmer-Lemeshow chi-square", None if hl is None else hl["chi2"]],
                        ["Hosmer-Lemeshow DF", None if hl is None else hl["df"]],
                        ["Hosmer-Lemeshow P value", None if hl is None else hl["p"]]]
        else:
            gofrows += [["McFadden's pseudo R squared", gof["mcfadden_r2"]], ["Model deviance (G squared)", gof["model_deviance"]],
                        ["Deviance P value", gof["deviance_p"]], ["Pearson chi-square", gof["pearson_chi2"]],
                        ["Pearson P value", gof["pearson_p"]], ["Dispersion (Pearson / DF)", gof["dispersion"]]]
            if gof["dispersion"] and gof["dispersion"] > 1.5:
                msgs.append("overdispersion (Pearson chi-square / DF > 1.5): standard errors may be too small")
        out["display"] = [_terms_table("Parameter estimates", coefs, level, ratio_label, vif),
                          _tbl("Goodness of fit", ["Statistic", "Value"], gofrows),
                          _tbl("Data summary", ["Quantity", "Value"], [["Rows analyzed", n], ["Number of parameters", k]])]
        return out
    return run


multiple_logistic_regression = _kind("multiple_logistic_regression")(_glm_kind("logit", "Odds ratio"))
poisson_regression = _kind("poisson_regression")(_glm_kind("poisson", "Rate ratio"))


# --------------------------------------------------------------------------- registry

_CH = lambda *a: tuple((v, l) for v, l in a)  # noqa: E731
_MV = (
    Param("dependent", "Dependent variable", "dataset", None),
    Param("predictors", "Predictor variables (blank = all others)", "datasets", None),
    Param("interactions", "Interactions ('all' or 'A:B; C:D')", "str", ""),
    Param("reference", "Reference levels ('Var=Level; ...', default: first level in order of appearance)", "str", ""),
    _LEVEL,
)
_GLM_OPT = (Param("ci_method", "CI method", "choice", "profile", _CH(("profile", "Profile likelihood"), ("wald", "Asymptotic (Wald)"))),)

KINDS = {
    "linear_regression": AnalysisKind(
        "linear_regression", "Simple linear regression", ("XY",), linear_regression,
        (Param("fit_to", "Fit", "choice", "replicates", _CH(("replicates", "Each replicate"), ("means", "Mean of replicates"))),
         Param("through", "Constraint", "choice", "none", _CH(("none", "None"), ("origin", "Through the origin"), ("point", "Through a fixed point"))),
         Param("x0", "Fixed point X", "float", 0.0), Param("y0", "Fixed point Y", "float", 0.0),
         Param("compare", "Compare slopes and intercepts", "bool", True), Param("alpha", "Significance level", "float", 0.05),
         Param("runs_test", "Runs test for deviation from linearity", "bool", False),
         Param("normality", "Test residuals for normality", "bool", False), _LEVEL)),
    "deming_regression": AnalysisKind(
        "deming_regression", "Deming regression", ("XY",), deming_regression,
        (Param("fit_to", "Fit", "choice", "replicates", _CH(("replicates", "Each replicate"), ("means", "Mean of replicates"))),
         Param("error_ratio", "SD of Y errors / SD of X errors", "float", 1.0), _LEVEL)),
    "simple_logistic_regression": AnalysisKind(
        "simple_logistic_regression", "Simple logistic regression", ("XY",), simple_logistic_regression,
        _GLM_OPT + (Param("cutoff", "Classification cutoff", "float", 0.5), _LEVEL)),
    "multiple_linear_regression": AnalysisKind(
        "multiple_linear_regression", "Multiple linear regression", ("MultipleVariables",), multiple_linear_regression, _MV),
    "multiple_logistic_regression": AnalysisKind(
        "multiple_logistic_regression", "Multiple logistic regression", ("MultipleVariables",), multiple_logistic_regression,
        _MV + _GLM_OPT + (Param("event", "Outcome level coded 1 (categorical dependent; default last observed)", "str", ""),
                          Param("cutoff", "Classification cutoff", "float", 0.5))),
    "poisson_regression": AnalysisKind(
        "poisson_regression", "Poisson regression", ("MultipleVariables",), poisson_regression, _MV + _GLM_OPT),
}

