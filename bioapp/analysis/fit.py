"""Nonlinear regression of XY tables: Prism's built-in equation library (``models.py``), global fits, interpolation.

Equations, parameter names and sign conventions follow the Prism Curve Fitting Guide
(``Y = Bottom + (Top-Bottom)/(1+10^((LogEC50-X)*HillSlope))`` etc.). Each data set is fitted
separately with ``scipy.optimize.least_squares`` unless parameters are shared (the ``shared`` option, or the
model's default for global models such as total+nonspecific binding, Schild, enzyme inhibition): then one fit
minimises the total SS of all selected data sets. Standard errors come from the Jacobian covariance
scaled by SS/DF; 1/Y^2 weighting is iteratively reweighted (weights 1/Yhat^2 frozen inside each step, as
Prism); AICc counts K = parameters + 1; confidence intervals are profile-likelihood (Prism's default, asymmetric) or
asymptotic (t based, symmetric). A fit is "ambiguous" when a parameter dependency exceeds 0.9999
(Prism's PAR_DEPENDENCY rule); such parameters get no SE / interval, never a made-up one. An open
profile limit is reported separately as "incomplete_ci" (Prism's INCOMPLETE_CI warning).

Interpolation (option ``interpolate``) follows Prism: X is found numerically (first of 1000 segments that contains
Y, then bisection; the search range is the data range, then the range widened by half the data range on each
side = extrapolation), its confidence interval is where the confidence bands of the curve cross Y (none when
extrapolated or when the fit is ambiguous). Like Prism, unknowns are interpolated for the row means
(``interpolated``) and for every replicate (``interpolated_replicates``); option ``interpolate_replicates``
= "both" (default), "mean" or "each" (``interpolated`` then holds the individual replicates).
Global fits: the overall R squared uses the total SS about the pooled mean of all data sets (Prism), each data
set's own R squared the SS about its own mean.
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import expit
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from ..data import Table
from . import AnalysisKind, Param
from .models import ALL, MODELS, Model, initial_values  # noqa: F401  (MODELS is part of this module's API)

TOL = 1e-14
AMBIGUOUS_DEPENDENCY = 0.9999
CURVE_POINTS = 200
INTERP_SEGMENTS = 1000


# ---------------------------------------------------------------- generic least squares

def _run(resid, p0, lo, hi, tol, nfev=None):
    if np.isfinite(lo).any() or np.isfinite(hi).any():
        return least_squares(resid, p0, bounds=(lo, hi), method="trf", x_scale="jac",
                             ftol=tol, xtol=tol, gtol=tol, max_nfev=nfev or 2000)
    return least_squares(resid, p0, method="lm", ftol=tol, xtol=tol, gtol=tol, max_nfev=nfev or 5000)


IRLS_MAX = 100
IRLS_TOL = 1e-10


def _lsq(resid: Callable, p0, lo, hi, quick=False, reweight: Callable | None = None):
    """Tight tolerances first; if that stalls (parameters drifting along a flat direction), accept the
    solution reached with looser tolerances (the fit is then normally flagged ambiguous).
    ``reweight(p)`` (iteratively reweighted least squares, Prism's 1/Y^2) returns a residual function whose
    weights are frozen at the curve of ``p``; the fit is repeated until the parameters stop changing.
    The result carries the residual function of its last step as ``.resid``."""
    p0 = np.clip(np.asarray(p0, float), lo, hi)
    with np.errstate(all="ignore"):
        if quick:  # profile refits: modest tolerance and iteration cap
            r = _run(resid, p0, lo, hi, 1e-10, 300)
        else:
            r = _run(resid, p0, lo, hi, TOL)
            if r.status <= 0:
                r2 = _run(resid, np.clip(r.x, lo, hi), lo, hi, 1e-10)
                if r2.status > 0 and r2.cost <= r.cost * (1 + 1e-9):
                    r = r2
    if reweight is not None:
        tol, cap = (1e-7, 30) if quick else (IRLS_TOL, IRLS_MAX)  # ponytail: no IRLS flag; flat directions stop at the cap
        for _ in range(cap):
            resid = reweight(r.x)
            r2 = _lsq(resid, r.x, lo, hi, quick)
            done = bool(np.all(np.abs(r2.x - r.x) <= tol * (1 + np.abs(r.x))))
            r = r2
            if done:
                break
    r.resid = resid
    return r


def _jacobian(resid, p):
    """Central-difference Jacobian (step relative to the parameter; floor 1e-6 for parameters near zero)."""
    cols = []
    for i in range(len(p)):
        h = 1e-5 * max(abs(p[i]), 1e-6)
        a, b = p.copy(), p.copy()
        a[i] += h
        b[i] -= h
        cols.append((resid(a) - resid(b)) / (2 * h))
    return np.column_stack(cols)

EIG_FLOOR = 1e-15  # relative eigenvalue floor of the normalised information matrix (singular directions)


def generic_fit(resid: Callable, p0, lo=None, hi=None, reweight: Callable | None = None) -> dict:
    """Least squares of ``resid(p)`` (weighted residual vector). Returns p, ss, df, n, converged,
    cov (SS/DF scaled, or None), se (or None), dep (Prism parameter dependency = 1 - 1/(C_ii (C^-1)_ii);
    1.0 only for the parameters that take part in a singular direction) and the final ``resid``.

    ``reweight(p)`` (iteratively reweighted least squares, Prism's 1/Y^2) returns a residual function whose
    weights are frozen at the curve of ``p``; the fit is repeated until the parameters stop changing, so
    ss, cov and dep describe the frozen-weight problem at the solution (J'WJ with the plain Jacobian)."""
    p0 = np.asarray(p0, float)
    k = len(p0)
    lo = np.full(k, -np.inf) if lo is None else np.asarray(lo, float)
    hi = np.full(k, np.inf) if hi is None else np.asarray(hi, float)
    r = _lsq(resid, p0, lo, hi, reweight=reweight)
    resid = r.resid
    n = len(r.fun)
    ss = float(r.fun @ r.fun)
    df = n - k
    out = {"p": r.x, "ss": ss, "n": n, "df": df, "k": k, "converged": bool(r.status > 0 and np.isfinite(ss)),
           "cov": None, "se": None, "dep": np.ones(k), "resid": resid}
    with np.errstate(all="ignore"):
        jac = _jacobian(resid, r.x)
        norm = np.linalg.norm(jac, axis=0)
        if not np.all(np.isfinite(jac)) or not np.any(norm > 0):
            return out
        zero = norm == 0  # a parameter without effect on the curve: dependency 1, others unaffected
        norm[zero] = 1
        jn = jac / norm
        hn = jn.T @ jn
        ev, vec = np.linalg.eigh(hn)
        hinv = (vec / np.maximum(ev, EIG_FLOOR * ev.max())) @ vec.T  # floored inverse: only singular directions blow up
        dep = np.clip(1 - 1 / (np.diag(hn) * np.diag(hinv)), 0, 1)
    dep[zero] = 1
    out["dep"] = np.where(np.isfinite(dep), dep, 1.0)
    if df > 0:
        out["cov"] = ss / df * hinv / np.outer(norm, norm)
        out["se"] = np.sqrt(np.diag(out["cov"]))
    return out


def _profile_limits(resid, p, j, ss, df, step0, lo, hi, level, reweight=None):
    """Profile-likelihood limits of free parameter j: where refitting the others with j held fixed
    raises SS to SS*(1 + F(level,1,df)/df) (Prism: extra sum-of-squares F test, 1 DF).
    ``reweight`` (1/Y^2) makes every refit an iteratively reweighted fit, as Prism does.
    Returns (low, high); None = open side (no finite limit)."""
    target = ss * (1 + f_dist.ppf(level, 1, df) / df)
    others = [i for i in range(len(p)) if i != j]
    cur = {"q": p[others].copy()}

    def g(v):
        def embed(q):
            full = np.empty(len(p))
            full[j], full[others] = v, q
            return full

        if not others:
            r = (reweight(np.array([v])) if reweight else resid)(np.array([v]))
            return float(r @ r) - target

        sub = lambda q: resid(embed(q))  # noqa: E731
        rw = None if reweight is None else lambda qref: (lambda q: reweight(embed(qref))(embed(q)))  # noqa: E731
        # two starts (last solution, best fit); keep the lower SS so a stalled refit cannot fake a limit
        fits = [_lsq(sub, q0, lo[others], hi[others], quick=True, reweight=rw) for q0 in (cur["q"], p[others])]
        fit = min(fits, key=lambda r: r.cost)
        cur["q"] = fit.x
        return float(fit.fun @ fit.fun) - target

    cache = {p[j]: ss - target}  # brentq must see the same value at the bracket ends that the search saw
    gc = lambda v: cache[v] if v in cache else cache.setdefault(v, g(v))  # noqa: E731

    limits = []
    for sign, bound in ((-1, lo[j]), (1, hi[j])):
        cur["q"] = p[others].copy()
        a, step, limit = p[j], step0, None
        for _ in range(20):  # up to ~1e6 standard errors, then the side counts as open
            v = p[j] + sign * step
            hit = v <= bound if sign < 0 else v >= bound
            if hit:
                v = bound
            if gc(v) > 0:
                a_, b_ = sorted((a, v))
                limit = brentq(gc, a_, b_, xtol=1e-7 * step0, rtol=1e-12)
                break
            if hit:
                limit = float(bound)  # interval runs to the user constraint
                break
            a, step = v, step * 2
        limits.append(limit)
    return tuple(limits)


# ---------------------------------------------------------------- data of one data set

def _make_resid(func, x, y, sd, weighting, ref=None):
    """Weighted residuals of the full parameter vector. 1/Y^2: weights 1/f^2 of the curve at parameters
    ``ref`` when given (held fixed, as in Prism's iteratively reweighted fit), else of the curve itself."""
    f0 = None if ref is None else func(x, ref)

    def resid(p):
        f = func(x, p)
        if weighting == "1/Y^2":
            return (y - f) / (f if f0 is None else f0)
        if weighting == "1/SD^2":
            return (y - f) / sd
        return y - f
    return resid


def _prep(table: Table, ds_id: str, fit_to: str, weighting: str):
    """x, y, sd (or None), number of replicate values behind the fitted points."""
    title = table.dataset(ds_id).title
    if fit_to == "replicates" and table.y_format == "replicates":
        x, y, _ = table.points(ds_id)
        sd, nobs = None, float(len(y))
    else:
        s = table.summary(ds_id)
        ok = ~np.isnan(table.x) & ~np.isnan(s["mean"])
        x, y, sd = table.x[ok], s["mean"][ok], s["sd"][ok]
        nobs = float(np.nansum(s["n"][ok])) if np.any(~np.isnan(s["n"][ok])) else None
    if weighting == "1/SD^2" and (not np.all(np.isfinite(sd)) or np.any(sd <= 0)):
        raise ValueError(f"{title}: 1/SD^2 weighting needs a positive SD in every row")
    return x, y, sd, nobs


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _title_number(title: str) -> float | None:
    m = _NUMBER.search(title or "")
    return float(m.group()) if m else None


# ---------------------------------------------------------------- the fit (one or several data sets)

def _fit_group(m: Model, dss: list, shared, lo, hi, level, method, weighting, ci=True, start=None) -> dict:
    """Fit data sets ``dss`` together: parameters named in ``shared`` take one value for all data sets that use
    them, every other parameter is fitted per data set. One data set without sharing is an ordinary fit.

    Each entry of ``dss``: x, y, sd, func, used (bool per parameter), fixed (NaN = fitted), cvals.
    ``start``: optional full parameter vector per data set (replaces the data-based starting values).
    ``ci=False`` stops after the fit (comparison fits need only SS and DF).
    Returns a dict with ok, flags [(code, text, scope)] (scope None = applies to all data sets), and per data
    set ``views`` (full parameter vector, SE, CI, dependency, covariance, flags-relevant sets)."""
    k, D = len(m.params), len(dss)
    flags: list = []
    res = {"ok": False, "flags": flags, "dss": dss}
    free_d = [[i for i in range(k) if ds["used"][i] and np.isnan(ds["fixed"][i])] for ds in dss]
    if not any(free_d):
        flags.append(("all_fixed", "every parameter is constant, nothing to fit", None))
        return res
    sh = [i for i in range(k) if m.params[i] in shared]
    vars_ = [(i, [d for d in range(D) if i in free_d[d]]) for i in sh]
    vars_ = [v for v in vars_ if v[1]] + [(i, [d]) for d in range(D) for i in free_d[d] if i not in sh]
    nv, n_all = len(vars_), sum(len(ds["x"]) for ds in dss)
    if n_all <= nv:
        flags.append(("too_few_points", f"too few points: {n_all} points for {nv} parameters to fit", None))
        return res
    for d, ds in enumerate(dss):
        if m.x_scale == "dose" and np.any(ds["x"] < 0):
            flags.append(("negative_x", "dose-scale model needs X >= 0 (X is used as given, not converted)", d))
            return res
    assign = [[(i, j) for j, (i, users) in enumerate(vars_) if d in users] for d in range(D)]

    def full(d, q):
        p = dss[d]["fixed"].copy()
        for i, j in assign[d]:
            p[i] = q[j]
        return p

    def make(ref=None):
        fns = [_make_resid(ds["func"], ds["x"], ds["y"], ds["sd"], weighting, None if ref is None else full(d, ref))
               for d, ds in enumerate(dss)]
        return lambda q: np.concatenate([fn(full(d, q)) for d, fn in enumerate(fns)])

    lo_v, hi_v = np.array([lo[i] for i, _ in vars_]), np.array([hi[i] for i, _ in vars_])
    try:
        p0s = start if start is not None else [initial_values(m, ds["x"], ds["y"], ds["cvals"]) for ds in dss]
        q0 = []
        for i, users in vars_:
            vals = np.array([p0s[d][i] for d in users], float)
            q0.append(float(np.mean(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else (1.0 if lo[i] >= 0 else 0.0))
        gf = generic_fit(make(), q0, lo_v, hi_v, make if weighting == "1/Y^2" else None)
    except (ValueError, np.linalg.LinAlgError, IndexError, KeyError) as e:
        flags.append(("not_converged", f"fit failed: {e}", None))
        return res
    if not gf["converged"]:
        flags.append(("not_converged", "the fit did not converge", None))
        return res
    q, resid_free = gf["p"], gf["resid"]  # 1/Y^2: weights frozen at the solution
    res.update(ok=True, gf=gf, q=q, vars=vars_, full=full, ss=gf["ss"], df=gf["df"], nv=nv, n=n_all, amb=set())
    if not ci:
        return res

    dep = gf["dep"]
    amb = {j for j in range(nv) if dep[j] > AMBIGUOUS_DEPENDENCY} if gf["se"] is not None else set(range(nv))
    se = np.full(nv, np.nan)
    if gf["se"] is not None:
        se[:] = gf["se"]
    tcrit = t_dist.ppf(0.5 + level / 2, gf["df"]) if gf["df"] > 0 else np.nan
    cis = [[None, None] for _ in range(nv)]
    open_side = set()
    for j in range(nv):
        if j in amb or gf["se"] is None:
            continue
        if method == "asymptotic":
            cis[j] = [float(q[j] - tcrit * se[j]), float(q[j] + tcrit * se[j])]
        else:
            step0 = se[j] if se[j] > 0 else 0.1 * abs(q[j]) + 1e-3
            cis[j] = list(_profile_limits(resid_free, q, j, gf["ss"], gf["df"], step0, lo_v, hi_v, level, make if weighting == "1/Y^2" else None))
            if None in cis[j]:
                open_side.add(j)
    for j in amb:
        se[j] = np.nan

    rr = resid_free(q)
    views, off = [], 0
    for d, ds in enumerate(dss):
        n_d = len(ds["x"])
        ss_d = gf["ss"] if D == 1 else float(rr[off:off + n_d] @ rr[off:off + n_d])
        off += n_d
        p = full(d, q)
        f = ds["func"](ds["x"], p)
        scale = {"1/Y^2": np.abs(f), "1/SD^2": ds["sd"]}.get(weighting)
        w = np.ones_like(ds["y"]) if scale is None else 1 / scale ** 2
        ybar = np.sum(w * ds["y"]) / np.sum(w)
        sst = float(np.sum(w * (ds["y"] - ybar) ** 2))
        se_d, dep_d, ci_d = np.full(k, np.nan), np.full(k, np.nan), {i: [None, None] for i in range(k)}
        cov_d = None if gf["cov"] is None else np.zeros((k, k))
        for i, j in assign[d]:
            se_d[i], dep_d[i], ci_d[i] = se[j], dep[j], cis[j]
            if cov_d is not None:
                for i2, j2 in assign[d]:
                    cov_d[i, i2] = gf["cov"][j, j2]
        for n_, f_ in m.se_scale:
            se_d[m.params.index(n_)] *= f_
        views.append({
            "w_y": (w, ds["y"]),
            "p": p, "used": ds["used"], "free": free_d[d], "n": n_d, "ss": ss_d, "se": se_d, "dep": dep_d, "ci": ci_d,
            "amb": {i for i, j in assign[d] if j in amb}, "open_side": {i for i, j in assign[d] if j in open_side},
            "cov": cov_d, "tcrit": tcrit, "r2": 1 - ss_d / sst if sst > 0 else None, "sst": sst,
            "shared_params": {i for i, j in assign[d] if len(vars_[j][1]) > 1},
            "sy_x": float(np.sqrt(gf["ss"] / gf["df"])) if gf["df"] > 0 else None,
            "aicc": _aicc(gf["ss"], n_all, nv) if D == 1 else None})
    res.update(views=views, se=se, ci=cis, dep=dep, amb=amb, open_side=open_side, tcrit=tcrit,
               sy_x=views[0]["sy_x"], aicc=_aicc(gf["ss"], n_all, nv))

    # flags: ambiguous, incomplete CI, at a constraint bound, potency outside the X range
    for j in sorted(amb):
        i, users = vars_[j]
        scope = None if len(users) > 1 else users[0]
        flags.append(("ambiguous", f"ambiguous: {m.params[i]} is not determined by the data"
                      f" (dependency {dep[j]:.5f})" if gf["se"] is not None
                      else f"ambiguous: {m.params[i]} cannot be separated from the other parameters", scope))
    for j in sorted(open_side):
        i, users = vars_[j]
        a_, b_ = cis[j]
        side = "lower and upper" if a_ is None and b_ is None else ("lower" if a_ is None else "upper")
        # Prism: warning INCOMPLETE_CI, not "ambiguous" (ambiguity is the dependency rule only)
        flags.append(("incomplete_ci", f"incomplete CI: the {side} confidence limit of {m.params[i]} is open (not defined by the data)",
                      None if len(users) > 1 else users[0]))
    for j in range(nv):
        i, users = vars_[j]
        for b in (lo_v[j], hi_v[j]):
            if np.isfinite(b) and abs(q[j] - b) <= 1e-9 * (1 + abs(b)):
                flags.append(("at_bound", f"{m.params[i]} is at its constraint bound ({b:g})", None if len(users) > 1 else users[0]))
    pot = m.potency
    if pot is not None:
        ip = m.params.index(pot)
        for d, ds in enumerate(dss):
            v = views[d]["p"][ip]
            if v < ds["x"].min() or v > ds["x"].max():
                flags.append(("potency_outside_x", f"best-fit {pot} ({v:.4g}) is outside the X range of the data; "
                              "the plateau(s) are not defined by the data", d))
    return res


# ---------------------------------------------------------------- curve, derived values

def _grid(scale: str, x) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if scale == "dose" and lo > 0 and hi / lo > 100:
        return np.geomspace(lo, hi, CURVE_POINTS)
    return np.linspace(lo, hi, CURVE_POINTS)


def _entry(v, se=None, lo=None, hi=None, **extra):
    fl = lambda a: None if a is None or (isinstance(a, float) and np.isnan(a)) else float(a)  # noqa: E731
    return {"value": fl(v), "se": fl(se), "ci_low": fl(lo), "ci_high": fl(hi), **extra}


def _fin(v):
    return None if v is None or not np.isfinite(v) else float(v)


def _derived(m: Model, res: dict, level: float) -> dict:
    """EC50 = 10^LogEC50 (or log10 of EC50), Span = Top - Bottom, X-intercept of a line, plus the model's own
    derived values (transforms of one parameter keep the transformed CI; others use the delta method)."""
    names, p, used = m.params, res["p"], res["used"]
    k = len(names)
    out = {}
    sh = res["shared_params"]
    amb = res["amb"]

    def mark(e, involved):
        if sh and involved and set(involved) <= sh:
            e["shared"] = True
        return e

    pot = m.potency
    if pot is not None and used[names.index(pot)]:
        i = names.index(pot)
        a, b = res["ci"][i] if i in res["ci"] else [None, None]
        if pot.startswith("Log"):
            p10 = lambda v: None if v is None or not np.isfinite(np.power(10.0, v)) else float(np.power(10.0, v))  # noqa: E731
            out[pot[3:]] = mark({"value": p10(p[i]), "ci_low": p10(a), "ci_high": p10(b), "ambiguous": i in amb}, [i])
        else:
            ok = p[i] > 0
            out["Log" + pot] = mark({"value": float(np.log10(p[i])) if ok else None,
                                     "ci_low": float(np.log10(a)) if a is not None and a > 0 else None,
                                     "ci_high": float(np.log10(b)) if b is not None and b > 0 else None,
                                     "ambiguous": i in amb}, [i])
    cov = res["cov"] if res["cov"] is not None else np.zeros((k, k))

    def delta(name, value, grad, involved):
        ok = res["cov"] is not None and not (set(involved) & amb)
        se = float(np.sqrt(grad @ cov @ grad)) if ok else None
        t_ = res["tcrit"]
        out[name] = mark({"value": float(value), "se": se, "ambiguous": bool(set(involved) & amb),
                          "ci_low": None if se is None else float(value - t_ * se),
                          "ci_high": None if se is None else float(value + t_ * se)}, involved)

    if "Top" in names and "Bottom" in names:
        g = np.zeros(k)
        g[names.index("Top")], g[names.index("Bottom")] = 1, -1
        delta("Span", p[names.index("Top")] - p[names.index("Bottom")], g, [names.index("Top"), names.index("Bottom")])
    if m.id == "line" and p[0] != 0:
        delta("X-intercept", -p[1] / p[0], np.array([p[1] / p[0] ** 2, -1 / p[0]]), [0, 1])

    with np.errstate(all="ignore"):
        for spec in m.derived:
            name = spec[1]
            if spec[0] == "t":  # monotone function of one parameter: transform value and limits
                i, fn = names.index(spec[2]), spec[3]
                if not used[i]:
                    continue
                tf = lambda v: _fin(fn(v)) if v is not None else None  # noqa: E731
                h = 1e-6 * (1 + abs(p[i]))
                up, dn = tf(p[i] + h), tf(p[i] - h)
                a, b = res["ci"][i]
                fa, fb = tf(a), tf(b)
                dec = up is not None and dn is not None and up < dn
                lo_, hi_ = (fb, fa) if dec else (fa, fb)
                out[name] = mark({"value": tf(p[i]), "ci_low": lo_, "ci_high": hi_, "ambiguous": i in amb}, [i])
            else:  # delta method with a numerical gradient
                fn = spec[2]
                q = dict(zip(names, p))
                try:
                    val = fn(q)
                except (ZeroDivisionError, ValueError, TypeError):
                    continue
                if not np.isfinite(val):
                    continue
                grad = np.zeros(k)
                for i in range(k):
                    if used[i] and np.isfinite(p[i]):
                        h = 1e-6 * max(abs(p[i]), 1e-6)
                        qa, qb = dict(q), dict(q)
                        qa[names[i]], qb[names[i]] = p[i] + h, p[i] - h
                        grad[i] = (fn(qa) - fn(qb)) / (2 * h)
                grad = np.where(np.isfinite(grad), grad, 0.0)
                delta(name, val, grad, [i for i in res["free"] if grad[i] != 0])
    return out


# ---------------------------------------------------------------- comparison (shared vs separate parameters)

def _aicc(ss, n, nparams):
    """Prism's AICc: K = fitted parameters + 1 (the variance counts as a parameter)."""
    k = nparams + 1
    if ss <= 0 or n - k - 1 <= 0:
        return None
    return float(n * np.log(ss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1))


def _compare(m: Model, dss: list, alts: list, pnames: list, shared, lo, hi, alpha: float, weighting: str) -> dict:
    """Extra sum-of-squares F test (and AICc) of one shared value of each parameter in ``pnames`` vs the
    current fit ``alts`` (separate fits per data set, or one global fit with the sharing ``shared``)."""
    label = ", ".join(pnames)
    skip = lambda why: {"status": "skipped", "parameter": label, "parameters": list(pnames), "reason": why}  # noqa: E731
    if len(dss) < 2:
        return skip("comparison needs at least two data sets")
    idx = [m.params.index(n) for n in pnames]
    for n, j in zip(pnames, idx):
        if all(not ds["used"][j] or not np.isnan(ds["fixed"][j]) for ds in dss):
            return skip(f"{n} is constant, nothing to compare")
        if n in shared:
            return skip(f"{n} is already shared in this fit")
    if any(not g["ok"] or g["amb"] for g in alts):
        return skip("a data set failed or its fit is ambiguous (Prism skips the comparison)")
    start = [None] * len(dss)
    for g in alts:
        for ds, v in zip(g["dss"], g["views"]):
            start[next(i for i, x in enumerate(dss) if x is ds)] = v["p"]
    try:
        null = _fit_group(m, dss, tuple(shared) + tuple(pnames), lo, hi, 0.95, "asymptotic", weighting, ci=False, start=start)
    except (ValueError, np.linalg.LinAlgError):
        return skip("the shared-parameter fit did not converge")
    if not null["ok"]:
        return skip("the shared-parameter fit did not converge")
    ss_null = null["ss"]
    n = null["n"]
    k_alt, k_null = sum(g["nv"] for g in alts), null["nv"]
    ss_alt = sum(g["ss"] for g in alts)
    df_alt, df_null = n - k_alt, n - k_null
    if ss_alt <= 0 or df_alt <= 0:
        return skip("the separate fits are perfect or have no degrees of freedom")
    dfn = df_null - df_alt
    f = max((ss_null - ss_alt) / dfn / (ss_alt / df_alt), 0.0)
    p_val = float(f_dist.sf(f, dfn, df_alt))
    a_null, a_alt = _aicc(ss_null, n, k_null), _aicc(ss_alt, n, k_alt)
    prob_sep = None if a_null is None or a_alt is None else float(expit(0.5 * (a_null - a_alt)))
    # value of the parameter(s) under the null hypothesis: the shared variable, or None if none was created
    sv = {}
    for j, (i, users) in enumerate(null["vars"]):
        if m.params[i] in pnames and len(users) > 1:
            sv[m.params[i]] = float(null["q"][j])
    return {
        "status": "ok", "parameter": label, "parameters": list(pnames), "alpha": alpha,
        "shared_value": sv[pnames[0]] if len(pnames) == 1 and pnames[0] in sv else sv,
        "null": {"label": f"One shared value of {label} for all data sets", "ss": ss_null, "df": int(df_null), "aicc": a_null},
        "alt": {"label": f"Separate values of {label} for each data set", "ss": float(ss_alt), "df": int(df_alt), "aicc": a_alt},
        "f": float(f), "dfn": int(dfn), "dfd": int(df_alt), "p": p_val,
        "preferred": "separate" if p_val < alpha else "shared",
        "aicc_diff": None if prob_sep is None else float(a_null - a_alt),  # > 0 favours separate values
        "prob_separate": prob_sep, "prob_shared": None if prob_sep is None else 1 - prob_sep,
        "aicc_preferred": None if prob_sep is None else ("separate" if prob_sep > 0.5 else "shared"),
    }


# ---------------------------------------------------------------- bands and interpolation

def _band_fn(g: dict, d: int, level: float, kind: str, weighting: str):
    """band(xs) -> (curve, half width) of the confidence ("confidence") or prediction band of data set d
    (delta method on the joint covariance); None when the fit has no usable covariance (ambiguous)."""
    gf = g["gf"]
    if gf["cov"] is None or g["amb"] or g["df"] <= 0 or not np.all(np.isfinite(gf["cov"])):
        return None
    if kind == "prediction" and weighting == "1/SD^2":
        return None  # the SD at an arbitrary X is unknown
    ds, q, full = g["dss"][d], g["q"], g["full"]
    tcrit = t_dist.ppf(0.5 + level / 2, g["df"])
    s2 = g["ss"] / g["df"]

    def band(xs):
        fn = lambda qq: ds["func"](xs, full(d, qq))  # noqa: E731
        with np.errstate(all="ignore"):
            f = fn(q)
            jac = _jacobian(fn, q)
            var = np.einsum("ij,jk,ik->i", jac, gf["cov"], jac)
            if kind == "prediction":
                var = var + s2 * (f ** 2 if weighting == "1/Y^2" else 1.0)
            return f, tcrit * np.sqrt(np.maximum(var, 0.0))
    return band


def _first_cross(fn, xs, vals, y):
    """Lowest X where the curve (sampled as ``vals`` on ``xs``) reaches y: segment scan, then root finding."""
    v = vals - y
    ok = np.isfinite(v)
    hit = np.nonzero(ok[:-1] & ok[1:] & (v[:-1] * v[1:] <= 0))[0]
    if not len(hit):
        return None
    i = hit[0]
    if v[i] == 0:
        return float(xs[i])
    if v[i + 1] == 0:
        return float(xs[i + 1])
    with np.errstate(all="ignore"):
        return float(brentq(lambda t: float(fn(np.array([t]))[0]) - y, xs[i], xs[i + 1],
                            xtol=1e-10 * (xs[i + 1] - xs[i]), rtol=1e-13))


def _unknowns(table: Table, ds_id: str, each: bool):
    """(rows with Y but no X: (row, sub, y, n)), (rows with X but no Y: (row, x))."""
    ds = table.dataset(ds_id)
    v = np.where(ds.excluded, np.nan, ds.values)
    if table.y_format != "replicates":
        v = v[:, :1]
    ys, xs = [], []
    for r in range(table.n_rows):
        vals = v[r][~np.isnan(v[r])]
        if np.isnan(table.x[r]) and len(vals):
            if each:
                ys += [(r, s, float(y), 1) for s, y in enumerate(v[r]) if not np.isnan(y)]
            else:
                ys.append((r, None, float(vals.mean()), len(vals)))
        elif not np.isnan(table.x[r]) and not len(vals):
            xs.append((r, float(table.x[r])))
    return ys, xs


def _interpolate(table: Table, ds_id: str, m: Model, g: dict, d: int, level: float, params: dict, each: bool) -> list:
    """Standard-curve interpolation for one fitted data set (see module docstring); ``each``: every replicate
    of an unknown separately instead of the row mean."""
    ys, xs_known = _unknowns(table, ds_id, each)
    if not ys and not xs_known:
        return []
    ds, p = g["dss"][d], g["views"][d]["p"]

    def curve(xs):
        with np.errstate(all="ignore"):
            return np.asarray(ds["func"](xs, p), float)

    xmin, xmax = float(ds["x"].min()), float(ds["x"].max())
    geom = m.x_scale == "dose" and xmin > 0 and xmax > xmin
    ext = ((xmin * (xmin / xmax) ** 0.5, xmax * (xmax / xmin) ** 0.5) if geom else
           (xmin - (xmax - xmin) / 2, xmax + (xmax - xmin) / 2))
    grid = (lambda a, b, n: np.geomspace(a, b, n)) if geom else (lambda a, b, n: np.linspace(a, b, n))
    kind = params.get("interpolate_band", "confidence")
    weighting = g["weighting"]
    band = _band_fn(g, d, level, kind, weighting)
    ga, gb = grid(xmin, xmax, INTERP_SEGMENTS + 1), grid(ext[0], ext[1], 2 * INTERP_SEGMENTS + 1)
    fa, fb = curve(ga), curve(gb)
    bands = None
    if band is not None and xmax > xmin:
        f_, h_ = band(gb)
        if h_ is not None:
            bands = (lambda xs: (lambda r: r[0] + r[1])(band(xs)), f_ + h_, lambda xs: (lambda r: r[0] - r[1])(band(xs)), f_ - h_)
    out = []
    for row, sub, y, n in ys:
        e = {"row": row, "sub": sub, "n": n, "direction": "x", "y": y, "x": None, "x_ci_low": None, "x_ci_high": None,
             "extrapolated": False}
        x = _first_cross(curve, ga, fa, y) if xmax > xmin else None
        if x is None and xmax > xmin:
            x = _first_cross(curve, gb, fb, y)
            e["extrapolated"] = x is not None
        e["x"] = x
        if x is not None and not e["extrapolated"] and bands is not None:
            up = _first_cross(bands[0], gb, bands[1], y)
            dn = _first_cross(bands[2], gb, bands[3], y)
            inc = curve(np.array([x + 1e-6 * (xmax - xmin)]))[0] > curve(np.array([x - 1e-6 * (xmax - xmin)]))[0]
            e["x_ci_low"], e["x_ci_high"] = (up, dn) if inc else (dn, up)
        out.append(e)
    for row, x in xs_known:
        y = float(curve(np.array([x]))[0])
        e = {"row": row, "sub": None, "n": 0, "direction": "y", "x": x, "y": _fin(y), "x_ci_low": None, "x_ci_high": None,
             "y_ci_low": None, "y_ci_high": None, "extrapolated": bool(x < xmin or x > xmax)}
        if band is not None:
            f_, h_ = band(np.array([x]))
            if h_ is not None:
                e["y_ci_low"], e["y_ci_high"] = _fin(f_[0] - h_[0]), _fin(f_[0] + h_[0])
        out.append(e)
    return sorted(out, key=lambda e: e["row"])


# ---------------------------------------------------------------- the analysis

def _failed(messages, **kw):
    return {"kind": "nonlin_fit", "status": "failed", "messages": messages, "display": [], "datasets": {}, **kw}


def _fmt(v):
    return "" if v is None else f"{v:.4g}"


def _ci_cell(ent: dict | None, ambiguous: bool):
    if ent is None or ent["value"] is None:
        return None
    a, b = ent["ci_low"], ent["ci_high"]
    if a is None and b is None:
        return "Very wide" if ambiguous else None
    return f"{_fmt(a) if a is not None else '-infinity'} to {_fmt(b) if b is not None else '+infinity'}"


def _names(v) -> list[str]:
    """A parameter list given as a list or as 'A, B' text."""
    if v is None:
        return []
    if isinstance(v, str):
        return [t.strip() for t in v.split(",") if t.strip()]
    return [str(t) for t in v]


def nonlin_fit(table: Table, dataset_ids, params: dict) -> dict:
    ids = list(dataset_ids)
    known = {d.id for d in table.datasets}
    bad = [i for i in ids if i not in known]
    if bad or not ids:
        return _failed([f"unknown data set {i!r}" for i in bad] or ["no data set selected"])
    m = MODELS.get(params.get("model"))
    if m is None:
        return _failed([f"unknown model {params.get('model')!r}"])
    fit_to, weighting = params.get("fit_to", "replicates"), params.get("weighting", "none")
    level, method = float(params.get("ci_level", 0.95)), params.get("ci_method", "profile")
    cmp_names, alpha = _names(params.get("compare", "none")), float(params.get("compare_alpha", 0.05))
    if cmp_names == ["none"]:
        cmp_names = []
    if fit_to not in ("replicates", "mean") or weighting not in ("none", "1/Y^2", "1/SD^2") \
            or method not in ("asymptotic", "profile") or not 0 < level < 1 or not 0 < alpha < 1 \
            or params.get("interpolate_band", "confidence") not in ("confidence", "prediction") \
            or params.get("interpolate_replicates", "both") not in ("both", "mean", "each"):
        return _failed(["invalid fit option (fit_to, weighting, ci_method, ci_level, compare_alpha or interpolation option)"])
    wrong = [n for n in cmp_names if n not in m.params]
    if wrong:
        return _failed([f"cannot compare {wrong[0]!r}: the model has {', '.join(m.params)}"])
    share = _names(params.get("shared"))
    if share == ["none"]:
        share = []
    elif share:
        wrong = [n for n in share if n not in m.params]
        if wrong:
            return _failed([f"cannot share {wrong[0]!r}: the model has {', '.join(m.params)}"])
    elif len(ids) > 1:
        share = list(m.shared_default())
    if m.n_datasets is not None and len(ids) != m.n_datasets:
        return _failed([f"{m.name}: needs exactly {m.n_datasets} data sets (in the order of the equation's <A>, <B> lines), got {len(ids)}"])
    if len(ids) < m.min_datasets:
        return _failed([f"{m.name}: needs at least {m.min_datasets} data sets"])
    if weighting == "1/SD^2" and fit_to == "replicates" and table.y_format == "replicates":
        return _failed(["1/SD^2 weighting needs fit_to='mean' (fit the row means), or a mean/error table"])
    dsc = params.get("dataset_constants") or {}
    if not isinstance(dsc, dict) or any(k not in known or not isinstance(v, dict) or any(n not in m.params for n in v)
                                         for k, v in dsc.items()):
        return _failed(["dataset_constants must be {data set id: {parameter name: value}} for this model's parameters"])
    k = len(m.params)
    lo, hi, fixed = np.full(k, -np.inf), np.full(k, np.inf), np.full(k, np.nan)
    for n_, a, b in m.bounds:
        lo[m.params.index(n_)], hi[m.params.index(n_)] = a, b
    for n_, c in (params.get("constraints") or {}).items():
        if n_ not in m.params:
            return _failed([f"constraint on unknown parameter {n_!r}"])
        i = m.params.index(n_)
        if "fixed" in c:
            fixed[i] = float(c["fixed"])
        else:
            lo[i], hi[i] = float(c.get("min", -np.inf)), float(c.get("max", np.inf))
            if lo[i] >= hi[i]:
                return _failed([f"constraint on {n_}: min must be below max"])

    messages, notes = [], []
    if table.y_format != "replicates" and fit_to == "replicates":
        notes.append("mean/error table: the fit uses the row means")
    results, dflags, specs = {}, {}, []   # specs: data sets that can be fitted
    for pos, ds_id in enumerate(ids):
        title = table.dataset(ds_id).title
        try:
            x, y, sd, nobs = _prep(table, ds_id, fit_to, weighting)
        except ValueError as e:
            results[ds_id] = {"status": "failed", "params": {}, "derived": {}, "stats": {}, "flags": [str(e)],
                              "flag_codes": ["bad_input"], "curve": None}
            dflags[ds_id] = [("bad_input", str(e), False)]
            continue
        fx = fixed.copy()
        for n_, v_ in m.const_defaults:
            if np.isnan(fx[m.params.index(n_)]):
                fx[m.params.index(n_)] = v_
        if m.title_const and np.isnan(fixed[m.params.index(m.title_const)]) and _title_number(title) is not None:
            fx[m.params.index(m.title_const)] = _title_number(title)
        for n_, v_ in (dsc.get(ds_id) or {}).items():
            fx[m.params.index(n_)] = float(v_)
        func, used = m.variant(pos)
        used = np.array(used)
        missing = [n_ for n_ in m.constants if used[m.params.index(n_)] and np.isnan(fx[m.params.index(n_)])]
        if missing:
            why = (f"constant {missing[0]} is not set: enter a number in the data-set title" if m.title_const == missing[0]
                   else f"constant {', '.join(missing)} must be set (constraints = fixed value, or dataset_constants)")
            results[ds_id] = {"status": "failed", "params": {}, "derived": {}, "stats": {"n": int(len(x))}, "flags": [why],
                              "flag_codes": ["missing_constant"], "curve": None}
            dflags[ds_id] = [("missing_constant", why, False)]
            continue
        cvals = {n_: float(fx[i]) for i, n_ in enumerate(m.params) if not np.isnan(fx[i])}
        cvals["_ds"] = pos
        specs.append({"id": ds_id, "title": title, "x": x, "y": y, "sd": sd, "nobs": nobs, "func": func, "used": used,
                      "fixed": fx, "cvals": cvals})
    is_global = len(share) > 0 and len(specs) > 1
    if is_global and len(specs) < len(ids):
        for sp in specs:
            why = "global fit not run: another data set could not be prepared"
            results[sp["id"]] = {"status": "failed", "params": {}, "derived": {}, "stats": {"n": int(len(sp["x"]))},
                                 "flags": [why], "flag_codes": ["global_not_run"], "curve": None}
            dflags[sp["id"]] = [("global_not_run", why, False)]
        specs = []
    groups = [specs] if is_global else [[sp] for sp in specs]
    fits = [_fit_group(m, grp, tuple(share) if is_global else (), lo, hi, level, method, weighting) for grp in groups]
    want_bands = bool(params.get("bands"))
    do_interp = bool(params.get("interpolate"))
    rep_mode = params.get("interpolate_replicates", "both")
    interpolated, interpolated_rep, glob = {}, {}, None
    for g, grp in zip(fits, groups):
        g["weighting"] = weighting
        gl = len(grp) > 1
        for d, sp in enumerate(grp):
            fl = [(c, t, sc is None and gl) for c, t, sc in g["flags"] if sc is None or sc == d]
            if not g["ok"]:
                results[sp["id"]] = {"status": "failed", "params": {}, "derived": {}, "stats": {"n": int(len(sp["x"]))},
                                     "flags": [t for _, t, _ in fl], "flag_codes": [c for c, _, _ in fl], "curve": None}
                dflags[sp["id"]] = fl
                continue
            r = g["views"][d]
            pdict = {}
            for i, name in enumerate(m.params):
                if not sp["used"][i]:
                    continue
                if i in r["free"]:
                    pdict[name] = _entry(r["p"][i], r["se"][i], *r["ci"][i], dependency=float(r["dep"][i]),
                                         ambiguous=i in r["amb"], fixed=False, **({"shared": True} if i in r["shared_params"] else {}))
                else:
                    pdict[name] = _entry(r["p"][i], fixed=True)
            xs = _grid(m.x_scale, sp["x"])
            with np.errstate(all="ignore"):
                curve = {"x": xs.tolist(), "y": [float(v) for v in sp["func"](xs, r["p"])]}
            if want_bands:
                for key, kind in (("ci", "confidence"), ("pi", "prediction")):
                    bf = _band_fn(g, d, level, kind, weighting)
                    if bf is None:
                        continue
                    f_, h_ = bf(xs)
                    curve[f"{key}_low"], curve[f"{key}_high"] = [_fin(v) for v in f_ - h_], [_fin(v) for v in f_ + h_]
            results[sp["id"]] = {
                "status": "warning" if fl else "ok", "params": pdict, "derived": _derived(m, r, level),
                "stats": {"n": int(r["n"]), "df": int(g["df"]), "ss": float(r["ss"]), "sy_x": r["sy_x"], "r2": r["r2"],
                          "aicc": r["aicc"], "n_obs": sp["nobs"]},
                "flags": [t for _, t, _ in fl], "flag_codes": [c for c, _, _ in fl], "curve": curve,
            }
            dflags[sp["id"]] = fl
        if g["ok"] and gl:
            # Prism's overall R2 (round-2 comparison): total SS about the (weighted) mean of ALL data sets pooled;
            # the per-data-set R2 stays about each data set's own mean
            w_all, y_all = (np.concatenate(a) for a in zip(*(v["w_y"] for v in g["views"])))
            sst = float(np.sum(w_all * (y_all - np.sum(w_all * y_all) / np.sum(w_all)) ** 2))
            glob = {"shared": sorted({m.params[i] for v in g["views"] for i in v["shared_params"]}, key=m.params.index),
                    "n": int(g["n"]), "df": int(g["df"]), "ss": float(g["ss"]), "sy_x": g["sy_x"], "aicc": g["aicc"],
                    "r2": float(1 - g["ss"] / sst) if sst > 0 else None}
    if do_interp:
        for g, grp in zip(fits, groups):
            if not g["ok"]:
                continue
            for d, sp in enumerate(grp):
                interpolated[sp["id"]] = _interpolate(table, sp["id"], m, g, d, level, params, rep_mode == "each")
                if rep_mode == "both" and table.y_format == "replicates":  # Prism: mean table and replicate table
                    interpolated_rep[sp["id"]] = [e for e in _interpolate(table, sp["id"], m, g, d, level, params, True)
                                                  if e["direction"] == "x"]
                n_ex = sum(1 for e in interpolated[sp["id"]] if e["extrapolated"] and e["direction"] == "x")
                n_miss = sum(1 for e in interpolated[sp["id"]] if e["direction"] == "x" and e["x"] is None)
                for cnt, why in ((n_ex, "extrapolated beyond the standard curve (no confidence interval)"),
                                 (n_miss, "outside the range of the curve (no X value)")):
                    if cnt:
                        t = f"{cnt} interpolated value(s) {why}"
                        results[sp["id"]]["flags"].append(t)
                        results[sp["id"]]["flag_codes"].append("interpolation")
                        dflags[sp["id"]].append(("interpolation", t, False))
                        results[sp["id"]]["status"] = "warning"
    for ds_id in ids:
        title = table.dataset(ds_id).title
        for _, t, g_ in dflags.get(ds_id, []):
            msg = f"Global fit: {t}" if g_ else f"{title}: {t}"
            if msg not in messages:
                messages.append(msg)

    out = {"kind": "nonlin_fit", "model": m.id,
           "model_info": {"name": m.name, "params": list(m.params), "x_scale": m.x_scale, "family": m.family,
                          "constants": list(m.constants)}, "datasets": results}
    if glob is not None:
        out["global"] = glob
    if do_interp:
        out["interpolated"] = interpolated
        if interpolated_rep:
            out["interpolated_replicates"] = interpolated_rep
    if cmp_names:
        if specs and all(g["ok"] for g in fits) and len(specs) == len(ids):
            comp = _compare(m, specs, fits, cmp_names, share if is_global else (), lo, hi, alpha, weighting)
        else:
            comp = {"status": "skipped", "parameter": ", ".join(cmp_names), "parameters": cmp_names,
                    "reason": "a data set failed to fit"}
        out["comparison"] = comp
        if comp["status"] == "skipped":
            notes.append(f"comparison of {comp['parameter']} skipped: {comp['reason']}")
    messages = notes + messages
    statuses = [r["status"] for r in results.values()]
    out["status"] = "failed" if all(s == "failed" for s in statuses) else (
        "ok" if all(s == "ok" for s in statuses) and not notes else "warning")
    out["messages"] = messages
    out["display"] = _display(m, table, ids, results, out.get("comparison"), level, method, weighting, glob,
                              interpolated if do_interp else None, interpolated_rep)
    return out


def _display(m, table, ids, results, comp, level, method, weighting, glob=None, interp=None, interp_rep=None):
    ok_ids = [i for i in ids if results[i]["status"] != "failed"]
    cols = ["Parameter"] + (["Global (shared)"] if glob else []) + [table.dataset(i).title for i in ok_ids]
    der_names = list(dict.fromkeys(n for i in ok_ids for n in results[i]["derived"]))
    pct = f"{level * 100:g}%"

    def row(name, f):
        """One result row; shared entries appear once in the Global column."""
        ents = [results[i]["params"].get(name) or results[i]["derived"].get(name) for i in ok_ids]
        if not glob:
            return [name] + [f(e) for e in ents]
        first = next((e for e in ents if e is not None and e.get("shared")), None)
        return [name, None if first is None else f(first)] + [None if e is not None and e.get("shared") else f(e) for e in ents]

    best, se_rows, ci_rows = [], [], []
    for name in list(m.params) + der_names:
        ents = [results[i]["params"].get(name) or results[i]["derived"].get(name) for i in ok_ids]
        if not any(e is not None for e in ents):
            continue
        best.append(row(name, lambda e: None if e is None else e["value"]))
        if name in m.params or any(e is not None and "se" in e for e in ents):
            se_rows.append(row(name, lambda e: None if e is None else e.get("se")))
        ci_rows.append(row(name, lambda e: _ci_cell(e, bool(e and e.get("ambiguous")))))
    kind = "profile likelihood" if method == "profile" else "asymptotic, symmetrical"
    gl = (lambda v: [v]) if glob else (lambda v: [])
    st = lambda i, key: results[i]["stats"][key]  # noqa: E731
    gof = [["Degrees of Freedom"] + gl(glob["df"] if glob else None) + [st(i, "df") if not glob else None for i in ok_ids],
           ["R squared"] + gl(glob["r2"] if glob else None) + [st(i, "r2") for i in ok_ids],
           ["Sum of Squares" + (" (weighted)" if weighting != "none" else "")] + gl(glob["ss"] if glob else None) + [st(i, "ss") for i in ok_ids],
           ["Sy.x"] + gl(glob["sy_x"] if glob else None) + [st(i, "sy_x") if not glob else None for i in ok_ids],
           ["Ambiguous"] + gl(None) + ["Yes" if "ambiguous" in results[i]["flag_codes"] else "No" for i in ok_ids]]
    npts = [["Number of points analyzed"] + gl(glob["n"] if glob else None) + [st(i, "n") for i in ok_ids]]
    d = []
    if ok_ids:
        d = [{"title": "Best-fit values", "columns": cols, "rows": best},
             {"title": "Std. Error", "columns": cols, "rows": se_rows},
             {"title": f"{pct} Confidence Interval ({kind})", "columns": cols, "rows": ci_rows},
             {"title": "Goodness of Fit", "columns": cols, "rows": gof},
             {"title": "Number of points", "columns": cols, "rows": npts}]
    flag_rows = [[table.dataset(i).title, t] for i in ids for t in results[i]["flags"]]
    if flag_rows:
        d.append({"title": "Flags", "columns": ["Data set", "Flag"], "rows": flag_rows})
    if comp is not None and comp["status"] == "ok":
        p = comp["parameter"]
        d.append({"title": f"Comparison of fits: {p}", "columns": ["", "Null hypothesis", "Alternative hypothesis"], "rows": [
            ["Hypothesis", comp["null"]["label"], comp["alt"]["label"]],
            ["Sum of squares", comp["null"]["ss"], comp["alt"]["ss"]],
            ["Degrees of freedom", comp["null"]["df"], comp["alt"]["df"]],
            ["AICc", comp["null"]["aicc"], comp["alt"]["aicc"]],
            ["Probability that the model is correct", comp["prob_shared"], comp["prob_separate"]]]})
        sv = comp["shared_value"]
        d.append({"title": f"Extra sum-of-squares F test: {p}", "columns": ["Quantity", "Value"], "rows": [
            ["F", comp["f"]], ["DFn", comp["dfn"]], ["DFd", comp["dfd"]], ["P value", comp["p"]],
            [f"Preferred model (alpha = {comp['alpha']:g})", comp["preferred"]],
            ["Shared value under the null hypothesis", sv if not isinstance(sv, dict) else "; ".join(f"{k} = {v:.4g}" for k, v in sv.items())],
            ["AICc difference (null - alternative)", comp["aicc_diff"]],
            ["Preferred model by AICc", comp["aicc_preferred"]]]})
    if interp:
        rows = []
        for ds_id, lst in interp.items():
            for e in lst:
                note = ("extrapolated" if e["extrapolated"] else "") if e["x"] is not None else "outside the curve"
                if e["direction"] == "x":
                    rows.append([table.dataset(ds_id).title, e["row"] + 1, e["y"], e["x"], e["x_ci_low"], e["x_ci_high"], note])
                else:
                    rows.append([table.dataset(ds_id).title, e["row"] + 1, e["y"], e["x"], e["y_ci_low"], e["y_ci_high"],
                                 "Y computed from X" + ("; extrapolated" if e["extrapolated"] else "")])
        if rows:
            d.append({"title": "Interpolated values", "columns": ["Data set", "Row", "Y", "X", "Lower limit", "Upper limit", "Note"],
                      "rows": rows})
    rows = [[table.dataset(i).title, e["row"] + 1, e["y"], e["x"], e["x_ci_low"], e["x_ci_high"],
             ("extrapolated" if e["extrapolated"] else "") if e["x"] is not None else "outside the curve"]
            for i, lst in (interp_rep or {}).items() for e in lst]
    if rows:
        d.append({"title": "Interpolated values (each replicate)",
                  "columns": ["Data set", "Row", "Y", "X", "Lower limit", "Upper limit", "Note"], "rows": rows})
    return d


def _choice_label(m: Model) -> str:
    return m.name if m.family == "Dose-response" else f"{m.family}: {m.name}"


KINDS = {
    "nonlin_fit": AnalysisKind(
        "nonlin_fit", "Nonlinear regression (curve fit)", ("XY",), nonlin_fit,
        (Param("model", "Model", "choice", "log_agonist_variable", tuple((m.id, _choice_label(m)) for m in ALL)),
         Param("fit_to", "Fit to", "choice", "replicates",
               (("replicates", "Each replicate as an individual point"), ("mean", "Row means only"))),
         Param("weighting", "Weighting", "choice", "none",
               (("none", "No weighting"), ("1/Y^2", "Relative weighting (1/Y^2)"), ("1/SD^2", "Weight by 1/SD^2 (row means)"))),
         Param("constraints", "Constraints", "constraints", {}),
         Param("shared", "Parameters shared by all data sets (comma separated; empty = the model's default, 'none' = fit separately)", "str", ""),
         Param("ci_level", "Confidence level", "float", 0.95),
         Param("ci_method", "Confidence interval method", "choice", "profile",
               (("profile", "Asymmetrical (profile likelihood)"), ("asymptotic", "Symmetrical (asymptotic)"))),
         Param("compare", "Compare data sets: parameter(s) (comma separated names, or 'none')", "str", "none"),
         Param("compare_alpha", "Significance level for the comparison", "float", 0.05),
         Param("interpolate", "Interpolate unknowns (rows with Y but no X; rows with X but no Y give Y)", "bool", False),
         Param("bands", "Confidence and prediction bands of the curve", "bool", False))),
}
