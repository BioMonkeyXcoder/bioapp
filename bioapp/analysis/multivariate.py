"""Multivariate analyses on MultipleVariables tables: PCA, hierarchical clustering, K-means.

Prism conventions used here (checked against Prism's Wine and Cities sample results):
  * "standardize" = (x - mean) / SD with the n-1 SD (total within-cluster SS of standardized data is (n-1)*p);
  * PCA eigen-decomposes the correlation matrix (standardized data) or the covariance matrix (centered data);
    loadings are eigenvector * sqrt(eigenvalue) (variable-component correlations when standardized);
    scores are the transformed data times the eigenvectors. The sign of a component is arbitrary;
    here the largest absolute loading of each component is made positive;
  * parallel analysis: eigenvalues of random normal data of the same size, 1000 simulations, the 95th percentile
    is the retention threshold, components are kept while their eigenvalue exceeds it (the first failure stops);
  * hierarchical clustering: scipy linkage on the chosen distance (merge heights and the cophenetic correlation match
    Prism for average linkage), clusters from a cut by number or height, numbered by dendrogram leaf order;
    elbow = within-cluster SS of the transformed data for 1..n-1 clusters;
  * K-means: Lloyd iterations from k-means++ starts with several restarts and a fixed default seed (reproducible; the
    algorithm is named in the result and the display). Prism runs Hartigan-Wong from one random start and stops at a
    poorer local optimum for k >= 3, so its WCSS is equal or higher: not imitated. Silhouette score, gap statistic
    (Tibshirani, uniform reference data aligned with the principal components) and a per-variable ANOVA table.
Every function returns the result contract of DESIGN.md; invalid input gives status "failed" with a reason.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import cophenet, cut_tree, leaves_list, linkage
from scipy.spatial.distance import cdist, pdist, squareform

from . import AnalysisKind, Param
from .regression import _kind, _tbl
from .stats import _Bad

_CH = lambda *a: tuple((v, l) for v, l in a)  # noqa: E731
_TRANSFORMS = _CH(("none", "None"), ("center", "Center"), ("standardize", "Standardize (mean 0, SD 1)"),
                  ("normalize", "Normalize (0 to 1)"))


# --------------------------------------------------------------------------- data

def _matrix(table, ids):
    """Complete rows of the chosen numeric variables -> (matrix, variable titles, source rows, row labels, messages)."""
    msgs = []
    if table.kind == "Grouped":
        cols = [table.summary(i)["mean"] for i in ids]
        titles = [table.dataset(i).title for i in ids]
    else:
        use = [i for i in ids if table.dataset(i).categories is None]
        if len(use) < len(ids):
            msgs.append("categorical variables were not used")
        ids, cols = use, [np.where(table.dataset(i).excluded[:, 0], np.nan, table.dataset(i).values[:, 0]) for i in use]
        titles = [table.dataset(i).title for i in use]
    if len(cols) < 2:
        raise _Bad("select at least 2 numeric variables")
    M = np.column_stack(cols)
    keep = ~np.isnan(M).any(axis=1)
    if keep.sum() < 3:
        raise _Bad(f"only {int(keep.sum())} complete row(s); at least 3 are needed")
    if (~keep).any():
        msgs.append(f"{int((~keep).sum())} row(s) with blank or excluded values were skipped")
    rows = np.flatnonzero(keep)
    rt = table.row_titles
    labels = [rt[r] if len(rt) == table.n_rows and rt[r] else str(r + 1) for r in rows]
    return M[keep], titles, rows, labels, msgs


def _transform(M, how, axis=0):
    """Transform along ``axis`` (0: each variable across rows, 1: each row across variables)."""
    if how == "none":
        return M
    mean = M.mean(axis=axis, keepdims=True)
    if how == "center":
        return M - mean
    if how == "standardize":
        sd = M.std(axis=axis, ddof=1, keepdims=True)
        if np.any(sd == 0):
            raise _Bad("a variable (or row) is constant, so it cannot be standardized")
        return (M - mean) / sd
    if how == "normalize":
        rng = M.max(axis=axis, keepdims=True) - M.min(axis=axis, keepdims=True)
        if np.any(rng == 0):
            raise _Bad("a variable (or row) is constant, so it cannot be normalized")
        return (M - M.min(axis=axis, keepdims=True)) / rng
    raise _Bad(f"unknown transform {how!r}")


# --------------------------------------------------------------------------- PCA

def _parallel(n, p, sd, corr, sims, seed):
    """Eigenvalues (descending) of random normal data with the same n, p and column SDs: sims x p.\n    ``corr``: eigenvalues of the correlation matrix (standardized data), else of the covariance matrix."""
    rng = np.random.default_rng(seed)
    out = np.empty((sims, p))
    for s in range(sims):
        R = rng.standard_normal((n, p)) * sd
        out[s] = np.linalg.eigvalsh(np.corrcoef(R, rowvar=False) if corr else np.cov(R, rowvar=False))[::-1]
    return out


@_kind("pca")
def pca(table, ids, params):
    M, titles, rows, labels, msgs = _matrix(table, ids)
    n, p = M.shape
    how = params.get("transform", "standardize")
    method = params.get("method", "parallel")
    if how == "normalize":
        raise _Bad("PCA works on centered or standardized data; use 'center' or 'standardize'")
    if how == "none":
        raise _Bad("PCA needs centered or standardized data")
    Z = _transform(M, how)
    cov = np.cov(Z, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vals, vecs = np.clip(vals[::-1], 0, None), vecs[:, ::-1]
    m = min(n - 1, p)
    vals, vecs = vals[:m], vecs[:, :m]
    for j in range(m):  # deterministic sign
        if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
            vecs[:, j] = -vecs[:, j]
    total = float(np.trace(cov))
    expl = vals / total * 100
    cum = np.cumsum(expl)
    par = None
    if method == "parallel":
        sims = int(params.get("simulations", 1000))
        pct = float(params.get("percentile", 95))
        if sims < 50 or not 50 <= pct < 100:
            raise _Bad("parallel analysis needs at least 50 simulations and a percentile in [50, 100)")
        S = _parallel(n, p, Z.std(axis=0, ddof=1), how == "standardize", sims, int(params.get("seed", 12345)))[:, :m]
        thr = np.percentile(S, pct, axis=0)
        par = {"percentile": pct, "simulations": sims, "threshold": thr, "median": np.median(S, axis=0)}
        keep = 0
        while keep < m and vals[keep] > thr[keep]:
            keep += 1
    elif method == "kaiser":
        t = float(params.get("eigenvalue_threshold", 1.0))
        keep = int(np.sum(vals > t))
    elif method == "variance":
        v = float(params.get("variance_pct", 75.0))
        if not 0 < v <= 100:
            raise _Bad("explained variance must be in (0, 100]")
        keep = int(np.searchsorted(cum, v - 1e-9) + 1)
    elif method == "fixed":
        keep = int(params.get("n_components", 2))
    else:
        raise _Bad(f"unknown component selection method {method!r}")
    if method != "fixed" and keep < 1:
        msgs.append("the selection rule kept no component; the first one is reported")
    keep = int(min(max(keep, 1), m))
    if method == "fixed" and keep != int(params.get("n_components", 2)):
        msgs.append(f"only {m} components exist; {keep} reported")
    shown = max(keep, min(2, m))  # the PC1-vs-PC2 graphs need two components even if the rule keeps one
    scores = Z @ vecs[:, :shown]
    loadings = vecs[:, :shown] * np.sqrt(vals[:shown])
    pcs = [f"PC{j + 1}" for j in range(shown)]
    out = {"pca": {"variables": titles, "explained": expl, "eigenvalues": vals, "cumulative": cum,
                   "loadings": loadings, "eigenvectors": vecs[:, :shown], "scores": scores, "obs_labels": labels,
                   "pc_names": pcs, "n_selected": keep, "method": method, "transform": how, "parallel": par},
           "rows": rows, "messages": msgs}
    ev_rows = [[f"PC{j + 1}", vals[j], expl[j], cum[j]] + ([par["threshold"][j]] if par else []) for j in range(m)]
    out["display"] = [
        _tbl("Eigenvalues", ["Component", "Eigenvalue", "% variance", "Cumulative %"] + ([f"Parallel analysis {par['percentile']:g}th percentile"] if par else []), ev_rows),
        _tbl(f"Loadings (variable-component correlations), {keep} component(s) kept" + (f", {shown} reported" if shown > keep else ""), ["Variable"] + pcs,
             [[titles[i]] + loadings[i].tolist() for i in range(p)]),
        _tbl("Component scores", ["Row"] + pcs, [[labels[i]] + scores[i].tolist() for i in range(n)]),
        _tbl("Data summary", ["Quantity", "Value"], [["Variables", p], ["Rows analyzed", n], ["Components selected", keep],
                                                       ["Components reported (at least 2 for the PC plots)", shown],
                                                       ["Selection method", method]])]
    return out


# --------------------------------------------------------------------------- hierarchical clustering

def _distances(Z, how, power):
    if how == "euclidean":
        return pdist(Z)
    if how == "manhattan":
        return pdist(Z, "cityblock")
    if how == "minkowski":
        return pdist(Z, "minkowski", p=power)
    if how == "chebyshev":
        return pdist(Z, "chebyshev")
    if how in ("pearson", "spearman"):
        if Z.shape[1] < 3:
            raise _Bad("correlation distance needs at least 3 values per item")
        A = np.apply_along_axis(stats.rankdata, 1, Z) if how == "spearman" else Z
        if np.any(A.std(axis=1) == 0):
            raise _Bad("an item is constant, so its correlation distance is undefined")
        return pdist(A, "correlation")
    raise _Bad(f"unknown distance {how!r}")


def _wcss(Z, labels):
    return float(sum(np.sum((Z[labels == c] - Z[labels == c].mean(0)) ** 2) for c in np.unique(labels)))


def _hclust_axis(Z, labels, how, link, power, method, count, cutoff):
    n = len(Z)
    if n < 3:
        raise _Bad("need at least 3 items to cluster")
    if link in ("ward", "centroid", "median") and how != "euclidean":
        raise _Bad(f"{link} linkage requires Euclidean distance")
    d = _distances(Z, how, power)
    Lk = linkage(d, link)
    coph = cophenet(Lk, d)[0]
    order = leaves_list(Lk)
    cl = None
    if method == "count":
        if not 1 <= count <= n:
            raise _Bad(f"number of clusters must be between 1 and {n}")
        raw = cut_tree(Lk, n_clusters=count).ravel()
    elif method == "cutoff":
        raw = cut_tree(Lk, height=cutoff).ravel()
    elif method == "none":
        raw = None
    else:
        raise _Bad(f"unknown cluster definition {method!r}")
    if raw is not None:  # number clusters by first appearance along the dendrogram
        new = {}
        for i in order:
            new.setdefault(int(raw[i]), len(new) + 1)
        cl = [new[int(c)] for c in raw]
    elbow = {"k": list(range(1, n)), "wcss": [_wcss(Z, cut_tree(Lk, n_clusters=k).ravel()) for k in range(1, n)]}
    return {"linkage": Lk, "labels": labels, "order": order, "cophenetic_correlation": float(coph),
            "elbow": elbow, "dist": d, "coph": cophenet(Lk)}, cl


@_kind("hierarchical_clustering")
def hierarchical_clustering(table, ids, params):
    M, titles, rows, labels, msgs = _matrix(table, ids)
    Z = _transform(_transform(M, params.get("column_transform", "standardize"), 0), params.get("row_transform", "none"), 1)
    which = params.get("cluster", "rows")
    how, link, power = params.get("distance", "euclidean"), params.get("linkage", "average"), float(params.get("minkowski_power", 3.0))
    if how == "minkowski" and power < 1:
        raise _Bad("the Minkowski power must be at least 1")
    res = {}
    for ax, items, lab in (("rows", Z, labels), ("cols", Z.T, titles)):
        if (ax == "rows" and which == "columns") or (ax == "cols" and which == "rows"):
            res[ax] = (None, None)
            continue
        method = params.get(f"{ax}_method", "none")
        res[ax] = _hclust_axis(items, lab, how, link, power, method, int(params.get(f"{ax}_count", 2)),
                               float(params.get(f"{ax}_cutoff", 1.0)))
    out = {"dendrogram": {}, "matrix": {"values": Z, "row_labels": labels, "col_labels": titles},
           "clusters": {}, "rows": rows, "messages": msgs, "display": [], "elbow": {}, "cophenetic": {}}
    for ax in ("rows", "cols"):
        g, cl = res[ax]
        out["clusters"][ax] = cl
        if g is None:
            out["dendrogram"][ax] = None
            continue
        out["dendrogram"][ax] = {"linkage": g["linkage"], "labels": g["labels"], "order": g["order"]}
        out["elbow"][ax] = g["elbow"]
        out["cophenetic"][ax] = g["cophenetic_correlation"]
        if params.get("distance_matrix"):
            out.setdefault("matrices", {})[ax] = {"distance": squareform(g["dist"]), "cophenetic": squareform(g["coph"])}
        nm = len(g["labels"])
        merges = []
        for s, (a, b, h, c) in enumerate(g["linkage"]):
            name = lambda v: g["labels"][int(v)] if v < nm else f"Cluster {int(v) - nm + 1}"  # noqa: E731
            merges.append([s + 1, name(a), name(b), float(h), int(c)])
        title = "rows" if ax == "rows" else "columns"
        out["display"] += [
            _tbl(f"Merge map ({title})", ["Step", "Item 1", "Item 2", "Distance", "Items in new cluster"], merges),
            _tbl(f"Clustering fit ({title})", ["Statistic", "Value"], [["Cophenetic correlation", g["cophenetic_correlation"]]]),
            _tbl(f"Elbow ({title})", ["Clusters", "Within-cluster SS"], [[k, w] for k, w in zip(g["elbow"]["k"], g["elbow"]["wcss"])])]
        if cl is not None:
            out["display"].append(_tbl(f"Clusters ({title})", ["Item", "Cluster"], [[g["labels"][i], cl[i]] for i in g["order"]]))
    return out


# --------------------------------------------------------------------------- K-means

def _lloyd(Z, k, rng, max_iter):
    n = len(Z)
    idx = [int(rng.integers(n))]
    d2 = np.sum((Z - Z[idx[0]]) ** 2, axis=1)
    for _ in range(1, k):  # k-means++ seeding
        s = d2.sum()
        j = int(rng.choice(n, p=d2 / s)) if s > 0 else int(rng.integers(n))
        idx.append(j)
        d2 = np.minimum(d2, np.sum((Z - Z[j]) ** 2, axis=1))
    C = Z[idx].copy()
    zz, lab = np.sum(Z ** 2, axis=1), None
    for _ in range(max_iter):
        D = np.clip(zz[:, None] - 2 * Z @ C.T + np.sum(C ** 2, axis=1)[None], 0, None)
        new = D.argmin(axis=1)
        if lab is not None and np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            m = lab == j
            C[j] = Z[m].mean(0) if m.any() else Z[D.min(axis=1).argmax()]
    return lab, _wcss(Z, lab)


def _kmeans(Z, k, rng, n_init, max_iter):
    if k == 1:
        return np.zeros(len(Z), int), _wcss(Z, np.zeros(len(Z), int))
    best = min((_lloyd(Z, k, rng, max_iter) for _ in range(n_init)), key=lambda r: r[1])
    return best


def _silhouette(Z, lab):
    ks = np.unique(lab)
    if len(ks) < 2:
        return 0.0
    onehot = (lab[:, None] == ks[None]).astype(float)
    cnt = onehot.sum(0)
    s = np.zeros(len(Z))
    for a in range(0, len(Z), 500):
        sl = slice(a, a + 500)
        sums = cdist(Z[sl], Z) @ onehot
        own = np.searchsorted(ks, lab[sl])
        r = np.arange(len(own))
        with np.errstate(divide="ignore", invalid="ignore"):
            ai = sums[r, own] / (cnt[own] - 1)
            mean_other = sums / cnt
            mean_other[r, own] = np.inf
            bi = mean_other.min(axis=1)
            si = (bi - ai) / np.maximum(ai, bi)
        s[sl] = np.where(cnt[own] > 1, np.nan_to_num(si), 0.0)
    return float(s.mean())


def _anova(Z, lab, k):
    n, p = Z.shape
    gm = Z.mean(0)
    ssb, ssw = np.zeros(p), np.zeros(p)
    for c in range(k):
        Zc = Z[lab == c]
        ssb += len(Zc) * (Zc.mean(0) - gm) ** 2
        ssw += np.sum((Zc - Zc.mean(0)) ** 2, axis=0)
    out = []
    for j in range(p):
        f = (ssb[j] / (k - 1)) / (ssw[j] / (n - k)) if ssw[j] > 0 else math.inf
        out.append({"ss_between": float(ssb[j]), "df_between": k - 1, "ms_between": float(ssb[j] / (k - 1)),
                    "ss_within": float(ssw[j]), "df_within": n - k, "f": float(f),
                    "p": float(stats.f.sf(f, k - 1, n - k)) if np.isfinite(f) else 0.0})
    return out


def _partition(Z, lab, k):
    """Everything reported for one partition of the (transformed) data."""
    cent = np.array([Z[lab == c].mean(0) for c in range(k)])
    return {"wcss": _wcss(Z, lab), "silhouette": _silhouette(Z, lab), "centroids": cent,
            "sizes": [int(np.sum(lab == c)) for c in range(k)],
            "wcss_by_cluster": [float(np.sum((Z[lab == c] - cent[c]) ** 2)) for c in range(k)],
            "distances": np.linalg.norm(Z - cent[lab], axis=1),
            "anova": _anova(Z, lab, k) if k > 1 else None}


def _gap(Z, ks, rng, B, n_init, max_iter, logw):
    """Tibshirani gap statistic with uniform reference data aligned with the principal components."""
    Zc = Z - Z.mean(0)
    P = Zc @ np.linalg.svd(Zc, full_matrices=False)[2].T
    lo, hi = P.min(0), P.max(0)
    ref = {k: [] for k in ks}
    for _ in range(B):
        R = rng.uniform(lo, hi, P.shape)
        for k in ks:
            ref[k].append(math.log(max(_kmeans(R, k, rng, n_init, max_iter)[1], 1e-300)))
    out = {}
    for k in ks:
        r = np.array(ref[k])
        out[k] = {"statistic": float(r.mean() - logw[k]), "sd": float(r.std() * math.sqrt(1 + 1 / B)),
                  "log_w_observed": float(logw[k]), "log_w_expected": float(r.mean())}
    return out


@_kind("kmeans")
def kmeans(table, ids, params):
    M, titles, rows, labels, msgs = _matrix(table, ids)
    n, p = M.shape
    Z = _transform(M, params.get("transform", "standardize"))
    kmin, kmax = int(params.get("k_min", 2)), int(params.get("k_max", 6))
    if kmin < 1 or kmax < kmin:
        raise _Bad("cluster counts must satisfy 1 <= minimum <= maximum")
    if kmax > n - 1:
        msgs.append(f"maximum clusters reduced to {n - 1} (rows - 1)")
        kmax = n - 1
    if kmax < kmin:
        raise _Bad(f"{n} rows are too few for {kmin} clusters")
    n_init, max_iter = int(params.get("restarts", 25)), int(params.get("max_iter", 300))
    if n_init < 1 or max_iter < 1:
        raise _Bad("restarts and maximum iterations must be at least 1")
    B = int(params.get("gap_samples", 20))
    rng = np.random.default_rng(int(params.get("seed", 1234)))  # fixed default seed: results are reproducible
    ks = list(range(kmin, kmax + 1))
    tss = float(np.sum((Z - Z.mean(0)) ** 2))
    models, summary = {}, {"k": ks, "wcss": [], "pct_total": [], "pct_explained": [], "silhouette": [], "gap": [], "gap_sd": []}
    for k in ks:
        lab, _ = _kmeans(Z, k, rng, n_init, max_iter)
        order = []  # clusters numbered by first appearance
        for c in lab:
            if c not in order:
                order.append(int(c))
        lab = np.array([order.index(int(c)) for c in lab])
        m = _partition(Z, lab, k)
        m["labels"] = lab + 1
        m["centroids_original"] = np.array([M[lab == c].mean(0) for c in range(k)])
        models[str(k)] = m
        summary["wcss"].append(m["wcss"]), summary["silhouette"].append(m["silhouette"])
        summary["pct_total"].append(m["wcss"] / tss * 100), summary["pct_explained"].append(100 - m["wcss"] / tss * 100)
    suggested = {"silhouette": ks[int(np.argmax(summary["silhouette"]))] if len(ks) > 1 else None, "gap": None}
    if B >= 2:
        gp = _gap(Z, ks, rng, B, 3, max_iter, {k: math.log(max(models[str(k)]["wcss"], 1e-300)) for k in ks})
        summary["gap"] = [gp[k]["statistic"] for k in ks]
        summary["gap_sd"] = [gp[k]["sd"] for k in ks]
        for k in ks:
            models[str(k)]["gap"] = gp[k]
        for k in ks:  # smallest k with Gap(k) >= Gap(k+1) - s(k+1)
            if k + 1 in gp and gp[k]["statistic"] >= gp[k + 1]["statistic"] - gp[k + 1]["sd"]:
                suggested["gap"] = k
                break
    else:
        summary["gap"] = summary["gap_sd"] = [None] * len(ks)
    seed = int(params.get("seed", 1234))
    algo = (f"Lloyd iterations from k-means++ starts, best of {n_init} restarts per k (random seed {seed}; the same seed "
            "always gives the same result). Prism uses Hartigan-Wong from one random start, so for k >= 3 it can stop at a "
            "higher WCSS.")
    out = {"kmeans": {"variables": titles, "obs_labels": labels, "transform": params.get("transform", "standardize"),
                      "total_ss": tss, "summary": summary, "models": models, "suggested": suggested, "algorithm": algo, "seed": seed},
           "rows": rows, "messages": msgs}
    disp = [_tbl("Clustering fit metrics", ["Clusters", "WCSS", "% of total variation", "% explained", "Silhouette", "Gap", "Gap SD"],
                 [[k, summary["wcss"][i], summary["pct_total"][i], summary["pct_explained"][i], summary["silhouette"][i],
                   summary["gap"][i], summary["gap_sd"][i]] for i, k in enumerate(ks)]),
            _tbl("Algorithm", ["Setting", "Value"], [["Method", algo], ["Random seed", seed], ["Restarts per k", n_init],
                                                    ["Maximum iterations", max_iter]])]
    for k in ks:
        m = models[str(k)]
        disp.append(_tbl(f"{k} clusters: details", ["Cluster", "Items", "WCSS"] + [f"Mean {t}" for t in titles],
                         [[c + 1, m["sizes"][c], m["wcss_by_cluster"][c]] + m["centroids_original"][c].tolist() for c in range(k)]))
        if m["anova"]:
            disp.append(_tbl(f"{k} clusters: ANOVA by variable", ["Variable", "SS between", "DF", "MS", "SS within", "DFd", "F", "P value"],
                             [[titles[j], a["ss_between"], a["df_between"], a["ms_between"], a["ss_within"], a["df_within"], a["f"], a["p"]]
                              for j, a in enumerate(m["anova"])]))
    out["display"] = disp
    return out


# --------------------------------------------------------------------------- registry

_MV = ("MultipleVariables",)
KINDS = {
    "pca": AnalysisKind(
        "pca", "Principal component analysis", _MV, pca,
        (Param("transform", "Data transform", "choice", "standardize", _CH(("standardize", "Standardize"), ("center", "Center only"))),
         Param("method", "Select components by", "choice", "parallel",
               _CH(("parallel", "Parallel analysis"), ("kaiser", "Eigenvalue threshold"), ("variance", "Cumulative % variance"), ("fixed", "Fixed number"))),
         Param("eigenvalue_threshold", "Eigenvalue threshold", "float", 1.0), Param("variance_pct", "Cumulative % variance", "float", 75.0),
         Param("n_components", "Number of components", "int", 2), Param("percentile", "Parallel analysis percentile", "float", 95.0),
         Param("simulations", "Parallel analysis simulations", "int", 1000), Param("seed", "Random seed", "int", 12345))),
    "hierarchical_clustering": AnalysisKind(
        "hierarchical_clustering", "Hierarchical clustering", ("MultipleVariables", "Grouped"), hierarchical_clustering,
        (Param("cluster", "Cluster", "choice", "rows", _CH(("rows", "Rows"), ("columns", "Columns"), ("both", "Rows and columns"))),
         Param("column_transform", "Transform each variable", "choice", "standardize", _TRANSFORMS),
         Param("row_transform", "Transform each row", "choice", "none", _TRANSFORMS),
         Param("distance", "Distance", "choice", "euclidean",
               _CH(("euclidean", "Euclidean"), ("manhattan", "Manhattan"), ("minkowski", "Minkowski"), ("chebyshev", "Chebyshev"),
                   ("pearson", "1 - Pearson r"), ("spearman", "1 - Spearman r"))),
         Param("minkowski_power", "Minkowski power", "float", 3.0),
         Param("linkage", "Linkage", "choice", "average",
               _CH(("average", "Average (UPGMA)"), ("complete", "Complete"), ("single", "Single"), ("weighted", "Weighted (WPGMA)"),
                   ("ward", "Ward"), ("centroid", "Centroid"), ("median", "Median"))),
         Param("rows_method", "Define row clusters by", "choice", "none", _CH(("none", "None"), ("count", "Number of clusters"), ("cutoff", "Cutoff height"))),
         Param("rows_count", "Row clusters", "int", 2), Param("rows_cutoff", "Row cutoff height", "float", 1.0),
         Param("cols_method", "Define column clusters by", "choice", "none", _CH(("none", "None"), ("count", "Number of clusters"), ("cutoff", "Cutoff height"))),
         Param("cols_count", "Column clusters", "int", 2), Param("cols_cutoff", "Column cutoff height", "float", 1.0),
         Param("distance_matrix", "Also return distance and cophenetic matrices", "bool", False))),
    "kmeans": AnalysisKind(
        "kmeans", "K-means clustering", _MV, kmeans,
        (Param("transform", "Data transform", "choice", "standardize", _TRANSFORMS),
         Param("k_min", "Minimum clusters", "int", 2), Param("k_max", "Maximum clusters", "int", 6),
         Param("restarts", "Random starts per k", "int", 25), Param("max_iter", "Maximum iterations", "int", 300),
         Param("gap_samples", "Gap statistic reference data sets (0 = off)", "int", 20), Param("seed", "Random seed", "int", 1234))),
}


