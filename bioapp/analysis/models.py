"""Built-in model library (Prism's Curve Fitting Guide "Equation: ..." pages) for ``fit.py``.

A ``Model`` declares its parameters (in Prism's names), the equation ``func(x, p)``, default constraints,
the parameters that are experimental *constants* (held fixed by the user, e.g. HotNM; one of them may come
from the data-set title, e.g. the inhibitor concentration I), per-data-set equations (``variants``: Prism's
``<A>`` / ``<~A>`` lines), the parameters Prism shares by default in a global fit (``shared``), derived
results and a starting-value rule (``init``).

Equations were checked against the model files shipped with Prism 10.4.1 (``Equations\\*.pzt``). Prism's
own starting-value rules are stored in a binary format that is not read here: ``init`` rules are the usual
"plateaus from the data, potency at half response, rates from the half time" estimates (they only affect
convergence, never the converged fit).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import numpy as np

LN2 = float(np.log(2.0))
LOG10E = float(np.log10(np.e))
INF = np.inf


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    params: tuple[str, ...]
    func: Callable                      # func(x, p) with p in `params` order
    x_scale: str                        # "log" (X is log10 dose), "dose" or "linear"
    direction: int = 1                  # +1 agonist (rising), -1 inhibitor; sign of the starting HillSlope
    hill: str = "free"                  # "free" | "fixed" (standard slope: no HillSlope parameter)
    bounds: tuple = ()                  # default constraints ((param, lo, hi), ...)
    family: str = "Dose-response"
    constants: tuple[str, ...] = ()     # parameters that must be constant (experimental constants)
    const_defaults: tuple = ()          # ((name, value), ...) used when the user gives no value
    title_const: str | None = None      # constant read from each data-set title (a number)
    variants: tuple = ()                # ((func, used_param_names | None), ...): data set i uses [min(i, last)]
    n_datasets: int | None = None       # exact number of data sets the model needs
    min_datasets: int = 1               # minimum number of data sets
    shared: tuple | str = ()            # shared by default in multi-data-set fits ("all" = every non-constant)
    init: Callable | None = None        # init(summary, constants) -> {param: value}
    derived: tuple = ()                 # ("t", name, source_param, fn) | ("d", name, fn(dict of params))
    se_scale: tuple = ()                # ((param, factor), ...): Prism's reported SE of a parameter = factor * the Jacobian SE

    @property
    def potency(self) -> str | None:
        return next((n for n in self.params if "EC50" in n or "IC50" in n), None)

    def variant(self, i: int) -> tuple[Callable, tuple[bool, ...]]:
        """Equation and used-parameter mask of the i-th selected data set."""
        if not self.variants:
            return self.func, (True,) * len(self.params)
        f, used = self.variants[min(i, len(self.variants) - 1)]
        return f, tuple(True if used is None else n in used for n in self.params)

    def shared_default(self) -> tuple[str, ...]:
        if self.shared == "all":
            return tuple(n for n in self.params if n not in self.constants)
        return tuple(self.shared)


# ---------------------------------------------------------------- data summary for starting values

def summary(scale: str, x, y) -> SimpleNamespace:
    """Unique X, mean Y per X, trend direction, plateaus and the X where the response is half way."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xs = np.unique(x)
    ym = np.array([y[x == v].mean() for v in xs])
    n = len(ym)
    d = 1.0 if n < 2 or ym[n - n // 2:].mean() >= ym[:n // 2].mean() else -1.0
    lo_y, hi_y = float(ym.min()), float(ym.max())
    mid = (lo_y + hi_y) / 2
    half = float(np.median(xs))
    for i in range(n - 1):
        if (ym[i] - mid) * (ym[i + 1] - mid) <= 0 and ym[i] != ym[i + 1]:
            fr = (mid - ym[i]) / (ym[i + 1] - ym[i])
            if scale == "dose" and xs[i] > 0:
                half = float(xs[i] * (xs[i + 1] / xs[i]) ** fr)
            else:
                half = float(xs[i] + fr * (xs[i + 1] - xs[i]))
            break
    if scale == "dose" and half <= 0:
        half = float(np.median(xs[xs > 0])) if np.any(xs > 0) else 1.0
    return SimpleNamespace(x=x, y=y, xs=xs, ym=ym, n=n, d=d, lo=lo_y, hi=hi_y, half=half, scale=scale,
                           xmin=float(xs[0]), xmax=float(xs[-1]), R=float(xs[-1] - xs[0]) or 1.0)


def _at(s, frac):
    """X at a fraction of the X range (geometric on a dose scale)."""
    if s.scale == "dose" and s.xmin > 0:
        return float(s.xmin * (s.xmax / s.xmin) ** frac)
    return float(s.xmin + frac * s.R)


def _rate(s, y0, yinf):
    """Rate constant from the time to go half way from y0 to yinf (first/last mean)."""
    mid = (y0 + yinf) / 2
    t = None
    for i in range(s.n - 1):
        if (s.ym[i] - mid) * (s.ym[i + 1] - mid) <= 0 and s.ym[i] != s.ym[i + 1]:
            t = s.xs[i] + (mid - s.ym[i]) / (s.ym[i + 1] - s.ym[i]) * (s.xs[i + 1] - s.xs[i]) - s.xs[0]
            break
    return LN2 / max(t if t is not None else 0.3 * s.R, 0.02 * s.R, 1e-300)


def _legacy_init(m, s, c):
    """Starting values of the original dose-response models and the straight line."""
    if m.id == "line":
        slope, icpt = np.polyfit(s.x, s.y, 1) if s.n > 1 else (0.0, float(s.y.mean()))
        return {"Slope": float(slope), "Y-intercept": float(icpt)}
    start = {"Bottom": s.lo, "Top": s.hi, "HillSlope": s.d}
    if m.hill == "fixed" and s.d != m.direction:
        start["Bottom"], start["Top"] = s.hi, s.lo
    start[m.potency] = s.half
    return start


def initial_values(m: Model, x, y, c: dict | None = None) -> np.ndarray:
    """Starting values in ``m.params`` order (NaN where the rule gives none, e.g. constants)."""
    s = summary(m.x_scale, x, y)
    d = m.init(s, c or {}) if m.init else _legacy_init(m, s, c or {})
    return np.array([d.get(n, np.nan) for n in m.params], float)


# ---------------------------------------------------------------- equations: dose-response (existing)

def _hill(x, p):  # Bottom, Top, LogEC50, HillSlope
    return p[0] + (p[1] - p[0]) / (1 + 10.0 ** ((p[2] - x) * p[3]))


def _agonist_std(x, p):  # Bottom, Top, LogEC50
    return p[0] + (p[1] - p[0]) / (1 + 10.0 ** (p[2] - x))


def _inhibitor_std(x, p):  # Bottom, Top, LogIC50
    return p[0] + (p[1] - p[0]) / (1 + 10.0 ** (x - p[2]))


def _normalized(x, p):  # LogEC50, HillSlope
    return 100.0 / (1 + 10.0 ** ((p[0] - x) * p[1]))


def _agonist_dose(x, p):  # Bottom, Top, EC50, HillSlope
    xh = x ** p[3]
    return p[0] + xh * (p[1] - p[0]) / (xh + p[2] ** p[3])


def _inhibitor_dose(x, p):  # Bottom, Top, IC50, HillSlope (negative for inhibition)
    return p[0] + (p[1] - p[0]) / (1 + (p[2] / x) ** p[3])


def _line(x, p):  # Slope, Y-intercept
    return p[0] * x + p[1]


# ---------------------------------------------------------------- equations: binding

def _sat_total(x, p):  # Bmax, Kd, NS, Background
    return p[0] * x / (p[1] + x) + p[2] * x + p[3]


def _sat_specific(x, p):  # Bmax, Kd
    return p[0] * x / (p[1] + x)


def _sat_ns(x, p):  # nonspecific data set: NS, Background in slots 2, 3
    return p[2] * x + p[3]


def _two_total(x, p):  # BmaxHi, KdHi, BmaxLo, KdLo, NS, Background
    return p[0] * x / (x + p[1]) + p[2] * x / (x + p[3]) + p[4] * x + p[5]


def _two_ns(x, p):
    return p[4] * x + p[5]


def _two_specific(x, p):  # BmaxHi, KdHi, BmaxLo, KdLo
    return p[0] * x / (p[1] + x) + p[2] * x / (p[3] + x)


def _sat_hill(x, p):  # Bmax, Kd, h
    return p[0] * x ** p[2] / (p[1] ** p[2] + x ** p[2])


def _comp_ki(x, p):  # Bottom, Top, LogKi, HotNM, HotKdNM
    logec50 = p[2] + np.log10(1 + p[3] / p[4])
    return p[0] + (p[1] - p[0]) / (1 + 10.0 ** (x - logec50))


def _comp_two_ic50(x, p):  # Bottom, Top, FractionHi, LogIC50Hi, LogIC50Lo
    span = p[1] - p[0]
    return p[0] + p[2] * span / (1 + 10.0 ** (x - p[3])) + (1 - p[2]) * span / (1 + 10.0 ** (x - p[4]))


def _comp_two_ki(x, p):  # Bottom, Top, FractionHi, LogKiHi, LogKiLo, HotNM, HotKdNMHi, HotKdNMLo
    hi_ = p[3] + np.log10(1 + p[5] / p[6])
    lo_ = p[4] + np.log10(1 + p[5] / p[7])
    span = p[1] - p[0]
    return p[0] + p[2] * span / (1 + 10.0 ** (x - hi_)) + (1 - p[2]) * span / (1 + 10.0 ** (x - lo_))


def _assoc(x, p):  # Kon, Koff, Bmax, HotNM   (Kon in 1/(M*time), HotNM in nM)
    L = p[3] * 1e-9
    kob = p[0] * L + p[1]
    return p[2] * L / (L + p[1] / p[0]) * (1 - np.exp(-kob * x))


def _dissoc(x, p):  # Y0, NS, K
    return (p[0] - p[1]) * np.exp(-p[2] * x) + p[1]


def _assoc_dissoc(x, p):  # Kon, Koff, Bmax, HotNM, Time0, NS
    L = p[3] * 1e-9
    kob = p[0] * L + p[1]
    eq = p[2] * L / (L + p[1] / p[0])
    y_t0 = eq * (1 - np.exp(-kob * p[4]))
    return np.where(x < p[4], eq * (1 - np.exp(-kob * x)), y_t0 * np.exp(-p[1] * (x - p[4]))) + p[5]


# ---------------------------------------------------------------- equations: enzyme kinetics

def _mm(x, p):  # Vmax, Km
    return p[0] * x / (p[1] + x)


def _kcat(x, p):  # Et, kcat, Km
    return p[0] * p[1] * x / (p[2] + x)


def _allosteric(x, p):  # Vmax, Khalf, h
    return p[0] * x ** p[2] / (p[1] ** p[2] + x ** p[2])


def _sub_inhib(x, p):  # Vmax, Km, Ki
    return p[0] * x / (p[1] + x * (1 + x / p[2]))


def _enz_comp(x, p):  # Vmax, Km, Ki, I
    return p[0] * x / (p[1] * (1 + p[3] / p[2]) + x)


def _enz_noncomp(x, p):  # Vmax, Km, Ki, I
    return p[0] / (1 + p[3] / p[2]) * x / (p[1] + x)


def _enz_uncomp(x, p):  # Vmax, Km, Ki, I
    f = 1 + p[3] / p[2]
    return p[0] / f * x / (p[1] / f + x)


def _enz_mixed(x, p):  # Vmax, Km, Ki, Alpha, I
    a = 1 + p[4] / (p[3] * p[2])
    vapp = p[0] / a
    kapp = p[1] * (1 + p[4] / p[2]) / a
    return vapp * x / (kapp + x)


def _morrison(x, p):  # Vo, Ki, Et, S, Km   (X = inhibitor concentration)
    b = p[2] + x + p[1] * (1 + p[3] / p[4])
    return p[0] * (1 - (b - np.sqrt(np.maximum(b * b - 4 * p[2] * x, 0.0))) / (2 * p[2]))


# ---------------------------------------------------------------- equations: special dose-response

def _five_log(x, p):  # Bottom, Top, LogEC50, HillSlope, S
    logxb = p[2] + np.log10(2.0 ** (1 / p[4]) - 1) / p[3]
    return p[0] + (p[1] - p[0]) / (1 + 10.0 ** ((logxb - x) * p[3])) ** p[4]


def _five_dose(x, p):  # Bottom, Top, EC50, HillSlope, S
    return p[0] + (p[1] - p[0]) / (1 + (2.0 ** (1 / p[4]) - 1) * (p[2] / x) ** p[3]) ** p[4]


def _biphasic_log(x, p):  # Bottom, Top, Frac, LogEC50_1, LogEC50_2, nH1, nH2
    span = p[1] - p[0]
    return p[0] + span * p[2] / (1 + 10.0 ** ((p[3] - x) * p[5])) + span * (1 - p[2]) / (1 + 10.0 ** ((p[4] - x) * p[6]))


def _biphasic_dose(x, p):  # Bottom, Top, Frac, EC50_1, EC50_2, nH1, nH2
    span = p[1] - p[0]
    return p[0] + span * p[2] / (1 + (p[3] / x) ** p[5]) + span * (1 - p[2]) / (1 + (p[4] / x) ** p[6])


def _bell_log(x, p):  # Dip, Plateau1, Plateau2, LogEC50_1, LogEC50_2, nH1, nH2
    return p[0] + (p[1] - p[0]) / (1 + 10.0 ** ((p[3] - x) * p[5])) + (p[2] - p[0]) / (1 + 10.0 ** ((x - p[4]) * p[6]))


def _bell_dose(x, p):  # Dip, Plateau1, Plateau2, EC50_1, EC50_2, nH1, nH2
    return p[0] + (p[1] - p[0]) / (1 + (p[3] / x) ** p[5]) + (p[2] - p[0]) / (1 + (x / p[4]) ** p[6])


def _op_log(x, p):  # Basal, Effectmax, logKA, logTau, n
    operate = ((10.0 ** p[2] + 10.0 ** x) / 10.0 ** (p[3] + x)) ** p[4]
    return p[0] + (p[1] - p[0]) / (1 + operate)


def _op_full_log(x, p):  # data set A (full agonist): Basal, Effectmax, LogEC50, n
    return p[0] + (p[1] - p[0]) / (1 + 10.0 ** ((p[2] - x) * p[5]))


def _op_global(x, p):  # Basal, Effectmax, LogEC50, logKA, logTau, n  (partial agonist data sets)
    return _op_log(x, [p[0], p[1], p[3], p[4], p[5]])


def _op_dose(x, p):  # Basal, Effectmax, KA, Tau, n
    return p[0] + (p[1] - p[0]) / (1 + ((p[2] + x) / (p[3] * x)) ** p[4])


def _schild_log(x, p):  # Bottom, Top, LogEC50, HillSlope, pA2, SchildSlope, B
    antag = 1 + (p[6] / 10.0 ** (-p[4])) ** p[5]
    return p[0] + (p[1] - p[0]) / (1 + 10.0 ** ((p[2] + np.log10(antag) - x) * p[3]))


def _schild_dose(x, p):  # Bottom, Top, EC50, HillSlope, pA2, SchildSlope, B
    antag = 1 + (p[6] / 10.0 ** (-p[4])) ** p[5]
    return p[0] + (p[1] - p[0]) / (1 + (p[2] * antag / x) ** p[3])


def _schild_pa2_log(x, p):  # Bottom, Top, LogEC50, HillSlope, pA2, B
    return _schild_log(x, [*p[:5], 1.0, p[5]])


def _schild_pa2_dose(x, p):  # Bottom, Top, EC50, HillSlope, pA2, B
    return _schild_dose(x, [*p[:5], 1.0, p[5]])


def _allo_log(x, p):  # Bottom, Top, LogEC50, HillSlope, LogKB, LogAlpha, B
    kb = 10.0 ** p[4]
    antag = (1 + p[6] / kb) / (1 + 10.0 ** p[5] * p[6] / kb)
    return p[0] + (p[1] - p[0]) / (1 + 10.0 ** ((p[2] + np.log10(antag) - x) * p[3]))


def _allo_dose(x, p):  # Bottom, Top, EC50, HillSlope, KB, Alpha, B
    antag = (1 + p[6] / p[4]) / (1 + p[5] * p[6] / p[4])
    return p[0] + (p[1] - p[0]) / (1 + (p[2] * antag / x) ** p[3])


def _ecf_log(x, p):  # Bottom, Top, LogECF, HillSlope, F
    logec50 = p[2] - np.log10(p[4] / (100 - p[4])) / p[3]
    return p[0] + (p[1] - p[0]) / (1 + 10.0 ** ((logec50 - x) * p[3]))


def _ecf_dose(x, p):  # Bottom, Top, ECF, HillSlope, F
    ec50 = p[2] / (p[4] / (100 - p[4])) ** (1 / p[3])
    return p[0] + (p[1] - p[0]) / (1 + (ec50 / x) ** p[3])


# ---------------------------------------------------------------- equations: general

def _exp_growth(x, p):  # Y0, k
    return p[0] * np.exp(p[1] * x)


def _decay1(x, p):  # Y0, Plateau, K
    return (p[0] - p[1]) * np.exp(-p[2] * x) + p[1]


def _assoc1(x, p):  # Y0, Plateau, K
    return p[0] + (p[1] - p[0]) * (1 - np.exp(-p[2] * x))


def _decay2(x, p):  # Y0, Plateau, PercentFast, KFast, KSlow
    span = p[0] - p[1]
    return p[1] + span * p[2] * 0.01 * np.exp(-p[3] * x) + span * (100 - p[2]) * 0.01 * np.exp(-p[4] * x)


def _assoc2(x, p):  # Y0, Plateau, PercentFast, KFast, KSlow
    span = p[1] - p[0]
    return p[0] + span * p[2] * 0.01 * (1 - np.exp(-p[3] * x)) + span * (100 - p[2]) * 0.01 * (1 - np.exp(-p[4] * x))


def _plateau_decay(x, p):  # X0, Y0, Plateau, K
    return np.where(x < p[0], p[1], p[2] + (p[1] - p[2]) * np.exp(-p[3] * (x - p[0])))


def _poly(deg):
    return lambda x, p: sum(p[i] * x ** i for i in range(deg + 1))


def _gauss(x, p):  # Amplitude, Mean, SD
    return p[0] * np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2)


def _lorentz(x, p):  # Amplitude, Center, Width
    return p[0] / (1 + ((x - p[1]) / p[2]) ** 2)


def _lognormal(x, p):  # A, GeoMean, GeoSD
    return (p[0] / x) * np.exp(-0.5 * (np.log(x / p[1]) / np.log(p[2])) ** 2)


def _sine(x, p):  # Amplitude, Wavelength, PhaseShift
    return p[0] * np.sin(2 * np.pi * x / p[1] + p[2])


def _sine_base(x, p):  # Baseline, Amplitude, Wavelength, PhaseShift
    return p[0] + p[1] * np.sin(2 * np.pi * x / p[2] + p[3])


def _sine_damped(x, p):  # Amplitude, K, Wavelength, PhaseShift
    return p[0] * np.exp(-p[1] * x) * np.sin(2 * np.pi * x / p[2] + p[3])


def _gompertz(x, p):  # Y0, YM, K
    return p[1] * (p[0] / p[1]) ** np.exp(-p[2] * x)


def _logistic(x, p):  # Y0, YM, k
    return p[1] * p[0] / ((p[1] - p[0]) * np.exp(-p[2] * x) + p[0])


def _beta(x, p):  # Ym, Te, Tm
    return p[0] * (1 + (p[1] - x) / (p[1] - p[2])) * (x / p[1]) ** (p[1] / (p[1] - p[2]))


def _lq(kind):
    """Linear-quadratic survival. kind: (scale, y0 parameter present, killed = 1 - survival)."""
    scale, y0, killed = kind

    def f(x, p):
        a, b = (p[1], p[2]) if y0 else (p[0], p[1])
        s = np.exp(-(a * x + b * x * x))
        top = p[0] if y0 else scale
        return top * (1 - s) if killed else top * s
    return f


# ---------------------------------------------------------------- starting values

def _dr_base(s):
    return {"Bottom": s.lo, "Top": s.hi, "HillSlope": s.d}


def _i_sat_specific(s, c):
    return {"Bmax": s.hi, "Kd": s.half}


def _i_sat_total(s, c):
    slope = (s.ym[-1] - s.ym[-2]) / (s.xs[-1] - s.xs[-2]) if s.n > 1 else 0.0
    ns = max(slope, 0.0) * 0.5
    span = s.hi - s.lo
    return {"Bmax": max(span - ns * s.xmax, 0.3 * span), "Kd": s.half, "NS": ns, "Background": s.lo}


def _i_sat_total_ns(s, c):
    if c.get("_ds", 0) == 0:
        return _i_sat_total(s, c)
    slope, icpt = np.polyfit(s.x, s.y, 1)
    return {"NS": float(slope), "Background": float(icpt)}


def _i_two_specific(s, c):
    return {"BmaxHi": 0.4 * s.hi, "KdHi": 0.1 * s.half, "BmaxLo": 0.6 * s.hi, "KdLo": 10 * s.half}


def _i_two_total_ns(s, c):
    if c.get("_ds", 0) == 0:
        t = _i_sat_total(s, c)
        return {"BmaxHi": 0.4 * t["Bmax"], "KdHi": 0.1 * s.half, "BmaxLo": 0.6 * t["Bmax"], "KdLo": 10 * s.half,
                "NS": t["NS"], "Background": t["Background"]}
    slope, icpt = np.polyfit(s.x, s.y, 1)
    return {"NS": float(slope), "Background": float(icpt)}


def _i_hill_binding(s, c):
    return {"Bmax": s.hi, "Kd": s.half, "h": 1.0}


def _tb(s):  # competition: Top is the response at the lowest competitor concentration
    return (s.hi, s.lo) if s.d < 0 else (s.lo, s.hi)


def _i_comp_ki(s, c):
    tp, bt = _tb(s)
    return {"Top": tp, "Bottom": bt, "LogKi": s.half - np.log10(1 + c["HotNM"] / c["HotKdNM"])}


def _i_comp_two_ic50(s, c):
    tp, bt = _tb(s)
    return {"Top": tp, "Bottom": bt, "FractionHi": 0.5, "LogIC50Hi": s.half - 1, "LogIC50Lo": s.half + 1}


def _i_comp_two_ki(s, c):
    tp, bt = _tb(s)
    sh = lambda k: np.log10(1 + c["HotNM"] / c[k])  # noqa: E731
    return {"Top": tp, "Bottom": bt, "FractionHi": 0.5, "LogKiHi": s.half - 1 - sh("HotKdNMHi"),
            "LogKiLo": s.half + 1 - sh("HotKdNMLo")}


def _assoc_rates(s, c, y0, yinf):
    """Kon, Koff from the half time of the rise to ``yinf`` (Bmax is the user's constant if given)."""
    L = c["HotNM"] * 1e-9
    bmax = c.get("Bmax", 2 * max(abs(yinf - y0), 1e-12))
    occ = float(np.clip(abs(yinf - y0) / bmax, 0.05, 0.95))
    kob = _rate(s, y0, yinf)
    koff = kob * (1 - occ)  # kob = Koff * (1 + L/Kd) = Koff / (1 - occ)
    return {"Kon": koff * occ / (L * (1 - occ)), "Koff": koff, "Bmax": bmax}


def _i_assoc(s, c):
    return _assoc_rates(s, c, 0.0, s.hi)


def _i_dissoc(s, c):
    return {"Y0": float(s.ym[0]), "NS": float(s.ym[-1]), "K": _rate(s, s.ym[0], s.ym[-1])}


def _i_assoc_dissoc(s, c):
    pre = s.xs <= c["Time0"]
    ym_pre, xs_pre = s.ym[pre], s.xs[pre]
    sub = SimpleNamespace(**{**vars(s), "xs": xs_pre, "ym": ym_pre, "n": len(xs_pre), "R": float(xs_pre[-1] - xs_pre[0]) or 1.0})
    base = float(s.ym[0])
    out = _assoc_rates(sub, c, base, float(ym_pre.max()))
    out["NS"] = base
    return out


def _i_mm(s, c):
    return {"Vmax": s.hi, "Km": s.half}


def _i_kcat(s, c):
    return {"kcat": s.hi / c["Et"], "Km": s.half}


def _i_allosteric(s, c):
    return {"Vmax": s.hi, "Khalf": s.half, "h": 1.0}


def _i_sub_inhib(s, c):
    return {"Vmax": 1.5 * s.hi, "Km": s.half, "Ki": 3 * max(s.xmax, float(s.xs[int(np.argmax(s.ym))]))}


def _i_enz(s, c):
    out = {"Vmax": s.hi, "Km": s.half}
    if c.get("I", 0) > 0:
        out["Ki"] = c["I"]
    out["Alpha"] = 1.0
    return out


def _i_morrison(s, c):
    return {"Vo": float(s.ym[0]), "Ki": max(0.5 * s.half, 1e-3 * s.xmax)}


def _i_five(s, c):
    return {**_dr_base(s), "LogEC50" if s.scale == "log" else "EC50": s.half, "S": 1.0}


def _i_biphasic(s, c):
    k = "LogEC50_" if s.scale == "log" else "EC50_"
    return {"Bottom": s.lo, "Top": s.hi, "Frac": 0.5, k + "1": _at(s, 0.25), k + "2": _at(s, 0.75), "nH1": s.d, "nH2": s.d}


def _i_bell(s, c):
    ip = int(np.argmax(np.abs(s.ym - (s.ym[0] + s.ym[-1]) / 2)))
    k = "LogEC50_" if s.scale == "log" else "EC50_"
    xp = s.xs[ip]
    mid = lambda a, b: float(np.sqrt(a * b)) if s.scale == "dose" and a > 0 and b > 0 else float((a + b) / 2)  # noqa: E731
    return {"Dip": float(s.ym[0] + s.ym[-1] - s.ym[ip]), "Plateau1": float(s.ym[-1]), "Plateau2": float(s.ym[0]),
            k + "1": mid(s.xs[0], xp), k + "2": mid(xp, s.xs[-1]), "nH1": 1.0, "nH2": 1.0}


def _i_operational(s, c):
    return {"Basal": s.lo, "Effectmax": s.lo + 2 * (s.hi - s.lo), "logKA": s.half + np.log10(2), "logTau": 0.0, "n": 1.0}


def _i_op_global(s, c):
    if c.get("_ds", 0) == 0:
        return {"Basal": s.lo, "Effectmax": s.hi, "LogEC50": s.half, "n": 1.0}
    return {"Basal": s.lo, "logKA": s.half + 0.3, "logTau": 0.0, "n": 1.0}


def _i_op_dose(s, c):
    return {"Basal": s.lo, "Effectmax": s.lo + 2 * (s.hi - s.lo), "KA": 2 * s.half, "Tau": 1.0, "n": 1.0}


def _i_schild(pa2=True, slope=True, log=True):
    def f(s, c):
        out = _dr_base(s)
        b = c.get("B", 0.0)
        shift = np.log10(2) if (b > 0 and pa2) else 0.0
        out["LogEC50" if log else "EC50"] = s.half - shift if log else s.half / 10 ** shift
        if b > 0 and pa2:
            out["pA2"] = -np.log10(b)
        if slope:
            out["SchildSlope"] = 1.0
        return out
    return f


def _i_allo(log):
    def f(s, c):
        out = _dr_base(s)
        out["LogEC50" if log else "EC50"] = s.half
        b = c.get("B", 0.0)
        if b > 0:
            out["LogKB" if log else "KB"] = np.log10(b) if log else b
        out["LogAlpha" if log else "Alpha"] = -0.5 if log else 0.3
        return out
    return f


def _i_ecf(s, c):
    return {**_dr_base(s), "LogECF" if s.scale == "log" else "ECF": s.half}


def _i_exp_growth(s, c):
    y0 = float(s.ym[0]) if s.ym[0] != 0 else float(max(s.ym[1], 1e-6)) if s.n > 1 else 1.0
    ratio = s.ym[-1] / y0
    k = np.log(ratio) / s.R if ratio > 0 else 0.1 / s.R
    return {"Y0": y0, "k": float(k)}


def _i_decay1(s, c):
    return {"Y0": float(s.ym[0]), "Plateau": float(s.ym[-1]), "K": _rate(s, s.ym[0], s.ym[-1])}


def _i_decay2(s, c):
    k = _rate(s, s.ym[0], s.ym[-1])
    return {"Y0": float(s.ym[0]), "Plateau": float(s.ym[-1]), "PercentFast": 50.0, "KFast": 5 * k, "KSlow": 0.3 * k}


def _i_plateau_decay(s, c):
    start = next((i for i in range(s.n) if abs(s.ym[i] - s.ym[0]) > 0.05 * (s.hi - s.lo)), 1)
    x0 = float(s.xs[max(start - 1, 0)])
    sub = SimpleNamespace(**{**vars(s), "xs": s.xs[max(start - 1, 0):], "ym": s.ym[max(start - 1, 0):]})
    sub.n, sub.R = len(sub.xs), float(sub.xs[-1] - sub.xs[0]) or 1.0
    return {"X0": x0, "Y0": float(s.ym[0]), "Plateau": float(s.ym[-1]), "K": _rate(sub, s.ym[0], s.ym[-1])}


def _i_poly(deg):
    def f(s, c):
        co = np.linalg.lstsq(np.vander(s.x, deg + 1, increasing=True), s.y, rcond=None)[0]
        return {f"B{i}": float(v) for i, v in enumerate(co)}
    return f


def _moments(s):
    w = np.clip(s.ym - s.lo, 0, None)
    w = w if w.sum() > 0 else np.ones_like(w)
    mean = float(np.sum(w * s.xs) / w.sum())
    return mean, float(np.sqrt(np.sum(w * (s.xs - mean) ** 2) / w.sum())) or s.R / 6


def _i_gauss(s, c):
    mean, sd = _moments(s)
    return {"Amplitude": s.hi, "Mean": mean, "SD": sd}


def _i_lorentz(s, c):
    xp = float(s.xs[int(np.argmax(s.ym))])
    above = s.xs[s.ym >= s.lo + 0.5 * (s.hi - s.lo)]
    return {"Amplitude": s.hi, "Center": xp, "Width": max(float(above.max() - above.min()) / 2, s.R / 20)}


def _i_lognormal(s, c):
    lx = np.log(s.xs)
    w = np.clip(s.ym - s.lo, 0, None)
    w = w if w.sum() > 0 else np.ones_like(w)
    mu = float(np.sum(w * lx) / w.sum())
    sg = float(np.sqrt(np.sum(w * (lx - mu) ** 2) / w.sum())) or 0.5
    xp = float(s.xs[int(np.argmax(s.ym))])
    a = s.hi * xp / np.exp(-0.5 * ((np.log(xp) - mu) / sg) ** 2)
    return {"A": float(a), "GeoMean": float(np.exp(mu)), "GeoSD": float(max(np.exp(sg), 1.05))}


def _sine_scan(s, baseline, damped):
    """Wavelength (and K) by linear least squares of sin/cos terms over a grid; returns a start dict."""
    dx = np.diff(s.xs)
    wls = np.geomspace(2 * float(np.median(dx)), 2 * s.R, 400)
    ks = (0.0, 1 / s.R, 3 / s.R) if damped else (0.0,)
    best = None
    for k in ks:
        env = np.exp(-k * s.x)
        for wl in wls:
            cols = [env * np.sin(2 * np.pi * s.x / wl), env * np.cos(2 * np.pi * s.x / wl)]
            if baseline:
                cols.insert(0, np.ones_like(s.x))
            a = np.column_stack(cols)
            sol, res, *_ = np.linalg.lstsq(a, s.y, rcond=None)
            sse = float(np.sum((s.y - a @ sol) ** 2))
            if best is None or sse < best[0]:
                best = (sse, wl, k, sol)
    _, wl, k, sol = best
    sa, ca = sol[-2], sol[-1]
    out = {"Amplitude": float(np.hypot(sa, ca)), "Wavelength": float(wl), "PhaseShift": float(np.arctan2(ca, sa))}
    if baseline:
        out["Baseline"] = float(sol[0])
    if damped:
        out["K"] = float(max(k, 0.1 / s.R))
    return out


def _i_gompertz(s, c):
    y0 = float(max(s.ym[0], 1e-6 * s.hi))
    ym_ = 1.1 * s.hi
    return {"Y0": y0, "YM": ym_, "K": _rate(s, s.ym[0], s.ym[-1])}


def _i_logistic(s, c):
    y0, ym_ = float(max(s.ym[0], 1e-6 * s.hi)), 1.1 * s.hi
    th = max(float(s.half - s.xmin), 0.02 * s.R)
    return {"Y0": y0, "YM": ym_, "k": float(np.log(max((ym_ - y0) / y0, 1.5)) / th)}


def _i_beta(s, c):
    tm = float(s.xs[int(np.argmax(s.ym))])
    te = s.xmax if s.xmax > tm else 1.5 * tm
    return {"Ym": s.hi, "Te": float(te), "Tm": tm}


def _i_lq(kind):
    scale, y0, killed = kind

    def f(s, c):
        top = float(s.ym[0]) if y0 else (scale or 1.0)
        surv = s.ym / top
        surv = 1 - surv if killed else surv
        ok = surv > 1e-9
        t = -np.log(surv[ok])
        a = np.column_stack([s.xs[ok], s.xs[ok] ** 2])
        sol = np.linalg.lstsq(a, t, rcond=None)[0] if ok.any() else np.array([0.1, 0.01])
        a_, b_ = max(float(sol[0]), 1e-3), max(float(sol[1]), 1e-4)
        out = {"A": a_, "B": b_}
        if y0:
            out["Y0"] = top
        return out
    return f


# ---------------------------------------------------------------- the registry

_POS = (("EC50", 0.0, INF),)
_POS_I = (("IC50", 0.0, INF),)


def _pos(*names):
    return tuple((n, 0.0, INF) for n in names)


def _tr(name, src, fn):
    return ("t", name, src, fn)


_10 = lambda v: 10.0 ** v  # noqa: E731
_half_life = lambda k: LN2 / k  # noqa: E731
_tau = lambda k: 1.0 / k  # noqa: E731

B, EN, SP, GEN = "Binding", "Enzyme kinetics", "Dose-response (special)", "General"
_LQ = {"": (1.0, False, False), "_pct": (100.0, False, False), "_y0": (None, True, False)}

_MODELS = [
    # ---- dose-response (the original set)
    Model("log_agonist_variable", "log(agonist) vs. response -- Variable slope (four parameters)",
          ("Bottom", "Top", "LogEC50", "HillSlope"), _hill, "log", +1),
    Model("log_agonist_standard", "log(agonist) vs. response (three parameters)",
          ("Bottom", "Top", "LogEC50"), _agonist_std, "log", +1, "fixed"),
    Model("log_inhibitor_variable", "log(inhibitor) vs. response -- Variable slope (four parameters)",
          ("Bottom", "Top", "LogIC50", "HillSlope"), _hill, "log", -1),
    Model("log_inhibitor_standard", "log(inhibitor) vs. response (three parameters)",
          ("Bottom", "Top", "LogIC50"), _inhibitor_std, "log", -1, "fixed"),
    Model("log_agonist_normalized", "log(agonist) vs. normalized response -- Variable slope",
          ("LogEC50", "HillSlope"), _normalized, "log", +1),
    Model("log_inhibitor_normalized", "log(inhibitor) vs. normalized response -- Variable slope",
          ("LogIC50", "HillSlope"), _normalized, "log", -1),
    Model("agonist_variable", "[Agonist] vs. response -- Variable slope",
          ("Bottom", "Top", "EC50", "HillSlope"), _agonist_dose, "dose", +1, "free", _POS),
    Model("inhibitor_variable", "[Inhibitor] vs. response -- Variable slope",
          ("Bottom", "Top", "IC50", "HillSlope"), _inhibitor_dose, "dose", -1, "free", _POS_I),
    Model("line", "Straight line", ("Slope", "Y-intercept"), _line, "linear", family="General"),

    # ---- binding: saturation
    Model("sat_total", "One site -- Total", ("Bmax", "Kd", "NS", "Background"), _sat_total, "dose",
          bounds=_pos("Bmax", "Kd"), family=B, init=_i_sat_total),
    Model("sat_specific", "One site -- Specific binding", ("Bmax", "Kd"), _sat_specific, "dose",
          bounds=_pos("Bmax", "Kd"), family=B, init=_i_sat_specific),
    Model("sat_total_ns", "One site -- Fit total and nonspecific binding (data set A total, B nonspecific)",
          ("Bmax", "Kd", "NS", "Background"), _sat_total, "dose", bounds=_pos("Bmax", "Kd"), family=B,
          variants=((_sat_total, None), (_sat_ns, ("NS", "Background"))), n_datasets=2, shared="all",
          init=_i_sat_total_ns),
    Model("sat_two_total_ns", "Two sites -- Fit total and nonspecific binding (data set A total, B nonspecific)",
          ("BmaxHi", "KdHi", "BmaxLo", "KdLo", "NS", "Background"), _two_total, "dose",
          bounds=_pos("BmaxHi", "KdHi", "BmaxLo", "KdLo"), family=B,
          variants=((_two_total, None), (_two_ns, ("NS", "Background"))), n_datasets=2, shared="all",
          init=_i_two_total_ns,
          derived=(("d", "Percent high affinity", lambda q: 100 * q["BmaxHi"] / (q["BmaxHi"] + q["BmaxLo"])),
                   ("d", "Affinity ratio (KdLo/KdHi)", lambda q: q["KdLo"] / q["KdHi"]))),
    Model("sat_two_specific", "Two sites -- Specific binding", ("BmaxHi", "KdHi", "BmaxLo", "KdLo"),
          _two_specific, "dose", bounds=_pos("BmaxHi", "KdHi", "BmaxLo", "KdLo"), family=B, init=_i_two_specific,
          derived=(("d", "Percent high affinity", lambda q: 100 * q["BmaxHi"] / (q["BmaxHi"] + q["BmaxLo"])),
                   ("d", "Affinity ratio (KdLo/KdHi)", lambda q: q["KdLo"] / q["KdHi"]))),
    Model("sat_hill", "Specific binding with Hill slope", ("Bmax", "Kd", "h"), _sat_hill, "dose",
          bounds=_pos("Bmax", "Kd"), family=B, init=_i_hill_binding),

    # ---- binding: competition
    Model("comp_logic50", "Competitive binding, one site -- Fit logIC50", ("Bottom", "Top", "LogIC50"),
          _inhibitor_std, "log", -1, "fixed", family=B),
    Model("comp_ki", "Competitive binding, one site -- Fit Ki", ("Bottom", "Top", "LogKi", "HotNM", "HotKdNM"),
          _comp_ki, "log", -1, family=B, constants=("HotNM", "HotKdNM"), init=_i_comp_ki,
          derived=(_tr("Ki", "LogKi", _10),
                   ("d", "IC50", lambda q: 10.0 ** q["LogKi"] * (1 + q["HotNM"] / q["HotKdNM"])))),
    Model("comp_two_logic50", "Competitive binding, two sites -- Fit logIC50",
          ("Bottom", "Top", "FractionHi", "LogIC50Hi", "LogIC50Lo"), _comp_two_ic50, "log", -1, family=B,
          bounds=(("FractionHi", 0.0, 1.0),), init=_i_comp_two_ic50,
          derived=(_tr("IC50Hi", "LogIC50Hi", _10), _tr("IC50Lo", "LogIC50Lo", _10))),
    Model("comp_two_ki", "Competitive binding, two sites -- Fit Ki",
          ("Bottom", "Top", "FractionHi", "LogKiHi", "LogKiLo", "HotNM", "HotKdNMHi", "HotKdNMLo"),
          _comp_two_ki, "log", -1, family=B, bounds=(("FractionHi", 0.0, 1.0),),
          constants=("HotNM", "HotKdNMHi", "HotKdNMLo"), init=_i_comp_two_ki,
          derived=(_tr("KiHi", "LogKiHi", _10), _tr("KiLo", "LogKiLo", _10))),

    # ---- binding: kinetics
    Model("kin_association", "Association kinetics (one conc. of hot)", ("Kon", "Koff", "Bmax", "HotNM"),
          _assoc, "linear", family=B, bounds=_pos("Kon", "Koff", "Bmax"), constants=("HotNM",), init=_i_assoc,
          derived=(("d", "Kd", lambda q: q["Koff"] / q["Kon"]),
                   ("d", "Kob", lambda q: q["Kon"] * q["HotNM"] * 1e-9 + q["Koff"]))),
    Model("kin_dissociation", "Dissociation -- One phase exponential decay", ("Y0", "NS", "K"), _dissoc, "linear",
          family=B, bounds=_pos("K"), init=_i_dissoc, derived=(_tr("Half-life", "K", _half_life),)),
    Model("kin_assoc_dissoc", "Association then dissociation", ("Kon", "Koff", "Bmax", "HotNM", "Time0", "NS"),
          _assoc_dissoc, "linear", family=B, bounds=_pos("Kon", "Koff", "Bmax"), constants=("HotNM", "Time0"),
          init=_i_assoc_dissoc, derived=(("d", "Kd", lambda q: q["Koff"] / q["Kon"]),)),

    # ---- enzyme kinetics
    Model("enz_mm", "Michaelis-Menten", ("Vmax", "Km"), _mm, "dose", bounds=_pos("Vmax", "Km"), family=EN, init=_i_mm),
    Model("enz_kcat", "Michaelis-Menten, kcat (Et constant)", ("Et", "kcat", "Km"), _kcat, "dose",
          bounds=_pos("kcat", "Km"), family=EN, constants=("Et",), init=_i_kcat,
          derived=(("d", "Vmax", lambda q: q["Et"] * q["kcat"]),)),
    Model("enz_allosteric", "Allosteric sigmoidal", ("Vmax", "Khalf", "h"), _allosteric, "dose",
          bounds=_pos("Vmax", "Khalf", "h"), family=EN, init=_i_allosteric),
    Model("enz_substrate_inhibition", "Substrate inhibition", ("Vmax", "Km", "Ki"), _sub_inhib, "dose",
          bounds=_pos("Vmax", "Km", "Ki"), family=EN, init=_i_sub_inhib),
    Model("enz_competitive", "Competitive inhibition (X substrate; inhibitor I from the data-set title)",
          ("Vmax", "Km", "Ki", "I"), _enz_comp, "dose", bounds=_pos("Vmax", "Km", "Ki"), family=EN,
          constants=("I",), title_const="I", shared="all", init=_i_enz),
    Model("enz_noncompetitive", "Noncompetitive inhibition (X substrate; inhibitor I from the data-set title)",
          ("Vmax", "Km", "Ki", "I"), _enz_noncomp, "dose", bounds=_pos("Vmax", "Km", "Ki"), family=EN,
          constants=("I",), title_const="I", shared="all", init=_i_enz),
    Model("enz_uncompetitive", "Uncompetitive inhibition (X substrate; inhibitor I from the data-set title)",
          ("Vmax", "Km", "Ki", "I"), _enz_uncomp, "dose", bounds=_pos("Vmax", "Km", "Ki"), family=EN,
          constants=("I",), title_const="I", shared="all", init=_i_enz),
    Model("enz_mixed", "Mixed-model inhibition (X substrate; inhibitor I from the data-set title)",
          ("Vmax", "Km", "Ki", "Alpha", "I"), _enz_mixed, "dose", bounds=_pos("Vmax", "Km", "Ki", "Alpha"),
          family=EN, constants=("I",), title_const="I", shared="all", init=_i_enz),
    Model("enz_morrison", "Morrison equation (tight-binding inhibitor Ki; X inhibitor)",
          ("Vo", "Ki", "Et", "S", "Km"), _morrison, "dose", -1, bounds=_pos("Vo", "Ki"), family=EN,
          constants=("Et", "S", "Km"), init=_i_morrison),

    # ---- special dose-response
    Model("dr_5pl_log", "log(agonist) vs. response -- Asymmetric (five parameters)",
          ("Bottom", "Top", "LogEC50", "HillSlope", "S"), _five_log, "log", +1, bounds=_pos("S"), family=SP,
          init=_i_five),
    Model("dr_5pl_dose", "[Agonist] vs. response -- Asymmetric (five parameters)",
          ("Bottom", "Top", "EC50", "HillSlope", "S"), _five_dose, "dose", +1, bounds=_pos("EC50", "S"), family=SP,
          init=_i_five,
          # Prism 10.4.1 reports SE(EC50) = SE / ln(10) for this equation (round-2 comparison, 2 data sets, ratio
          # 0.434298 and 0.434298; inconsistent with its own CI); value, CI, dependency and other SEs are unaffected
          se_scale=(("EC50", LOG10E),)),
    Model("dr_biphasic_log", "log(agonist) vs. response -- Biphasic",
          ("Bottom", "Top", "Frac", "LogEC50_1", "LogEC50_2", "nH1", "nH2"), _biphasic_log, "log", +1,
          bounds=(("Frac", 0.0, 1.0),), family=SP, init=_i_biphasic),
    Model("dr_biphasic_dose", "[Agonist] vs. response -- Biphasic",
          ("Bottom", "Top", "Frac", "EC50_1", "EC50_2", "nH1", "nH2"), _biphasic_dose, "dose", +1,
          bounds=(("Frac", 0.0, 1.0), ("EC50_1", 0.0, INF), ("EC50_2", 0.0, INF)), family=SP, init=_i_biphasic),
    Model("dr_bell_log", "log(dose) vs. response -- Bell-shaped",
          ("Dip", "Plateau1", "Plateau2", "LogEC50_1", "LogEC50_2", "nH1", "nH2"), _bell_log, "log", +1,
          family=SP, init=_i_bell),
    Model("dr_bell_dose", "[Dose] vs. response -- Bell-shaped",
          ("Dip", "Plateau1", "Plateau2", "EC50_1", "EC50_2", "nH1", "nH2"), _bell_dose, "dose", +1,
          bounds=(("EC50_1", 0.0, INF), ("EC50_2", 0.0, INF)), family=SP, init=_i_bell),
    Model("dr_operational_log", "Operational model -- log(agonist) vs. response (single data set)",
          ("Basal", "Effectmax", "logKA", "logTau", "n"), _op_log, "log", +1, family=SP, init=_i_operational,
          derived=(_tr("KA", "logKA", _10), _tr("Tau", "logTau", _10))),
    Model("dr_operational_global", "Operational model -- partial agonists (data set A = full agonist, global fit)",
          ("Basal", "Effectmax", "LogEC50", "logKA", "logTau", "n"), _op_global, "log", +1, family=SP,
          variants=((_op_full_log, ("Basal", "Effectmax", "LogEC50", "n")), (_op_global, ("Basal", "Effectmax", "logKA", "logTau", "n"))),
          shared=("Basal", "Effectmax", "n"), min_datasets=2, init=_i_op_global,
          derived=(_tr("KA", "logKA", _10), _tr("Tau", "logTau", _10))),
    Model("dr_operational_dose", "Operational model -- [agonist] vs. response (single data set)",
          ("Basal", "Effectmax", "KA", "Tau", "n"), _op_dose, "dose", +1, bounds=_pos("KA", "Tau"), family=SP,
          init=_i_op_dose),
    Model("dr_schild_log", "Gaddum/Schild EC50 shift -- log(agonist); antagonist B from the data-set title",
          ("Bottom", "Top", "LogEC50", "HillSlope", "pA2", "SchildSlope", "B"), _schild_log, "log", +1, family=SP,
          constants=("B",), title_const="B", shared="all", init=_i_schild(True, True, True),
          derived=(_tr("Kb", "pA2", lambda v: 10.0 ** -v),)),
    Model("dr_schild_dose", "Gaddum/Schild EC50 shift -- [agonist]; antagonist B from the data-set title",
          ("Bottom", "Top", "EC50", "HillSlope", "pA2", "SchildSlope", "B"), _schild_dose, "dose", +1,
          bounds=_pos("EC50"), family=SP, constants=("B",), title_const="B", shared="all",
          init=_i_schild(True, True, False), derived=(_tr("Kb", "pA2", lambda v: 10.0 ** -v),)),
    Model("dr_schild_pa2_log", "Gaddum/Schild EC50 shift, pA2 (Schild slope = 1) -- log(agonist)",
          ("Bottom", "Top", "LogEC50", "HillSlope", "pA2", "B"), _schild_pa2_log, "log", +1, family=SP,
          constants=("B",), title_const="B", shared="all", init=_i_schild(True, False, True),
          derived=(_tr("Kb", "pA2", lambda v: 10.0 ** -v),)),
    Model("dr_schild_pa2_dose", "Gaddum/Schild EC50 shift, pA2 (Schild slope = 1) -- [agonist]",
          ("Bottom", "Top", "EC50", "HillSlope", "pA2", "B"), _schild_pa2_dose, "dose", +1, bounds=_pos("EC50"),
          family=SP, constants=("B",), title_const="B", shared="all", init=_i_schild(True, False, False),
          derived=(_tr("Kb", "pA2", lambda v: 10.0 ** -v),)),
    Model("dr_allosteric_log", "Allosteric EC50 shift -- log(agonist); modulator B from the data-set title",
          ("Bottom", "Top", "LogEC50", "HillSlope", "LogKB", "LogAlpha", "B"), _allo_log, "log", +1, family=SP,
          constants=("B",), title_const="B", shared="all", init=_i_allo(True),
          derived=(_tr("KB", "LogKB", _10), _tr("Alpha", "LogAlpha", _10))),
    Model("dr_allosteric_dose", "Allosteric EC50 shift -- [agonist]; modulator B from the data-set title",
          ("Bottom", "Top", "EC50", "HillSlope", "KB", "Alpha", "B"), _allo_dose, "dose", +1,
          bounds=_pos("EC50", "KB", "Alpha"), family=SP, constants=("B",), title_const="B", shared="all",
          init=_i_allo(False)),
    Model("dr_ecanything_log", "log(agonist) vs. response -- Find ECanything",
          ("Bottom", "Top", "LogECF", "HillSlope", "F"), _ecf_log, "log", +1, family=SP, constants=("F",),
          const_defaults=(("F", 50.0),), init=_i_ecf, derived=(_tr("ECF", "LogECF", _10),)),
    Model("dr_ecanything_dose", "[Agonist] vs. response -- Find ECanything",
          ("Bottom", "Top", "ECF", "HillSlope", "F"), _ecf_dose, "dose", +1, bounds=_pos("ECF"), family=SP,
          constants=("F",), const_defaults=(("F", 50.0),), init=_i_ecf),

    # ---- general: exponentials
    Model("gen_exp_growth", "Exponential growth", ("Y0", "k"), _exp_growth, "linear", family=GEN,
          init=_i_exp_growth, derived=(_tr("Doubling time", "k", lambda k: LN2 / k),)),
    Model("gen_decay1", "One phase decay", ("Y0", "Plateau", "K"), _decay1, "linear", family=GEN,
          bounds=_pos("K"), init=_i_decay1,
          derived=(_tr("Half-life", "K", _half_life), _tr("Tau", "K", _tau),
                   ("d", "Span", lambda q: q["Y0"] - q["Plateau"]))),
    Model("gen_assoc1", "One phase association", ("Y0", "Plateau", "K"), _assoc1, "linear", family=GEN,
          bounds=_pos("K"), init=_i_decay1,
          derived=(_tr("Half-life", "K", _half_life), _tr("Tau", "K", _tau),
                   ("d", "Span", lambda q: q["Plateau"] - q["Y0"]))),
    Model("gen_decay2", "Two phase decay", ("Y0", "Plateau", "PercentFast", "KFast", "KSlow"), _decay2, "linear",
          family=GEN, bounds=(("PercentFast", 0.0, 100.0), ("KFast", 0.0, INF), ("KSlow", 0.0, INF)), init=_i_decay2,
          derived=(_tr("Half-life (fast)", "KFast", _half_life), _tr("Half-life (slow)", "KSlow", _half_life),
                   ("d", "Span (fast)", lambda q: (q["Y0"] - q["Plateau"]) * q["PercentFast"] / 100),
                   ("d", "Span (slow)", lambda q: (q["Y0"] - q["Plateau"]) * (100 - q["PercentFast"]) / 100))),
    Model("gen_assoc2", "Two phase association", ("Y0", "Plateau", "PercentFast", "KFast", "KSlow"), _assoc2,
          "linear", family=GEN, bounds=(("PercentFast", 0.0, 100.0), ("KFast", 0.0, INF), ("KSlow", 0.0, INF)),
          init=_i_decay2,
          derived=(_tr("Half-life (fast)", "KFast", _half_life), _tr("Half-life (slow)", "KSlow", _half_life),
                   ("d", "Span (fast)", lambda q: (q["Plateau"] - q["Y0"]) * q["PercentFast"] / 100),
                   ("d", "Span (slow)", lambda q: (q["Plateau"] - q["Y0"]) * (100 - q["PercentFast"]) / 100))),
    Model("gen_plateau_decay", "Plateau followed by one phase decay", ("X0", "Y0", "Plateau", "K"),
          _plateau_decay, "linear", family=GEN, bounds=_pos("K"), init=_i_plateau_decay,
          derived=(_tr("Half-life", "K", _half_life), _tr("Tau", "K", _tau))),
    # ---- general: polynomials
    *[Model(f"gen_poly{d}", f"{['First', 'Second', 'Third', 'Fourth'][d - 1]} order polynomial",
            tuple(f"B{i}" for i in range(d + 1)), _poly(d), "linear", family=GEN, init=_i_poly(d))
      for d in (1, 2, 3, 4)],
    # ---- general: peaks and waves
    Model("gen_gaussian", "Gaussian", ("Amplitude", "Mean", "SD"), _gauss, "linear", family=GEN, bounds=_pos("SD"),
          init=_i_gauss),
    Model("gen_lorentzian", "Lorentzian", ("Amplitude", "Center", "Width"), _lorentz, "linear", family=GEN,
          bounds=_pos("Width"), init=_i_lorentz),
    Model("gen_lognormal", "Log-normal", ("A", "GeoMean", "GeoSD"), _lognormal, "dose", family=GEN,
          bounds=(("GeoMean", 0.0, INF), ("GeoSD", 1.0, INF)), init=_i_lognormal,
          derived=(_tr("Ln(GeoMean)", "GeoMean", np.log), _tr("Ln(GeoSD)", "GeoSD", np.log))),
    Model("gen_sine", "Sine wave", ("Amplitude", "Wavelength", "PhaseShift"), _sine, "linear", family=GEN,
          bounds=(("Amplitude", 0.0, INF), ("Wavelength", 0.0, INF)),
          init=lambda s, c: _sine_scan(s, False, False), derived=(_tr("Frequency", "Wavelength", lambda w: 1.0 / w),)),
    Model("gen_sine_baseline", "Sine wave with baseline", ("Baseline", "Amplitude", "Wavelength", "PhaseShift"),
          _sine_base, "linear", family=GEN, bounds=(("Amplitude", 0.0, INF), ("Wavelength", 0.0, INF)),
          init=lambda s, c: _sine_scan(s, True, False), derived=(_tr("Frequency", "Wavelength", lambda w: 1.0 / w),)),
    Model("gen_sine_damped", "Damped sine wave", ("Amplitude", "K", "Wavelength", "PhaseShift"), _sine_damped,
          "linear", family=GEN, bounds=(("Amplitude", 0.0, INF), ("K", 0.0, INF), ("Wavelength", 0.0, INF)),
          init=lambda s, c: _sine_scan(s, False, True),
          derived=(_tr("Tau", "K", _tau), _tr("Frequency", "Wavelength", lambda w: 1.0 / w))),
    # ---- general: growth and survival
    Model("gen_gompertz", "Gompertz growth", ("Y0", "YM", "K"), _gompertz, "linear", family=GEN,
          bounds=_pos("Y0", "YM", "K"), init=_i_gompertz, derived=(_tr("Tau", "K", _tau),)),
    Model("gen_logistic", "Logistic growth", ("Y0", "YM", "k"), _logistic, "linear", family=GEN,
          bounds=_pos("Y0", "YM", "k"), init=_i_logistic),
    Model("gen_beta", "Beta growth", ("Ym", "Te", "Tm"), _beta, "linear", family=GEN,
          bounds=_pos("Ym", "Te", "Tm"), init=_i_beta),
    *[Model(f"gen_lq_survival{k}", name, ("Y0", "A", "B") if v[1] else ("A", "B"), _lq(v), "linear", family=GEN,
            bounds=_pos("A", "B"), init=_i_lq(v), derived=(("d", "Alpha/Beta (A/B)", lambda q: q["A"] / q["B"]),))
      for (k, v), name in zip(_LQ.items(), ("Linear-quadratic survival (fraction)", "Linear-quadratic survival (percent)",
                                            "Linear-quadratic survival (Y0 * survival)"))],
    Model("gen_lq_killed", "Linear-quadratic, fraction killed", ("A", "B"), _lq((1.0, False, True)), "linear",
          family=GEN, bounds=_pos("A", "B"), init=_i_lq((1.0, False, True)),
          derived=(("d", "Alpha/Beta (A/B)", lambda q: q["A"] / q["B"]),)),
]

ALL = tuple(_MODELS)
MODELS: dict[str, Model] = {m.id: m for m in _MODELS}
assert len(MODELS) == len(_MODELS), "duplicate model id"
