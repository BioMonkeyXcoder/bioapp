"""Prism-style number formatting for results and data tables. Pure functions; Qt is touched only to store the preference.

Rules (4 significant digits by default, set 2-10 with set_digits):
- trailing zeros are kept (6.670, 100.0), as Prism does;
- the integer part is never rounded away (12345.678 -> 12346);
- scientific notation below 1e-4 and from 1e10 up, written 9.284e-8 / 1.235e10 (no zero padding, no "+");
- P values: 4 decimals, "<0.0001" and ">0.9999" (Prism's GP style).
"""
from __future__ import annotations

import math
import re

KEY = "display/significant_digits"
MIN_DIGITS, MAX_DIGITS, DEFAULT_DIGITS = 2, 10, 4
_digits: int | None = None


def _settings():
    from PySide6.QtCore import QSettings
    return QSettings("bioapp", "bioapp")


def digits() -> int:
    """Significant digits used by the results view (persisted in QSettings)."""
    global _digits
    if _digits is None:
        try:
            n = int(_settings().value(KEY, DEFAULT_DIGITS))
        except (TypeError, ValueError):
            n = DEFAULT_DIGITS
        _digits = min(MAX_DIGITS, max(MIN_DIGITS, n))
    return _digits


def set_digits(n: int) -> None:
    """Set and persist the significant digits; open results views pick it up on their next `reformat()`."""
    global _digits
    n = int(n)
    if not MIN_DIGITS <= n <= MAX_DIGITS:
        raise ValueError(f"significant digits must be {MIN_DIGITS}-{MAX_DIGITS}, got {n}")
    _digits = n
    _settings().setValue(KEY, n)


def _sig(v: float, n: int, keep_zeros: bool) -> str:
    if v == 0:
        return "0"
    mant, exp = f"{v:.{n - 1}e}".split("e")  # rounded to n significant digits; exp is the exponent after rounding
    e = int(exp)
    if e < -4 or e >= 10:
        s, tail = (mant, f"e{e}")
    else:
        s, tail = f"{v:.{max(0, n - 1 - e)}f}", ""
    if not keep_zeros and "." in s:
        s = s.rstrip("0").rstrip(".")
    return s + tail


def _special(v: float) -> str | None:
    if math.isnan(v):
        return "NaN"
    return ("+infinity" if v > 0 else "-infinity") if math.isinf(v) else None


def fmt_num(v: float, n: int | None = None) -> str:
    """Prism look: n significant digits (default `digits()`), trailing zeros kept."""
    v = float(v)
    return _special(v) or _sig(v, n or digits(), True)


def fmt_p(v: float) -> str:
    """Prism GP style P value."""
    v = float(v)
    if math.isnan(v):
        return "NaN"
    return "<0.0001" if v < 0.0001 else ">0.9999" if v > 0.99995 else f"{v:.4f}"


def fmt_data(v: float, sig: int = 6) -> str:
    """Data-table cell: up to `sig` significant digits, the value's own shorter form kept (0.5, 12.25)."""
    v = float(v)
    return "" if math.isnan(v) else (_special(v) or _sig(v, sig, False))


def full(v) -> str:
    """Full precision text that parses back to the same float (blank for NaN)."""
    v = float(v)
    if math.isnan(v):
        return ""
    s = repr(v)
    return s[:-2] if s.endswith(".0") else s


_RANGE = re.compile(r"^(\S+) to (\S+)$")


def fmt_range(s: str, n: int | None = None) -> str:
    """'low to high' text re-rendered like Prism (both ends in the same precision); anything else is returned unchanged."""
    m = _RANGE.match(s)
    if m:
        try:
            return f"{fmt_num(float(m[1]), n)} to {fmt_num(float(m[2]), n)}"
        except ValueError:
            pass
    return s


_P_LABEL = re.compile(r"^(adjusted\s+)?p(\s|$|\()", re.I)
_INT_LABEL = re.compile(r"^(df[nd]?|n|degrees of freedom|number of .*)$", re.I)


def is_p_label(label) -> bool:
    return isinstance(label, str) and bool(_P_LABEL.match(label.strip()))


def is_int_label(label) -> bool:
    return isinstance(label, str) and bool(_INT_LABEL.match(label.strip()))
