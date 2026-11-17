"""Forecast GTA VI launch-day Twitch viewers, hour by hour (hours 2..24).

Method (own design, tuned by leave-one-out scoring on the kit's comparables):
  * Work in log space.  The anchor is hour 1's print, but a launch's peak
    does NOT scale linearly with it: across past launches log(peak) grows
    only ~0.58x as fast as log(hour-1 print) once the game is already live
    somewhere big at 00:00 UTC, and is nearly unrelated to it when the
    game has not unlocked anywhere big yet (tiny hour-1 print).
  * Two regimes, gated smoothly on the hour-1 print (gate centred at 10k):
      - Regime A (already live, print >= ~10k): per hour, a robust
        Theil-Sen fit of log(viewers_h) on log(print_1) over the
        comparables, so each hour's level tracks the print sub-linearly.
      - Regime B (not yet unlocked, print < ~10k): per hour, the median
        absolute log level over the comparables, with a wide spread --
        at the checkpoint there is genuinely little signal about the
        eventual peak, so the honest forecast is the typical launch
        curve with wide intervals.
    The two components are blended by the gate and the spread is the
    mixture variance, so near the boundary the interval widens.  The
    blended median is pulled 10% toward flat persistence, and the
    fitted spread is shrunk 20%: both won on leave-one-out scoring.
  * Quantiles are normal quantiles on the log scale around the blended
    median, with the per-hour residual spread of the fit.  Deterministic:
    no randomness, no network, stdlib only.

Reads every *.in file in $MAKEFUN_INPUT_DIR (else /input/) and writes one
/output/<id>.out per case ($MAKEFUN_OUTPUT_DIR else /output/).
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

# Standard normal quantiles for the five levels 0.05/0.25/0.5/0.75/0.95.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# Regime gate on the hour-1 print: logistic in log space.
GATE_MID = math.log(10000.0)   # centre of the live / not-yet-live boundary
GATE_SCALE = 0.2               # log-width of the transition

# Which comparables count as "already live" when splitting the fit.
REGIME_SPLIT = 10000.0

# Fit and quantile-shape choices (tuned by leave-one-out scoring).
FIT_METHOD = "theil-sen"       # "theil-sen" (robust) or "ols"
QUANTILE_METHOD = "normal"     # "normal" or "empirical" (pooled residuals)
MEDIAN_SHRINK = 0.9            # 1.0 = fitted median as-is; <1 pulls toward persistence

# Fallbacks / floors.
MIN_FIT_POINTS = 5     # need at least this many comparables to fit a regime
MIN_SPREAD = 0.30      # never claim more precision than this (log units)
MAX_SPREAD = 2.50      # never wider than this
SPREAD_MULT = 0.8      # shrink the fitted residual spread a little
MAX_VIEWERS = 1_000_000_000.0


def _parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _day_curve(comp):
    """Release-day hourly viewers {hour 1..24: viewers} for one comparable."""
    try:
        day_start = _parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    curve = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        try:
            at = _parse_ts(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        hour = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            curve[hour] = float(viewers)
    return curve


def _theil_sen(xs, ys):
    """Robust line fit: median pairwise slope, median intercept.

    Returns (intercept, slope) for y ~ intercept + slope * x, or None
    when there are too few points.
    """
    n = len(xs)
    if n < 2:
        return None
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if dx != 0.0:
                slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        return None
    slope = _median(slopes)
    intercept = _median([y - slope * x for x, y in zip(xs, ys)])
    return intercept, slope


def _ols(xs, ys):
    """Ordinary least squares fit; returns (intercept, slope) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - slope * mx, slope


def _fit_line(xs, ys):
    if FIT_METHOD == "ols":
        return _ols(xs, ys)
    return _theil_sen(xs, ys)


def _fit_spread(xs, ys, fit):
    """Residual SD of a fit (floor 0), used as the log spread."""
    intercept, slope = fit
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    if len(resid) < 2:
        return 0.0
    mean = sum(resid) / len(resid)
    var = sum((r - mean) ** 2 for r in resid) / (len(resid) - 1)
    return math.sqrt(max(var, 0.0))


def _log_spread(values, centre):
    """SD of log values around a centre (for the regime-B median level)."""
    if len(values) < 2:
        return 0.0
    var = sum((v - centre) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(var, 0.0))


def _empirical_quantiles(values, levels):
    """Linear-interpolation quantiles of a sorted sample."""
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    out = []
    for level in levels:
        pos = level * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out.append(vals[lo] * (1.0 - frac) + vals[hi] * frac)
    return out


def forecast_case(case):
    """Return {hour: [q05, q25, q50, q75, q95]} for hours 2..24."""
    known = (case.get("known") or [{}])[0]
    print1 = float(known.get("viewers") or 0)
    if print1 < 1.0:
        print1 = 1.0
    log_p1 = math.log(print1)

    # Collect usable release-day curves from the comparables.
    curves = []
    for comp in case.get("comparables") or []:
        curve = _day_curve(comp)
        if 1 in curve and len(curve) >= 20:
            curves.append(curve)

    gate = 1.0 / (1.0 + math.exp(-(log_p1 - GATE_MID) / GATE_SCALE))
    reg_a = [c for c in curves if c[1] >= REGIME_SPLIT]
    reg_b = [c for c in curves if c[1] < REGIME_SPLIT]

    # --- Pass 1: per-hour regime fits.
    per_hour = {}
    for hour in range(2, 25):
        med_a = var_a = fit_a = None
        pts_a = [(math.log(c[1]), math.log(max(c[hour], 1.0)))
                 for c in reg_a if hour in c]
        if len(pts_a) >= MIN_FIT_POINTS:
            fit_a = _fit_line([p[0] for p in pts_a], [p[1] for p in pts_a])
            if fit_a is not None:
                med_a = fit_a[0] + fit_a[1] * log_p1
                var_a = _fit_spread([p[0] for p in pts_a],
                                    [p[1] for p in pts_a], fit_a) ** 2
        if med_a is None:
            # Too few regime-A curves: fall back to the median ratio.
            ratios = [math.log(max(c[hour], 1.0)) - math.log(max(c[1], 1.0))
                      for c in reg_a if hour in c]
            if ratios:
                med_a = log_p1 + _median(ratios)
                var_a = _log_spread(ratios, _median(ratios)) ** 2

        med_b = var_b = None
        levels_b = [math.log(max(c[hour], 1.0)) for c in reg_b if hour in c]
        if levels_b:
            med_b = _median(levels_b)
            var_b = _log_spread(levels_b, med_b) ** 2

        per_hour[hour] = (med_a, var_a, fit_a, med_b, var_b)

    # --- Pooled standardised residuals, for the empirical quantile shape.
    z_values = list(Z)
    if QUANTILE_METHOD == "empirical":
        resid = []
        for hour in range(2, 25):
            med_a, var_a, fit_a, med_b, var_b = per_hour[hour]
            if fit_a is not None and var_a and var_a > 0:
                sd = math.sqrt(var_a)
                for c in reg_a:
                    if hour in c:
                        pred = fit_a[0] + fit_a[1] * math.log(max(c[1], 1.0))
                        resid.append((math.log(max(c[hour], 1.0)) - pred) / sd)
            if med_b is not None and var_b and var_b > 0:
                sd = math.sqrt(var_b)
                for c in reg_b:
                    if hour in c:
                        resid.append((math.log(max(c[hour], 1.0)) - med_b) / sd)
        emp = _empirical_quantiles(resid, (0.05, 0.25, 0.5, 0.75, 0.95))
        if emp is not None:
            z_values = emp

    # --- Pass 2: blend the regimes and emit quantiles.
    out = {}
    for hour in range(2, 25):
        med_a, var_a, _fit_a, med_b, var_b = per_hour[hour]
        if med_a is not None and med_b is not None:
            med = gate * med_a + (1.0 - gate) * med_b
            var = (gate * var_a + (1.0 - gate) * var_b
                   + gate * (1.0 - gate) * (med_a - med_b) ** 2)
        elif med_a is not None:
            med, var = med_a, var_a
        elif med_b is not None:
            med, var = med_b, var_b
        else:
            med, var = log_p1, 0.15 ** 2

        # Pull the median a little toward flat persistence if configured.
        if MEDIAN_SHRINK != 1.0:
            med = MEDIAN_SHRINK * med + (1.0 - MEDIAN_SHRINK) * log_p1

        spread = math.sqrt(max(var, 0.0)) * SPREAD_MULT
        spread = min(max(spread, MIN_SPREAD), MAX_SPREAD)

        quantiles = []
        prev = 0.0
        for z in z_values:
            q = math.exp(med + z * spread)
            q = min(max(q, 1.0), MAX_VIEWERS)
            q = max(q, prev)  # keep non-decreasing after rounding
            quantiles.append(round(q, 2))
            prev = q
        out[hour] = quantiles
    return out


def main(argv=None):
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "*.in")))
    if not paths:
        print("no .in files in %s" % input_dir, file=sys.stderr)
        return 1
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        case_id = case.get("id") or os.path.splitext(
            os.path.basename(path))[0]
        hours = forecast_case(case)
        payload = {"hours": [{"hour": h, "q": hours[h]}
                             for h in range(2, 25)]}
        out_path = os.path.join(output_dir, "%s.out" % case_id)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
