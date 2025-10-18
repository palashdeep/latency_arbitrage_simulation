import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Iterable, Union, Optional

_sqrt2 = math.sqrt(2.0)

def get_stock_price_random_walk(V0, t, sigma=1):
    """ Simulate stock price with a random walk """
    return V0 + np.cumsum(np.random.normal(0, sigma, t))

# Show running cumulative PnL over time (sampled)
def cumulative_pnl_over_time(df, T):
    arr = np.zeros(T+1)
    for idx, p in zip(df['time'], df['profit']):
        arr[idx] += p
    return np.cumsum(arr)

def ci_mean(arr, alpha= 0.05):
    """Return mean and (lower, upper) 95% CI (normal approx)."""
    a = np.asarray(arr, dtype=float)
    n = len(a)
    if n == 0:
        return 0.0, (0.0,0.0)
    m = float(a.mean())
    s = float(a.std(ddof=1)) if n > 1 else 0.0
    se = s / math.sqrt(n) if n > 1 else 0.0
    z = norm.ppf(1 - alpha / 2)  # approx for 95%
    return m, (m - z*se, m + z*se)

def aggregate_switch(signals,
                     method="ema",
                     m=10,
                     n=None,
                     span_for_ema=None):
    """
    Aggregate signals with the following behavior:
      - If len(signals) < n: raise ValueError
      - If len(signals) == m: return a single aggregate value as a one-element list [float]
      - If len(signals) > n: return a causal list of aggregates (same length as signals).
    
    Parameters
    ----------
    signals : iterable of floats
        Input signal sequence (ordered in time).
    method : {"ema", "sma", "median"}
        Aggregation method. Default "ema".
    m : int
        Window length used when computing a single aggregate (when len(signals) == m),
        and used as window/span parameter for SMA/median. For EMA, `span_for_ema` overrides it.
    n : int or None
        Minimum length required to proceed. If None, defaults to m.
        If len(signals) < n -> ValueError.
    span_for_ema : int or None
        Explicit span for EMA (if provided). If None, span = m is used.
    
    Returns
    -------
    list of floats
        - One-element list [value] when len(signals) == m
        - List of length len(signals) for causal aggregates when len(signals) > n
    """
    arr = np.asarray(list(signals), dtype=float)
    T = len(arr)

    if n is None:
        n = m
    if n < 1 or m < 1:
        raise ValueError("m and n must be positive integers.")
    if n < m:
        raise ValueError("Require n >= m to avoid ambiguous behavior. (n must be >= m)")
    if T < n:
        raise ValueError(f"Need at least n={n} signals; got {T}.")

    method = method.lower()
    if method not in ("ema", "sma", "median"):
        raise ValueError("method must be 'ema', 'sma', or 'median'")

    # Case: exact m -> single aggregate value (return [float])
    if T == m:
        if method == "sma":
            return [float(np.mean(arr))]
        elif method == "median":
            return [float(np.median(arr))]
        else:  # EMA
            span = span_for_ema if span_for_ema is not None else m
            alpha = 2.0 / (span + 1.0)
            s = arr[0]
            for i in range(1, m):
                s = alpha * arr[i] + (1 - alpha) * s
            return [float(s)]

    # Case: T > n -> return causal list of aggregates
    if method == "sma":
        rolled = pd.Series(arr).rolling(window=m, min_periods=1).mean().to_list()
        return rolled
    elif method == "median":
        rolled = pd.Series(arr).rolling(window=m, min_periods=1).median().to_list()
        return rolled
    else:  # EMA
        span = span_for_ema if span_for_ema is not None else m
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(arr)
        s = arr[0]
        out[0] = s
        for t in range(1, T):
            s = alpha * arr[t] + (1 - alpha) * s
            out[t] = s
        return out.tolist()

def estimate_QR_from_signals(signals, ema_span=10, floor=1e-6, clip=10.0):
    """
    Estimate process variance (Q) and signal variance (R) directly from signals.
    
    signals : array-like
        Time series of observed signals (float).
    ema_span : int
        Span for exponential moving average (used as 'true' trend proxy).
    floor : float
        Minimum variance to avoid zeros.
    clip : float
        Winsorization cap in std devs for outlier control.
    """

    signals = np.asarray(signals)

    # --- Step 1: Process variance (Q) ---
    diffs = np.diff(signals)
    med, mad = np.median(diffs), np.median(np.abs(diffs - np.median(diffs)))
    diffs_clipped = np.clip(diffs, med - clip*mad, med + clip*mad)
    Q_hat = max(np.var(diffs_clipped), floor)

    # --- Step 2: Signal variance (R) ---
    smoothed = pd.Series(signals).ewm(span=ema_span, adjust=False).mean().values
    residuals = signals - smoothed
    med_r, mad_r = np.median(residuals), np.median(np.abs(residuals - np.median(residuals)))
    residuals_clipped = np.clip(residuals, med_r - clip*mad_r, med_r + clip*mad_r)
    R_hat = max(np.var(residuals_clipped), floor)

    return {
        'process_var': Q_hat,   # variance of the random walk increments
        'signal_var': R_hat      # variance of the observation noise
    }

def _std_normal_cdf(x):
    # stable standard normal CDF using erf
    return 0.5 * (1.0 + math.erf(x / _sqrt2))

def estimate_probability(score,
                         confidence,
                         spread,
                         process_var,
                         signal_var,
                         floor_var=1e-12):
    """
    Estimate P( directional move will cross the stale quote ).

    Parameters
    ----------
    score : float
        Expected immediate move (e.g. S - mid). Directional: positive => upward.
    confidence : float in (0,1]
        Confidence weight: 1.0 means fully confident; smaller -> more uncertain.
        Used to inflate variance as var_eff = (process_var + signal_var) / confidence.
    spread : float
        Current spread (s). Threshold to cross = spread / 2.
    process_var : float
        Estimated process variance (per-step).
    signal_var : float
        Estimated signal (measurement) variance.
    floor_var : float
        Minimum variance to avoid division by zero.

    Returns
    -------
    prob : float in [0,1]
        Estimated probability that the (signed) move exceeds the stale quote.
    """
    # total model variance
    var_total = float(process_var) + float(signal_var)
    # floor and apply confidence scaling
    var_total = max(var_total, floor_var)
    # interpret confidence as reliability in (0,1]; avoid zero division
    conf = max(min(confidence, 1.0), 1e-6)
    var_eff = var_total / conf
    std_eff = math.sqrt(var_eff)

    # threshold in the direction of the score
    thresh = abs(spread) / 2.0

    # If score = 0, symmetric: prob that |ΔV| > thresh is 2*(1 - Phi(thresh/std))
    if abs(score) < 1e-16:
        # two-sided exceedance
        z = thresh / std_eff
        tail = 1.0 - _std_normal_cdf(z)
        prob = 2.0 * tail
        return float(min(max(prob, 0.0), 1.0))

    # directional one-sided probability
    # compute z = (threshold - |score|) / std
    z = (thresh - abs(score)) / std_eff
    # P(|ΔV| > thresh in direction of sign(score)) = 1 - Phi(z)
    # (if |score| >> thresh -> z negative large -> Phi(z) ~0 -> prob ~1)
    prob = 1.0 - _std_normal_cdf(z)
    # clip
    prob = float(min(max(prob, 0.0), 1.0))
    return prob


def expected_impact(size,
                    spread,
                    depth=1000,
                    impact_coeff=0.1,
                    base=None,
                    latency_steps=0,
                    latency_scale=10):
    """
    Estimate expected slippage/adverse selection cost (impact) per trade.

    Parameters
    ----------
    size : float
        Trade size (units).
    spread : float
        Current bid-ask spread.
    depth : float
        Available market depth at best quotes.
    impact_coeff : float
        Sensitivity constant (~0.05-0.2 typical).
    base : float or None
        Fixed minimum impact (defaults to half the spread).
    latency_steps : int
        Trader latency in simulation time steps.
    latency_scale : float
        Scaling for latency penalty (impact grows roughly linearly with latency/scale).

    Returns
    -------
    impact : float
        Expected impact cost (in price units).
    """
    if base is None:
        base = spread / 2.0  # minimum cost to cross the spread

    # baseline linear impact
    linear_component = impact_coeff * (size / max(depth, 1e-9))

    # latency penalty: slower traders suffer more (scaled)
    latency_penalty = base * (latency_steps / max(latency_scale, 1e-9))

    impact = base + linear_component + latency_penalty
    return impact