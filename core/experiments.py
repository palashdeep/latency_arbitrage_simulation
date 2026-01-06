"""
This module runs the main experiment for this project: a latency sweep

We simulate a market with a single price process, na naive market maker posting quotes based on lagged information,
and two traders following the same rational decision rule but differening in when they observe the incoming information.

By varying slow trader's information latency over different paths, we quantify how expected PnL decays as information becomes stale.
This experiment isolates the economic value of speed by holding everything else constant.
"""

import numpy as np
import pandas as pd

from core.model import LatentValueModel
from core.market import NaiveMarketMaker
from core.traders import FastTrader, SlowTrader

def run_single_simulation(T, sigma, spread, base_depth, impact_coeff, slow_latency, agg_window, seed):
    """Run one simulation and return final PnLs"""

    model = LatentValueModel(T=T, sigma=sigma, seed=seed)

    mm = NaiveMarketMaker(
        lag=2,
        spread=spread,
        base_depth=base_depth,
        impact_coeff=impact_coeff,
        start_price=model.V[0]
    )

    fast = FastTrader(
        name="Fast",
        latency=0,
        noise_std=0.01,
        base_size=1.0,
        max_inventory=10,
        prob_threshold=0.3,
        ev_threshold=1.0,
        impact_coeff=impact_coeff
    )

    slow = SlowTrader(
        name="Slow",
        latency=slow_latency,
        noise_std=0.02,
        base_size=1.0,
        max_inventory=10,
        prob_threshold=0.5,
        ev_threshold=1.0,
        impact_coeff=impact_coeff,
        agg_window = agg_window
    )

    total_variance =  sigma**2

    for t in range(1, T):

        quote = mm.get_quote(model.V, t)

        fast.observe(model.V, t)
        slow.observe(model.V, t)

        orders = []
        o_fast = fast.decide_and_order(quote, total_variance)
        if o_fast:
            orders.append(o_fast)

        o_slow = slow.decide_and_order(quote, total_variance)
        if o_slow:
            orders.append(o_slow)

        orders.sort(key=lambda x: x["latency"])

        for o in orders:
            res = mm.execute_market_order(o["side"], o["size"])
            o["trader"].apply_fill(o["side"], res["filled"], res["avg_price"])

        mm.end_of_timestamp_update(rho=0.1)
        slow.end_of_timestamp_update(quote)
        fast.end_of_timestamp_update(quote)

    final_price = model.V[-1]

    return {
        "Fast_PnL": fast.mark_to_market(final_price),
        "Slow_PnL": slow.mark_to_market(final_price),
        "MM_PnL": mm.mark_to_market(final_price),
    }

def latency_sweep(latencies, n_runs, **sim_kwargs):
    """Run Monte Carlo latency sweep and return summary df"""

    rows = []

    for latency in latencies:
        fast_pnls = []
        slow_pnls = []
        mm_pnls = []

        for seed in range(n_runs):
            res = run_single_simulation(
                slow_latency=latency,
                seed=seed,
                **sim_kwargs
            )

            fast_pnls.append(res["Fast_PnL"])
            slow_pnls.append(res["Slow_PnL"])
            mm_pnls.append(res["MM_PnL"])

        fast_pnls = np.array(fast_pnls)
        slow_pnls = np.array(slow_pnls)
        mm_pnls = np.array(mm_pnls)

        rows.append({
            "latency": latency,
            "fast_mean": fast_pnls.mean(),
            "fast_std": fast_pnls.std(),
            "slow_mean": slow_pnls.mean(),
            "slow_std": slow_pnls.std(),
            "mm_mean": mm_pnls.mean(),
            "advantage_mean": (fast_pnls - slow_pnls).mean(),
        })

    return pd.DataFrame(rows)