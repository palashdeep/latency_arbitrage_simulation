################################################################################################################################################
# Simulation of a simple latency-arbitrage scenario
# - Market fundamental value V_t follows a random walk
# - Market-maker posts quotes based on the last observed value (V_{t-1}) -> stale quotes when V jumps
# - Fast trader reacts instantly (latency = 0) and trades against stale quotes when profitable
# - Slow trader reacts after `slow_latency` steps and trades then (may be less profitable)
# This model is intentionally simple/transparent to illustrate the PnL advantage of speed.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from caas_jupyter_tools import display_dataframe_to_user

np.random.seed(42)

def simulate_latency_arbitrage(T=200000, sigma=0.5, spread=0.5, fee=0.0, slow_latency=5):
    """
    T : number of time steps
    sigma : std dev of fundamental shocks (epsilon_t)
    spread : constant spread posted by market-maker (s)
    fee : per-share transaction cost
    slow_latency : integer delay (in steps) for the slow trader
    Returns DataFrame with per-trade profits for fast and slow traders and summary stats.
    """
    # generate fundamentals V (start at 0)
    eps = np.random.normal(0, sigma, size=T)
    V = np.empty(T+1)
    V[0] = 0.0
    for t in range(1, T+1):
        V[t] = V[t-1] + eps[t-1]
    # quotes posted at time t are based on V[t-1]:
    # ask_t = V[t-1] + spread/2, bid_t = V[t-1] - spread/2
    ask = V[:-1] + spread/2
    bid = V[:-1] - spread/2

    fast_profits = []
    slow_profits = []
    fast_trades_idx = []
    slow_trades_idx = []

    # For each "event" at time t (1..T), a jump eps[t] occurs (V[t] - V[t-1])
    # Fast trader acts immediately at price ask_{t} or bid_{t} (which are based on V[t-1])
    # He profits if the jump magnitude exceeds spread/2 (so V[t] crosses the stale quote)
    for t in range(1, T+1):
        jump = V[t] - V[t-1]  # equals eps[t-1]
        # Upward move: profitable to buy at stale ask if V[t] > ask_{t-1}
        if jump > 0 and jump > spread/2:
            # Fast trader buys 1 share at ask based on V[t-1]
            fast_profit = V[t] - ask[t-1] - fee
            fast_profits.append(fast_profit)
            fast_trades_idx.append(t)
            # schedule slow trader to act at t + slow_latency (if within range)
            tau = t + slow_latency
            if tau <= T:
                slow_profit = V[tau] - ask[tau-1] - fee
                slow_profits.append(slow_profit)
                slow_trades_idx.append(tau)
        # Downward move: profitable to sell at stale bid if V[t] < bid_{t-1}
        elif jump < 0 and -jump > spread/2:
            # Fast trader sells 1 share at bid
            fast_profit = bid[t-1] - V[t] - fee  # profit from selling high and value being lower
            fast_profits.append(fast_profit)
            fast_trades_idx.append(t)
            tau = t + slow_latency
            if tau <= T:
                slow_profit = bid[tau-1] - V[tau] - fee
                slow_profits.append(slow_profit)
                slow_trades_idx.append(tau)

    # Build DataFrame of trades
    df_fast = pd.DataFrame({
        'time': fast_trades_idx,
        'profit': fast_profits
    })
    df_slow = pd.DataFrame({
        'time': slow_trades_idx,
        'profit': slow_profits
    })
    return V, df_fast, df_slow

def confidence_interval(data, alpha=0.05):
    """Compute mean and (1-alpha) confidence interval using normal approximation."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    se = std / np.sqrt(n)   # standard error
    z = 1.96  # ~95% CI
    lower = mean - z * se
    upper = mean + z * se
    return mean, (lower, upper)

# Run simulation with default parameters:
pnl_adv = []
fast_profits = []
slow_profits = []
for seed in range(50):
    V, df_fast, df_slow = simulate_latency_arbitrage(T=1000, sigma=10, spread=2, fee=0.1, slow_latency=5, seed=seed)
    fast_profits.append(df_fast['profit'].sum())
    slow_profits.append(df_slow['profit'].sum())
    pnl_adv.append(fast_profits[-1] - slow_profits[-1])

# Compute confidence intervals
fast_mean, fast_ci = confidence_interval(fast_profits)
slow_mean, slow_ci = confidence_interval(slow_profits)
adv_mean, adv_ci = confidence_interval(pnl_adv)

print("Fast trader total PnL: mean = %.2f, 95%% CI = [%.2f, %.2f]" % (fast_mean, fast_ci[0], fast_ci[1]))
print("Slow trader total PnL: mean = %.2f, 95%% CI = [%.2f, %.2f]" % (slow_mean, slow_ci[0], slow_ci[1]))
print("PnL Advantage: mean = %.2f, 95%% CI = [%.2f, %.2f]" % (adv_mean, adv_ci[0], adv_ci[1]))

# Summary statistics
def summary(df):
    return {
        'n_trades': len(df),
        'total_pnl': df['profit'].sum(),
        'mean_pnl_per_trade': df['profit'].mean() if len(df)>0 else 0.0,
        'std_pnl_per_trade': df['profit'].std() if len(df)>0 else 0.0
    }

fast_stats = summary(df_fast)
slow_stats = summary(df_slow)

summary_df = pd.DataFrame([fast_stats, slow_stats], index=['fast', 'slow']).T
# display_dataframe_to_user("Latency Arbitrage Summary", summary_df)

# Show per-trade profit distribution (histograms)
plt.figure(figsize=(8,4))
plt.hist(df_fast['profit'].clip(-2,2), bins=80)  # clip for visibility
plt.title('Fast trader per-trade profit distribution (clipped to [-2,2])')
plt.xlabel('Profit per trade')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df_slow['profit'].clip(-2,2), bins=80)
plt.title('Slow trader per-trade profit distribution (clipped to [-2,2])')
plt.xlabel('Profit per trade')
plt.ylabel('Count')
plt.show()

# Show running cumulative PnL over time (sampled)
def cumulative_pnl_over_time(df, T):
    arr = np.zeros(T+1)
    for idx, p in zip(df['time'], df['profit']):
        arr[idx] += p
    return np.cumsum(arr)

cum_fast = cumulative_pnl_over_time(df_fast, len(V)-1)
cum_slow = cumulative_pnl_over_time(df_slow, len(V)-1)

plt.figure(figsize=(10,4))
plt.plot(cum_fast, label='fast')
plt.plot(cum_slow, label='slow')
plt.legend()
plt.title('Cumulative PnL over time (fast vs slow)')
plt.xlabel('time step')
plt.ylabel('Cumulative PnL')
plt.show()

# Provide numeric summary as a small DataFrame
numeric_summary = pd.DataFrame({
    'metric': ['n_trades', 'total_pnl', 'mean_pnl_per_trade', 'std_pnl_per_trade'],
    'fast': [fast_stats['n_trades'], fast_stats['total_pnl'], fast_stats['mean_pnl_per_trade'], fast_stats['std_pnl_per_trade']],
    'slow': [slow_stats['n_trades'], slow_stats['total_pnl'], slow_stats['mean_pnl_per_trade'], slow_stats['std_pnl_per_trade']]
})
# display_dataframe_to_user("Numeric results (fast vs slow)", numeric_summary)