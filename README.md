# Project: Latency Arbitrage Simulation Framework

## Overview

This project studies the **economic value of speed** in electronic markets.

I build a minimal agent-based model in which multiple traders observe the same underlying value process with different delays and compete against a market maker posting quotes based on imperfect information. All agents follow the same rational decision rule; the only difference is when they see the information.

The goal is to isolate when and why execution latency creates real trading advantage, and when that advantage disappears.

## Core Question

> When does speed create economic value; and when it does not?

More precisely:
- How does expected PnL decay as a trader's information or execution latency increases?
- How do market maker updates, spreads and liquidity dynamics limit latency arbitrage?
- Can smarter quoting or batching neutralize speed advantages?

## Model

### Latent Value

A single "true value" process

$$ V_t = V_{t-1} + \epsilon_t, \\quad \epsilon_t \sim N(0, \sigma^2) $$

### Market Maker

- Posts bid/ask quotes around a lagged estimate of the latent value
- Earns the spread but faces adverse selection from faster traders
- Maintains finite bid and ask depth with stochastic refill

Two variants
- **Naive MM:** quotes based on lagged value with fixed spread
- **Smart MM (extension):** adaptive mid and spread using EWMA volatility estimation

### Traders

Two traders with identical logic but different information timing
- **Fast Trader**
    - Observes signals with minimal delay
    - Acts immediately on fresh information
- **Slow Trader**
    - Observes delayed or aggregated signals
    - Faces adverse selection and reduced expected value

For both type of traders, signals take the form:

$$ S_{t}^{(i)} = V_{t-\ell_i} + \eta_t $$

## Decision Rule

Both traders use the same rational rule:

$$ Trade \\quad if \mathbb{E}[\Delta V \mid S_t] > \frac{spread}{2} + impact $$

- Speed does not change intelligence
- Speed changes the information set
- Latency directly reduces conditional expected value

## Experiments

The main experiment is latency sweep:
- Vary trader latency $\ell$
- Run Monte Carlo simulations
- Measure:
    - Mean PnL
    - 95% confidence intervals
    - Distribution of outcomes

Key outputs:
- PnL vs Latency curves
- Demonstration of alpha decay with delay
- Identification of regimes where latency arbitrage disappears

## Results and Insights

- Latency advantage decays rapidly as information becomes stale
- Beyond a critical delay, expected PnL becomes negative
- Adaptive market making (wider spreads, faster updates) sharply reduces arbitrage
- Liquidity depth and impact costs dominate once speed advantage vanishes
- Batch-style execution significantly compress latency-based edge

### Conclusion
Speed only create when information arrives before price adjust. Otherwise, imapct and adverse selection dominate.

## Repo Structure

```graphql
core/
  model.py        # Latent value + execution loop
  market.py       # Market maker and execution logic
  traders.py      # Fast and slow trader definitions
  experiments.py  # Latency sweep and statistics

research/
  smart_mm.py     # Adaptive quoting extensions
  batching.py     # Auction / batching experiments
  depth_models.py # Alternative liquidity dynamics
```
The `core/` folder contains a minimal self-contained model
The `research/` folder explores extensions and robustness checks

## Project Relevance

This project emphasizes:
- Decision making under uncertainty
- Information timing and conditioning
- Market microst

## Possible Extensions

- Multi-agent competition among fast traders
- Endogenous spread optimization
- Queue position and limit order modelling
- Continuos time limits
- Empirical calibration to real tick data

## Takeaway

- Speed creates value only by changing the information set, not the strategy
- Latency rapidly destroys conditional expected value
- Adaptive market making and impact costs tightly bound the value of speed
