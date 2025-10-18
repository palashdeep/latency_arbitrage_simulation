import numpy as np
import pandas as pd
from utils import estimate_probability, estimate_QR_from_signals, expected_impact

class fast_Trader:

    def __init__(self, latency=5, noise_std=0.5, base_size=10, max_inventory=100, thresholds={"prob":0.3, "EV_min": 0.5}, name="fast"):
        self.lag = latency
        self.noise_std = noise_std
        self.base_size = base_size
        self.inventory = 0
        self.cash = 0
        self.max_inventory = max_inventory
        self.thresholds = thresholds    # dict {prob, EV}
        self.name = name
        self.lambda_inv = 1e-4  # inventory penalty coefficient
        self.estimated_vars = {
            'process_var': 0.2**2,   # variance of the random walk increments
            'signal_var': 0.5**2      # variance of the observation noise
        }
        self.signals = []

    def _inventory_ok(self, side, size):
        """checks if inventory limits are respected"""
        sign = 1 if side=="buy" else -1
        new_inv = self.inventory + sign*size
        return abs(new_inv) <= self.max_inventory

    def observe(self, V, t):
        """Observe new signal (mid-price with noise)"""
        eps = np.random.normal(0, self.noise_std)
        signal = V[t-self.lag] + eps
        self.signals.append(signal)
        return signal
    
    def get_QR(self):
        """Estimate process and signal variances from collected signals"""
        QR = estimate_QR_from_signals(self.signals, ema_span=10, floor=1e-6, clip=10.0)
        return QR

    def compute_score(self, mm_quote, estimated_vars):
        # simplest: expected move = S - mid
        mid = (mm_quote[0] + mm_quote[1]) / 2
        spread = mm_quote[1] - mm_quote[0]
        score = self.signals[-1] - mid
        confidence = 1.0 / (estimated_vars['signal_var'] + 1e-9)
        prob = estimate_probability(score, confidence, spread, estimated_vars['process_var'], estimated_vars['signal_var'])
        return score, confidence, prob    

    def sizing(self, confidence):
        """size based on confidence and inventory limits"""
        size = self.base_size * confidence
        if abs(self.inventory + size) > self.max_inventory:
            size = self.max_inventory - abs(self.inventory)
        return int(size)

    def decide_and_order(self, t, V, market, mm_quote, fees=0.01):
        """Decide whether to submit market order based on computed metrics"""
        signal = self.observe(V, t)
        estimated_vars = self.get_QR()
        score, confidence, prob = self.compute_score(mm_quote, estimated_vars)
        size = self.sizing(confidence)
        impact = expected_impact(size, spread=mm_quote[1]-mm_quote[0], depth=1000, impact_coeff=0.05, latency_scale=self.lag)
        EV = prob * abs(score) - fees - impact
        EV_adj = EV - self.lambda_inv * (self.inventory**2)
        side = "buy" if np.sign(score) else "sell"
        if EV_adj > self.thresholds['EV_min'] and self._inventory_ok(side, size):
            # aggressive market order to pick stale quote
            order_id = market.submit_market_order(t, self, mm_quote, side=side, size=size)
            return order_id
        return None

    def apply_fill(self, side: str, filled: float, avg_price: float):
        # side refers to trader's action: 'buy' means trader bought and now holds inventory
        if side == "buy":
            self.inventory += filled
            self.cash -= filled * avg_price
        else:
            self.inventory -= filled
            self.cash += filled * avg_price

    def mark_to_market(self, V, t):
        """Current PnL = cash + inventory * true value"""
        return self.cash + self.inventory * V[t]