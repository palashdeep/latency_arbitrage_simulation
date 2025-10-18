import numpy as np
from utils import aggregate_switch, estimate_QR_from_signals, estimate_probability, expected_impact

class slow_Trader:

    def __init__(self, latency=20, noise_std=0.3, base_size=10, max_inventory=100, thresholds={"prob":0.5, "EV_min": 0.6}, agg_window=10, name="slow"):
        self.lag = latency
        self.noise_std = noise_std
        self.base_size = base_size
        self.agg_window = agg_window
        self.inventory = 0
        self.cash = 0
        self.max_inventory = max_inventory
        self.thresholds = thresholds    # dict {prob, EV}
        self.name = name
        self.lambda_inv = 1e-4  # inventory penalty coefficient
        self.signals = []

    def _inventory_ok(self, side, size):
        sign = 1 if side=="buy" else -1
        new_inv = self.inventory + sign*size
        return abs(new_inv) <= self.max_inventory

    def observe(self, V, t):
        """Observe new signal (mid-price with noise)"""
        eps = np.random.normal(0, self.noise_std)
        self.signals.append(V[t-self.lag] + eps)
        if t < self.agg_window:
            return None
        signal = aggregate_switch(self.signals[-self.agg_window:])[0] + eps
        return signal
    
    def get_QR(self):
        """Estimate process and signal variances from collected signals"""
        QR = estimate_QR_from_signals(aggregate_switch(self.signals), ema_span=10, floor=1e-6, clip=10.0)
        return QR

    def compute_score(self, mm_quote, estimated_vars):
        # use aggregated S and a more conservative variance estimate
        mid = (mm_quote[0] + mm_quote[1]) / 2
        spread = mm_quote[1] - mm_quote[0]
        score = self.signals[-1] - mid
        confidence = 1.0 / np.var(self.signals[-self.agg_window:])  # more conservative
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
        if not signal:
            return None
        estimated_vars = self.get_QR()
        score, confidence, prob = self.compute_score(mm_quote, estimated_vars)
        size = self.sizing(confidence)
        impact = expected_impact(size, spread=mm_quote[1]-mm_quote[0], depth=500, impact_coeff=0.2, latency_scale=self.lag)
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