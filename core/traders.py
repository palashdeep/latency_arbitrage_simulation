import numpy as np
import scipy.stats as stats

class BaseTrader:
    """
    Base Trader implementing single rational decision logic
    Traders differ only by:
    - latency
    - signal aggregation logic
    """
    def __init__(self, name, latency, noise_std, base_size, max_inventory, prob_threshold, ev_threshold, impact_coeff, seed=None):
        self.name = name
        self.latency = latency
        self.noise_std = noise_std
        self.base_size = base_size
        self.max_inventory = max_inventory
        self.prob_threshold = prob_threshold
        self.ev_threshold = ev_threshold
        self.impact_coeff = impact_coeff

        self.inventory = 0
        self.cash = 0

        self.rng = np.random.default_rng(seed or 89)

        self.signals = []
    
    def observe(self, V, t):
        """Observe delayed noisy signal"""
        idx = max(0, t - self.latency)
        eps = self.rng.normal(0.0, self.noise_std)
        signal = V[idx] + eps
        self.signals.append(signal)
    
    def aggregate(self):
        """Aggregation of observer signals"""
        return self.signals[-1] # default: no aggregation (FastTrader)
    
    def compute_score(self, mid_price):
        """Score = belief - mid"""
        belief = self.aggregate()
        return belief - mid_price
    
    def compute_probability(self, score, spread, total_var):
        """
        Probability that abs move exceeds half spread
        Gaussian approximation
        """
        if total_var <= 0:
            return 0.0
        
        sigma = np.sqrt(total_var)
        threshold = spread / 2.0

        z = (threshold - abs(score)) / sigma
        return float(1.0 - stats.norm.cdf(z))
    
    def expected_impact(self, size, depth):
        """Linear impact model"""
        return self.impact_coeff * (size / max(depth, 1))
    
    def inventory_ok(self, side, size):
        """Inventory constraint"""
        sign = 1 if side == "buy" else -1
        new_inv = self.inventory + sign * size
        return abs(new_inv) <= self.max_inventory
    
    def decide_and_order(self, quote, total_var):
        """Decision rule: Trade if EV > threshold"""
        if not self.signals:
            return None
        
        bid, ask, depth_bid, depth_ask = quote
        mid = (bid + ask) / 2
        spread = ask - bid

        score = self.compute_score(mid)
        side = "buy" if score > 0 else "sell"

        depth = depth_ask if side == "buy" else depth_bid
        size = min(self.base_size, self.max_inventory - abs(self.inventory))

        if size <= 0 or not self.inventory_ok(side, size):
            return None
        
        if abs(score) < 1.5 * np.sqrt(total_var):
            return None
        
        prob = self.compute_probability(score, spread, total_var)
        impact = self.expected_impact(size, depth)

        EV = prob * abs(score) - impact
        if prob < self.prob_threshold or EV < self.ev_threshold:
            return None
        
        return {
            "trader":self, 
            "side":side, 
            "size":size, 
            "latency":self.latency,
        }
    
    def apply_fill(self, side, filled, price):
        """Accounting for filled orders"""
        if filled <= 0:
            return
        
        if side == "buy":
            self.inventory += filled
            self.cash -= filled * price
        else:
            self.inventory -= filled
            self.cash += filled * price

    def end_of_timestamp_update(self, quote):
        """Forces liquidation at mid at the end of each timestep"""
        price = 0.5 * (quote[0] + quote[1])
        self.cash += self.inventory * price
        self.inventory = 0.0
    
    def mark_to_market(self, price):
        """Current PnL = cash + inventory * price"""
        return self.cash + self.inventory * price

class FastTrader(BaseTrader):
    """
    Fast Trader:
    - minimal latency
    - no aggregation (uses most recent signal)
    """

    pass
    
class SlowTrader(BaseTrader):
    """
    Slow Trader:
    - delayed information
    - aggregates over a window
    """

    def __init__(self, *args, agg_window, **kwargs):
        super().__init__(*args, **kwargs)
        self.agg_window = agg_window

    def aggregate(self):
        """Aggregation of observer signals"""
        window = self.signals[-self.agg_window:]
        return float(np.mean(window))