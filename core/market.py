import numpy as np

class NaiveMarketMaker:
    """
    Naive Market Maker:
    - quotes around lagged true value
    - fixed spread
    - finite depth
    """

    def __init__(self, lag, spread, base_depth, impact_coeff, start_price=100.0):
        self.lag = lag
        self.spread = spread
        self.impact_coeff = impact_coeff
        
        self.base_depth = base_depth
        self.depth_bid = base_depth
        self.depth_ask = base_depth

        self.cash = 0.0
        self.inventory = 0

        self.last_mid = start_price

    def get_quote(self, V, t):
        """Quotes based on lagged true value"""
        idx = max(0, t - self.lag)
        self.last_mid = V[idx]
        
        bid = self.last_mid - self.spread / 2.0
        ask = self.last_mid + self.spread / 2.0

        return (bid, ask, self.depth_bid, self.depth_ask)
    
    def execute_market_order(self, side, size):
        """Execute market order against quotes"""

        bid = self.last_mid - self.spread / 2.0
        ask = self.last_mid + self.spread / 2.0

        filled = 0.0
        cash_change = 0.0
        inv_change = 0

        if side == "buy":
            take = min(size, self.depth_ask)
            filled += take
            cash_change += take * ask
            inv_change -= take
            self.depth_ask -= take

            remaining = size - take
            if remaining > 0:
                penalty = self.impact_coeff * remaining
                price = ask + penalty
                filled += remaining
                cash_change += remaining * price
                inv_change -= remaining

        else:
            take = min(size, self.depth_bid)
            filled += take
            cash_change -= take * bid
            inv_change += take
            self.depth_bid -= take

            remaining = size - take
            if remaining > 0:
                penalty = self.impact_coeff * remaining
                price = bid - penalty
                filled += remaining
                cash_change -= remaining * price
                inv_change += remaining

        self.cash += cash_change
        self.inventory += inv_change

        avg_price = cash_change / filled if filled > 0 else 0.0

        return {
            "filled": filled,
            "avg_price": avg_price,
        }
    
    def end_of_timestamp_update(self, rho):
        """Simple exponential recovery of depth"""
        self.depth_bid += rho * (self.base_depth - self.depth_bid)
        self.depth_ask += rho * (self.base_depth - self.depth_ask)

    def mark_to_market(self, price):
        """Current PnL = cash + inventory * true value"""
        return self.cash + self.inventory * price