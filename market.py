import uuid
import pandas as pd
import numpy as np

class market:
    def __init__(self):
        """
        mm : Market Maker object with method get_quote(V, t) and execute_market_order(side, size)
        """
        self.order_book = []  # list of pending market orders

    def submit_market_order(self, t, trader, mm_quote, side, size):
        """
        side : "buy" or "sell"
        size : float
        Executes a market order against the market maker's current quote.
        Returns a dict with keys:
            - filled: float
            - avg_price: float
            - mm_cash_change: float
            - mm_inventory_change: float
        """
        order_id = str(uuid.uuid4())

        self.order_book.append({
            "order_id": order_id,
            "arrival_time": t,
            "trader": trader.name,
            "side": side,
            "size": size,
            "latency": trader.lag,
            "quote": mm_quote
        })

        return order_id

    def clear_market_orders(self, market_maker):
        """
        orders: list of dicts with keys:
          - trader: Trader object
          - side: "buy" or "sell"
          - size: float
          - latency: float (lower is earlier)
          - order_id: optional id
        Orders are processed for the current self.time using MM.get_quote(V, time).
        """
        orders = self.order_book
        self.order_book = []

        # Add tiny jitter to break ties; sort by latency then jitter
        for o in orders:
            o["_sort_key"] = o.get("latency", 0.0) + np.random.normal(0, 1e-6)

        orders_sorted = sorted(orders, key=lambda x: x["_sort_key"])

        fills = []
        for o in orders_sorted:
            trader = o["trader"]
            side = o["side"]
            size = o["size"]
            # Execute against MM
            fill = market_maker.execute_market_order(side, size)
            
            fills.append({
                "order_id": o["order_id"],
                "trader": trader,
                "side": side,
                "requested": size,
                "filled": fill["filled"],
                "avg_price": fill["avg_price"],
            })

        return fills