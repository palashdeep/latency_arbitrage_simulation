import numpy as np
from scipy.stats import norm

class smart_MM:

    def __init__(self, lag=10, spread=5, k_sigma=0.5, k_I=0.05, freq=3, beta=0.7, imbalance_window=5, inventory=100, gamma=0.05, cash=0, start=100.0):
        self.lag = lag
        self.spread = spread
        self.freq = freq
        self.inventory = inventory
        self.signals = [start]
        self.mids = []
        self.estimates = []
        self.quotes = [(start - spread/2, spread + spread/2)]
        self.cash = cash
        self.orders = []
        self.V_hats = []
        self.var_hats = []
        self.k_sigma = k_sigma
        self.k_I = k_I
        self.gamma = gamma
        self.beta = beta
        self.imbalance_window = imbalance_window
        self.ewma_sigma = 0.2
        self.signal_var_tick = 0.02  # variance of signal per tick
        self.signal_var_noise = self.signal_var_tick * 5  # variance of signal noise
        self.process_var = 0.2**2  # variance of the random walk increments
        self.depth_at_best = 1e6
        self.base_depth = 100
        self.depth_bid = self.base_depth
        self.depth_ask = self.base_depth
        self.impact_per_unit = 0.1  # price impact per unit size beyond depth
        self.rho = 0.05            # refill fraction per timestep
        self.lambda_limit = 0.3
        self.avg_limit_size = 5.0
        self.min_depth = 1e-3

    def robust_calibrate_from_kalman(self, t, winsor_alpha=0.01, floor=1e-10, use_mad=False):
        """
        Robust calibration of Q (process variance) and R (measurement variance)
        using Kalman filter internals.

        Inputs:
        S           : array-like, observed signals (length T)
        pred_mean   : array-like, predicted means hat{V}_{t|t-1} (length T)
        P_pred      : array-like, predicted variances P_{t|t-1} (length T)
        P_upd       : array-like, posterior variances P_{t|t} (length T)
        winsor_alpha: fraction for winsorization on innovations (default 1%)
        floor       : minimum variance floor
        use_mad     : if True, use MAD-based robust variance for nu instead of var

        Returns:
        dict with keys:
            R_hat, Q_hat,
            diagnostics: dict holding var_nu, mean_P_pred, raw_R, raw_Q, n_samples
        Notes:
        Arrays must be same length T and aligned so pred_mean[t] corresponds to the prior
        used to update with observation S[t], and similarly P_pred[t] is prior var at t.
        """
        S = np.asarray(self.signals)
        T = len(S)
        pred_mean = np.asarray(self.V_hats[:T])
        P_pred = np.asarray(self.var_hats[:T])
        P_upd = np.asarray(self.var_hats[1:])
        assert T == len(pred_mean) == len(P_pred) == len(P_upd), "All inputs must have same length T"

        # 1) Innovations: nu_t = S_t - pred_mean_t
        nu = S - pred_mean  # length T

        # 2) Winsorize innovations to reduce influence of outliers
        lo, hi = np.quantile(nu, [winsor_alpha, 1 - winsor_alpha])
        nu_w = np.clip(nu, lo, hi)

        # 3) Robust variance estimate for innovations
        if use_mad:
            # convert MAD to variance approx: var ~ (1.4826 * MAD)^2 for normal-like
            mad = np.median(np.abs(nu_w - np.median(nu_w)))
            nu_var = (1.4826 * mad) ** 2
        else:
            nu_var = np.var(nu_w, ddof=1)

        mean_P_pred = np.mean(P_pred)

        # 4) Estimate R: Var(nu) - mean(P_pred)
        raw_R = nu_var - mean_P_pred
        R_hat = max(raw_R, floor)

        # 5) Estimate Q from P_pred and P_upd:
        #    P_pred[t] = P_upd[t-1] + Q  => Q ≈ mean( P_pred[1:] - P_upd[:-1] )
        if T < 2:
            raw_Q = floor
        else:
            diffs = P_pred[1:] - P_upd[:-1]
            raw_Q = np.mean(diffs)
        Q_hat = max(raw_Q, floor)

        # Lag-1 returns
        r1 = S[1:] - S[:-1]
        v1 = np.var(r1, ddof=1)
        
        # Lag-2 returns
        r2 = S[2:] - S[:-2]
        v2 = np.var(r2, ddof=1)
        
        # Solve system
        Q_hat_raw = v2 - v1
        R_hat_raw = 0.5 * (v1 - self.process_var)

        pct_Q = abs(Q_hat - Q_hat_raw)/max(Q_hat,1e-12)
        pct_R = abs(R_hat - R_hat_raw)/max(R_hat,1e-12)
        ratio_Q = max(Q_hat,Q_hat_raw)/max(min(Q_hat,Q_hat_raw),1e-12)
        ratio_R = max(R_hat,R_hat_raw)/max(min(R_hat,R_hat_raw),1e-12)

        status = "OK"
        
        if pct_Q > 0.5 or pct_R > 0.5 or ratio_Q > 2 or ratio_R > 2:
            status = "FAILED"
        elif pct_Q > 0.2 or pct_R > 0.2:
            status = "WARN"
        
        if status != "FAILED":
            self.process_var = Q_hat
            self.signal_var_tick = R_hat
            self.signal_var_noise = self.signal_var_tick * 5

        print("Calibration Status: %s | Q: %.4f | R: %.4f " % (status, self.process_var, self.signal_var_tick))

    def calibrate_variances(self):
        """
        Calibrate process variance and signal variance
        from tick data (mid-price series).
        """
        prices = np.array(self.signals)
        
        # Lag-1 returns
        r1 = prices[1:] - prices[:-1]
        v1 = np.var(r1, ddof=1)
        
        # Lag-2 returns
        r2 = prices[2:] - prices[:-2]
        v2 = np.var(r2, ddof=1)
        
        # Solve system
        self.process_var = v2 - v1
        self.signal_var_tick = 0.5 * (v1 - self.process_var)

    def estimator_update(self, signal):
        self.signals.append(signal)
        lambda_ewma = 0.98  # forgetting factor
        threshold_mult = 3.0  # outlier detection threshold
        # signal_var = get_signal_var(signal)
        V_hat_prev = self.V_hats[-1] if len(self.V_hats) > 0 else self.signals[0]
        var_hat_prev = self.var_hats[-1] if len(self.var_hats) > 0 else 1.0
        nu = signal - V_hat_prev
        if abs(nu) > threshold_mult * np.sqrt(var_hat_prev):
            # outlier detected, increase uncertainty
            signal_var = self.signal_var_noise
        else:
            signal_var = self.signal_var_tick

        inst = nu**2
        self.ewma_sigma = lambda_ewma * self.ewma_sigma + (1 - lambda_ewma) * inst
        signal_var = max(signal_var, 0.1 * self.ewma_sigma)
        K = var_hat_prev / (var_hat_prev + signal_var)
        V_hat = V_hat_prev + nu * K
        var_hat = (1 - K) * (var_hat_prev + self.process_var * 1)
        self.V_hats.append(V_hat)
        self.var_hats.append(var_hat)

        return V_hat, var_hat

    def observe(self, V, t):
        """Observe new signal (mid-price with noise)"""
        eps = np.random.normal(0, 0.5)
        signal = V[t-self.lag] + eps
        return signal
    
    def compute_imbalance(self, t):
        window = self.imbalance_window
        if len(self.orders) < window:
            return sum(self.orders)
        return sum(self.orders[-window:])

    def get_quote(self, V, t):
        """Simulate a smart Market Maker"""
        if t % self.freq == 0:
            signal = self.observe(V, t)
            V_hat, var_hat = self.estimator_update(signal)
            I = self.compute_imbalance(t)
            s = self.spread + self.k_sigma * np.sqrt(var_hat) + self.k_I * abs(I)
            mid = V_hat + self.beta * I - self.gamma * self.inventory
            self.mids.append(mid)
            quote = (mid - s/2, mid + s/2)
            self.quotes.append(quote)
        
        return self.quotes[-1]
    
    def consume_depth(self, amount, side='buy'):
        """Called when market orders consume liquidity."""
        if side == 'buy':
            taken = min(self.depth_ask, amount)
            self.depth_ask -= taken
            return taken
        else:
            taken = min(self.depth_bid, amount)
            self.depth_bid -= taken
            return taken

    def end_of_timestep_update(self):
        """Refill depth at timestep boundary (call once per simulation step)."""
        # deterministic exponential refill toward base_depth
        self.depth_ask += self.rho * (self.base_depth - self.depth_ask)
        arrivals_ask = np.random.poisson(self.lambda_limit)
        self.depth_ask += arrivals_ask * self.avg_limit_size
        # refill bid
        self.depth_bid += self.rho * (self.base_depth - self.depth_bid)
        arrivals_bid = np.random.poisson(self.lambda_limit)
        self.depth_bid += arrivals_bid * self.avg_limit_size
        # clip
        self.depth_ask = max(self.min_depth, min(self.depth_ask, self.depth_at_best))
        self.depth_bid = max(self.min_depth, min(self.depth_bid, self.depth_at_best))

    def execute_market_order(self, side: str, size: float):
        """
        side: "buy" means trader buys from MM (MM sells)
              "sell" means trader sells to MM (MM buys)
        size: requested size
        Returns: dict with fill details (filled_size, avg_price, mm_cash_change, mm_inventory_change, trader_fill_info)
        """
        bid, ask = self.quotes[-1]
        filled = 0.0
        cash_change = 0.0
        inv_change = 0.0
        # Determine execution price schedule
        if side == "buy":
            # Trader buys at ask: MM sells
            # Fill up to depth at ask
            take = self.consume_depth(size, side)
            filled += take
            cash_change += take * ask    # MM receives cash (selling)
            inv_change -= take           # MM sold => inventory decreases
            remaining = size - take
            if remaining > 0:
                # fill remaining at linearly worse price: ask + impact_per_unit * (units_over / depth)
                penalty = self.impact_per_unit * (remaining / max(self.base_depth, 1.0))
                worse_price = ask + penalty
                filled += remaining
                cash_change += remaining * worse_price
                inv_change -= remaining
        
        elif side == "sell":
            # Trader sells to MM at bid: MM buys
            take = self.consume_depth(size, side)
            filled += take
            cash_change -= take * bid   # MM pays cash to buy
            inv_change += take
            remaining = size - take
            if remaining > 0:
                penalty = self.impact_per_unit * (remaining / max(self.base_depth, 1.0))
                worse_price = bid - penalty
                filled += remaining
                cash_change -= remaining * worse_price
                inv_change += remaining
        else:
            raise ValueError("side must be 'buy' or 'sell'")
        
        # Update MM state
        self.cash += cash_change
        self.inventory += inv_change

        avg_price = cash_change / filled if filled > 0 else 0.0
        
        return {
            "filled": filled,
            "avg_price": avg_price
        }

    def mark_to_market(self, V, t):
        """Current PnL = cash + inventory * true value"""
        return self.cash + self.inventory * V[t]