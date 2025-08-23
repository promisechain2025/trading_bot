import pandas as pd

class BotModel:
    def __init__(self, capital=10000, risk_per_trade=0.015, trading_fee=0.001, max_drawdown=0.2, max_trade_amount=1000,
                 trend_threshold=0.0001, momentum_long=0.0005, momentum_short=-0.0005, sentiment_long=0.2, sentiment_short=0.8,
                 trail_multiplier=2.5):
        self.capital = capital
        self.initial_capital = capital
        self.risk_per_trade = risk_per_trade
        self.trading_fee = trading_fee
        self.max_drawdown = max_drawdown
        self.max_trade_amount = max_trade_amount
        self.trend_threshold = trend_threshold
        self.momentum_long = momentum_long
        self.momentum_short = momentum_short
        self.sentiment_long = sentiment_long
        self.sentiment_short = sentiment_short
        self.trail_multiplier = trail_multiplier
        self.position = 0
        self.entry_price = None
        self.trailing_stop = None
        self.trade_log = []
        self.trade_profits = []
        self.strategy_counts = {'trend_follow_long': 0, 'trend_follow_short': 0, 'scalp': 0}
        self.strategy_wins = {'trend_follow_long': 0, 'trend_follow_short': 0, 'scalp': 0}
    
    def get_sentiment_score(self, current_sentiment):
        return current_sentiment
    
    def select_strategy(self, pred_price_a, curr_price_a, curr_price_b, sentiment, momentum, rsi, macd, macd_signal, bb_width, stoch_k, obv, ma_crossover, atr):
        portfolio_value = self.capital + self.position * curr_price_a
        min_price_move = curr_price_a * max(self.trend_threshold, atr * 0.0001)
        
        if not (sentiment > self.sentiment_long):
            print(f"Trend Long condition failed: sentiment={sentiment:.3f} <= {self.sentiment_long}")
        if not (pred_price_a > curr_price_a + min_price_move):
            print(f"Trend Long condition failed: pred_price_a={pred_price_a:.2f} <= {curr_price_a + min_price_move:.2f}")
        if not (momentum > self.momentum_long):
            print(f"Trend Long condition failed: momentum={momentum:.3f} <= {self.momentum_long}")
        if not (bb_width > 0.002):
            print(f"Trend Long condition failed: bb_width={bb_width:.3f} <= 0.002")
        if not (stoch_k < 99.999):
            print(f"Trend Long condition failed: stoch_k={stoch_k:.2f} >= 99.999")
        if not (ma_crossover > -10.0):
            print(f"Trend Long condition failed: ma_crossover={ma_crossover:.3f} <= -10.0")
        
        if (sentiment > self.sentiment_long and 
            pred_price_a > curr_price_a + min_price_move and 
            momentum > self.momentum_long and 
            bb_width > 0.002 and 
            stoch_k < 99.999 and 
            ma_crossover > -10.0):
            return 'trend_follow_long'
        
        if not (sentiment < self.sentiment_short):
            print(f"Trend Short condition failed: sentiment={sentiment:.3f} >= {self.sentiment_short}")
        if not (pred_price_a < curr_price_a - min_price_move):
            print(f"Trend Short condition failed: pred_price_a={pred_price_a:.2f} >= {curr_price_a - min_price_move:.2f}")
        if not (momentum < self.momentum_short):
            print(f"Trend Short condition failed: momentum={momentum:.3f} >= {self.momentum_short}")
        if not (bb_width > 0.002):
            print(f"Trend Short condition failed: bb_width={bb_width:.3f} <= 0.002")
        if not (stoch_k > 0.001):
            print(f"Trend Short condition failed: stoch_k={stoch_k:.2f} <= 0.001")
        if not (ma_crossover < 10.0):
            print(f"Trend Short condition failed: ma_crossover={ma_crossover:.3f} >= 10.0")
        
        if (sentiment < self.sentiment_short and 
            pred_price_a < curr_price_a - min_price_move and 
            momentum < self.momentum_short and 
            bb_width > 0.002 and 
            stoch_k > 0.001 and 
            ma_crossover < 10.0):
            return 'trend_follow_short'
        
        if not (abs(pred_price_a - curr_price_a) < 0.12 * curr_price_a):
            print(f"Scalp condition failed: |pred_price_a - curr_price_a|={abs(pred_price_a - curr_price_a):.2f} >= {0.12 * curr_price_a:.2f}")
        if not (abs(momentum) > 0.0005):
            print(f"Scalp condition failed: |momentum|={abs(momentum):.3f} <= 0.0005")
        if not (bb_width > 0.002):
            print(f"Scalp condition failed: bb_width={bb_width:.3f} <= 0.002")
        if not (0.001 < stoch_k < 99.999):
            print(f"Scalp condition failed: stoch_k={stoch_k:.2f} not in (0.001, 99.999)")
        
        if (abs(pred_price_a - curr_price_a) < 0.12 * curr_price_a and 
            abs(momentum) > 0.0005 and 
            bb_width > 0.002 and 
            0.001 < stoch_k < 99.999):
            return 'scalp'
        
        if self.position > 0 and curr_price_a < self.trailing_stop:
            return 'stop_loss'
        if self.position < 0 and curr_price_a > self.trailing_stop:
            return 'stop_loss'
        if portfolio_value < self.initial_capital * (1 - self.max_drawdown):
            return 'max_drawdown'
        print(f"Holding: No strategy conditions met")
        return 'hold'
    
    def execute_trade(self, curr_price_a, curr_price_b, strategy, atr, bb_width, pred_price_a):
        if curr_price_a <= 0:
            print(f"Invalid price: curr_price_a={curr_price_a}")
            return False
        risk_per_trade = max(0.005, min(0.02, self.risk_per_trade / (1 + bb_width * 10)))
        amount = min(self.capital * risk_per_trade / curr_price_a, self.max_trade_amount, self.capital * 0.02 / curr_price_a)
        trail_multiplier = max(2.0, min(3.0, self.trail_multiplier / (1 + bb_width * 2)))
        profit_target = max(1.003, min(1.008, 1.005 + atr / curr_price_a))
        
        if strategy == 'trend_follow_long' and self.position == 0:
            self.position += amount
            self.capital -= amount * curr_price_a * (1 + self.trading_fee)
            self.entry_price = curr_price_a
            self.trailing_stop = curr_price_a - trail_multiplier * atr
            self.trade_log.append(f'Buy long {amount:.2f} at {curr_price_a:.2f}, Fee: {amount * curr_price_a * self.trading_fee:.2f}')
            self.strategy_counts['trend_follow_long'] += 1
            print(f"Executing trend_follow_long: Buy {amount:.2f} at {curr_price_a:.2f}")
            return True
        
        if strategy == 'trend_follow_short' and self.position == 0:
            self.position -= amount
            self.capital += amount * curr_price_a * (1 - self.trading_fee)
            self.entry_price = curr_price_a
            self.trailing_stop = curr_price_a + trail_multiplier * atr
            self.trade_log.append(f'Short sell {amount:.2f} at {curr_price_a:.2f}, Fee: {amount * curr_price_a * self.trading_fee:.2f}')
            self.strategy_counts['trend_follow_short'] += 1
            print(f"Executing trend_follow_short: Sell {amount:.2f} at {curr_price_a:.2f}")
            return True
        
        if strategy == 'scalp':
            if self.position > 0 and curr_price_a > self.entry_price * profit_target:
                sell_amount = self.position
                profit = (curr_price_a - self.entry_price) * sell_amount - self.trading_fee * (self.entry_price + curr_price_a) * sell_amount
                self.trade_profits.append(profit)
                self.capital += sell_amount * curr_price_a * (1 - self.trading_fee)
                self.position -= sell_amount
                self.trade_log.append(f'Scalp sell long {sell_amount:.2f} at {curr_price_a:.2f}, Fee: {sell_amount * curr_price_a * self.trading_fee:.2f}')
                self.strategy_counts['scalp'] += 1
                if profit > 0:
                    self.strategy_wins['scalp'] += 1
                print(f"Executing scalp: Sell {sell_amount:.2f} at {curr_price_a:.2f}, Profit: {profit:.2f}")
                self.entry_price = None
                self.trailing_stop = None
                return True
            if self.position < 0 and curr_price_a < self.entry_price * (2 - profit_target):
                cover_amount = -self.position
                profit = (self.entry_price - curr_price_a) * cover_amount - self.trading_fee * (self.entry_price + curr_price_a) * cover_amount
                self.trade_profits.append(profit)
                self.capital -= cover_amount * curr_price_a * (1 + self.trading_fee)
                self.position += cover_amount
                self.trade_log.append(f'Scalp cover short {cover_amount:.2f} at {curr_price_a:.2f}, Fee: {cover_amount * curr_price_a * self.trading_fee:.2f}')
                self.strategy_counts['scalp'] += 1
                if profit > 0:
                    self.strategy_wins['scalp'] += 1
                print(f"Executing scalp: Cover {cover_amount:.2f} at {curr_price_a:.2f}, Profit: {profit:.2f}")
                self.entry_price = None
                self.trailing_stop = None
                return True
        
        if strategy in ['stop_loss', 'max_drawdown']:
            if self.position > 0:
                sell_amount = self.position
                profit = (curr_price_a - self.entry_price) * sell_amount - self.trading_fee * (self.entry_price + curr_price_a) * sell_amount
                self.trade_profits.append(profit)
                self.capital += sell_amount * curr_price_a * (1 - self.trading_fee)
                self.position = 0
                self.trade_log.append(f'{strategy.replace("_", "-").title()} Sell long {sell_amount:.2f} at {curr_price_a:.2f}, Fee: {sell_amount * curr_price_a * self.trading_fee:.2f}')
                self.strategy_counts['trend_follow_long'] += 1
                if profit > 0:
                    self.strategy_wins['trend_follow_long'] += 1
                print(f"Executing {strategy}: Sell {sell_amount:.2f} at {curr_price_a:.2f}, Profit: {profit:.2f}")
            if self.position < 0:
                cover_amount = -self.position
                profit = (self.entry_price - curr_price_a) * cover_amount - self.trading_fee * (self.entry_price + curr_price_a) * cover_amount
                self.trade_profits.append(profit)
                self.capital -= cover_amount * curr_price_a * (1 + self.trading_fee)
                self.position = 0
                self.trade_log.append(f'{strategy.replace("_", "-").title()} Cover short {cover_amount:.2f} at {curr_price_a:.2f}, Fee: {cover_amount * curr_price_a * self.trading_fee:.2f}')
                self.strategy_counts['trend_follow_short'] += 1
                if profit > 0:
                    self.strategy_wins['trend_follow_short'] += 1
                print(f"Executing {strategy}: Cover {cover_amount:.2f} at {curr_price_a:.2f}, Profit: {profit:.2f}")
            self.entry_price = None
            self.trailing_stop = None
            return True
        
        if self.position > 0:
            self.trailing_stop = max(self.trailing_stop, curr_price_a - trail_multiplier * atr)
        if self.position < 0:
            self.trailing_stop = min(self.trailing_stop, curr_price_a + trail_multiplier * atr)
        return False