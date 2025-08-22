import pandas as pd

class BotModel:
    def __init__(self, capital=10000, risk_per_trade=0.01, trading_fee=0.001, max_drawdown=0.2, max_trade_amount=1000,
                 trend_threshold=0.017, momentum_long=0.035, momentum_short=-0.035, sentiment_long=0.95, sentiment_short=0.05,
                 trail_multiplier=2.0):
        self.capital = capital
        self.initial_capital = capital
        self.risk_per_trade = risk_per_trade
        self.trading_fee = trading_fee
        self.max_drawdown = max_drawdown
        self.max_trade_amount = max_trade_amount
        self.arbitrage_threshold = 0.002  # Adjusted dynamically
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
        self.strategy_counts = {'arbitrage': 0, 'trend_follow_long': 0, 'trend_follow_short': 0, 'scalp': 0}
        self.strategy_wins = {'arbitrage': 0, 'trend_follow_long': 0, 'trend_follow_short': 0, 'scalp': 0}
    
    def get_sentiment_score(self, current_sentiment):
        return current_sentiment
    
    def select_strategy(self, pred_price_a, curr_price_a, curr_price_b, sentiment, momentum, rsi, macd, bb_width, stoch_k, obv, atr):
        self.arbitrage_threshold = max(0.0003, min(0.0015, 0.002 / (1 + bb_width * 5)))
        diff = (curr_price_b - curr_price_a) / curr_price_a
        portfolio_value = self.capital + self.position * curr_price_a
        min_price_move = curr_price_a * 0.017
        obv_slope = obv - obv.shift(5).fillna(obv) if isinstance(obv, pd.Series) else obv
        
        if abs(diff) > self.arbitrage_threshold:
            return 'arbitrage'
        if (sentiment > self.sentiment_long and 
            pred_price_a > curr_price_a + min_price_move and 
            momentum > self.momentum_long and 
            rsi < 70 and 
            macd > 0 and 
            bb_width > 0.02 and 
            stoch_k < 80 and 
            obv_slope > 0):
            return 'trend_follow_long'
        elif (sentiment < self.sentiment_short and 
              pred_price_a < curr_price_a - min_price_move and 
              momentum < self.momentum_short and 
              rsi > 30 and 
              macd < 0 and 
              bb_width > 0.02 and 
              stoch_k > 20 and 
              obv_slope < 0):
            return 'trend_follow_short'
        elif (abs(pred_price_a - curr_price_a) < 0.05 * curr_price_a and 
              abs(momentum) > 0.005 and 
              bb_width > 0.01 and 
              20 < stoch_k < 80):
            return 'scalp'
        elif self.position > 0 and curr_price_a < self.trailing_stop:
            return 'stop_loss'
        elif self.position < 0 and curr_price_a > self.trailing_stop:
            return 'stop_loss'
        elif portfolio_value < self.initial_capital * (1 - self.max_drawdown):
            return 'max_drawdown'
        else:
            return 'hold'
    
    def execute_trade(self, curr_price_a, curr_price_b, strategy, atr, bb_width, pred_price_a):
        if curr_price_a <= 0 or curr_price_b <= 0:
            return False
        risk_per_trade = max(0.005, min(0.015, self.risk_per_trade / (1 + bb_width * 10)))
        amount = min(self.capital * risk_per_trade / curr_price_a, self.max_trade_amount, self.capital * 0.02 / curr_price_a)
        trail_multiplier = max(1.5, min(2.5, self.trail_multiplier / (1 + bb_width * 2)))
        profit_target = max(1.008, min(1.013, 1.01 + atr / curr_price_a))
        diff = (curr_price_b - curr_price_a) / curr_price_a
        
        if strategy == 'arbitrage' and self.position == 0:
            if diff > 0:
                profit = amount * (curr_price_b - curr_price_a) - amount * (curr_price_a + curr_price_b) * self.trading_fee
                if profit <= 0:
                    return False
                self.capital -= amount * curr_price_a * (1 + self.trading_fee)
                self.capital += amount * curr_price_b * (1 - self.trading_fee)
                self.trade_log.append(f'Arbitrage: Buy {amount:.2f} on A at {curr_price_a:.2f}, Sell on B at {curr_price_b:.2f}, Profit: {profit:.2f}')
                self.trade_profits.append(profit)
                self.strategy_counts['arbitrage'] += 1
                if profit > 0:
                    self.strategy_wins['arbitrage'] += 1
            else:
                profit = amount * (curr_price_a - curr_price_b) - amount * (curr_price_b + curr_price_a) * self.trading_fee
                if profit <= 0:
                    return False
                self.capital -= amount * curr_price_b * (1 + self.trading_fee)
                self.capital += amount * curr_price_a * (1 - self.trading_fee)
                self.trade_log.append(f'Arbitrage: Buy {amount:.2f} on B at {curr_price_b:.2f}, Sell on A at {curr_price_a:.2f}, Profit: {profit:.2f}')
                self.trade_profits.append(profit)
                self.strategy_counts['arbitrage'] += 1
                if profit > 0:
                    self.strategy_wins['arbitrage'] += 1
            return True
        
        if strategy == 'trend_follow_long' and self.position == 0:
            self.position += amount
            self.capital -= amount * curr_price_a * (1 + self.trading_fee)
            self.entry_price = curr_price_a
            self.trailing_stop = curr_price_a - trail_multiplier * atr
            self.trade_log.append(f'Buy long {amount:.2f} at {curr_price_a:.2f}, Fee: {amount * curr_price_a * self.trading_fee:.2f}')
            self.strategy_counts['trend_follow_long'] += 1
            return True
        
        elif strategy == 'trend_follow_short' and self.position == 0:
            self.position -= amount
            self.capital += amount * curr_price_a * (1 - self.trading_fee)
            self.entry_price = curr_price_a
            self.trailing_stop = curr_price_a + trail_multiplier * atr
            self.trade_log.append(f'Short sell {amount:.2f} at {curr_price_a:.2f}, Fee: {amount * curr_price_a * self.trading_fee:.2f}')
            self.strategy_counts['trend_follow_short'] += 1
            return True
        
        elif strategy == 'scalp':
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
                self.entry_price = None
                self.trailing_stop = None
                return True
            elif self.position < 0 and curr_price_a < self.entry_price * (2 - profit_target):
                cover_amount = -self.position
                profit = (self.entry_price - curr_price_a) * cover_amount - self.trading_fee * (self.entry_price + curr_price_a) * cover_amount
                self.trade_profits.append(profit)
                self.capital -= cover_amount * curr_price_a * (1 + self.trading_fee)
                self.position += cover_amount
                self.trade_log.append(f'Scalp cover short {cover_amount:.2f} at {curr_price_a:.2f}, Fee: {cover_amount * curr_price_a * self.trading_fee:.2f}')
                self.strategy_counts['scalp'] += 1
                if profit > 0:
                    self.strategy_wins['scalp'] += 1
                self.entry_price = None
                self.trailing_stop = None
                return True
        
        elif strategy in ['stop_loss', 'max_drawdown']:
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
            elif self.position < 0:
                cover_amount = -self.position
                profit = (self.entry_price - curr_price_a) * cover_amount - self.trading_fee * (self.entry_price + curr_price_a) * cover_amount
                self.trade_profits.append(profit)
                self.capital -= cover_amount * curr_price_a * (1 + self.trading_fee)
                self.position = 0
                self.trade_log.append(f'{strategy.replace("_", "-").title()} Cover short {cover_amount:.2f} at {curr_price_a:.2f}, Fee: {cover_amount * curr_price_a * self.trading_fee:.2f}')
                self.strategy_counts['trend_follow_short'] += 1
                if profit > 0:
                    self.strategy_wins['trend_follow_short'] += 1
            self.entry_price = None
            self.trailing_stop = None
            return True
        
        if self.position > 0:
            self.trailing_stop = max(self.trailing_stop, curr_price_a - trail_multiplier * atr)
        elif self.position < 0:
            self.trailing_stop = min(self.trailing_stop, curr_price_a + trail_multiplier * atr)
        return False