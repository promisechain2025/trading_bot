class TradeExecutor:
    def __init__(self, capital=10000, risk_per_trade=0.05, trading_fee=0.001, max_trade_amount=1000):
        self.capital = capital
        self.initial_capital = capital
        self.risk_per_trade = risk_per_trade
        self.trading_fee = trading_fee
        self.max_trade_amount = max_trade_amount
        self.position = 0
        self.entry_price = None
        self.trailing_stop = None
        self.trade_log = []

    def execute_trade(self, curr_price_a, curr_price_b, strategy, volatility):
        amount = min(self.capital * self.risk_per_trade / curr_price_a * min(2.0, 1 + volatility * 15), self.max_trade_amount)
        if strategy == 'arbitrage' and self.position == 0:
            self.capital -= amount * curr_price_a * (1 + self.trading_fee)
            self.capital += amount * curr_price_b * (1 - self.trading_fee)
            self.trade_log.append(f'Arbitrage: Buy {amount:.2f} A at {curr_price_a:.2f}, Sell B at {curr_price_b:.2f}, Profit: {amount*(curr_price_b-curr_price_a-2*self.trading_fee*curr_price_a):.2f}')
            return True
        if strategy == 'trend_follow' and self.position == 0:
            self.position += amount
            self.capital -= amount * curr_price_a * (1 + self.trading_fee)
            self.entry_price = curr_price_a
            self.trailing_stop = curr_price_a * (1 - 0.10 - volatility)
            self.trade_log.append(f'Buy {amount:.2f} at {curr_price_a:.2f}, Fee: {amount*curr_price_a*self.trading_fee:.2f}')
            return True
        elif strategy == 'scalp' and self.position > 0 and curr_price_a > self.entry_price * 1.02:
            sell_amount = self.position * 0.5
            self.capital += sell_amount * curr_price_a * (1 - self.trading_fee)
            self.position -= sell_amount
            self.trade_log.append(f'Sell {sell_amount:.2f} at {curr_price_a:.2f}, Fee: {sell_amount*curr_price_a*self.trading_fee:.2f}')
            if self.position == 0:
                self.entry_price = None
                self.trailing_stop = None
            else:
                self.trailing_stop = max(self.trailing_stop, curr_price_a * (1 - 0.10 - volatility))
            return True
        elif strategy in ['stop_loss', 'max_drawdown'] and self.position > 0:
            sell_amount = self.position
            self.capital += sell_amount * curr_price_a * (1 - self.trading_fee)
            self.position = 0
            self.trade_log.append(f'{strategy.replace("_","-").title()} Sell {sell_amount:.2f} at {curr_price_a:.2f}, Fee: {sell_amount*curr_price_a*self.trading_fee:.2f}')
            self.entry_price = None
            self.trailing_stop = None
            return True
        if self.position > 0:
            self.trailing_stop = max(self.trailing_stop, curr_price_a*(1-0.10-volatility))
        return False