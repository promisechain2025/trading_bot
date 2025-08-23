import pandas as pd
import numpy as np
import torch
from src.model import EnsembleModel
from src.bot import BotModel
from src.view import TradingView

def calculate_rsi(df, period=14):
    delta = df['price_a'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rs = rs.fillna(0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):
    high_low = df['price_a'].rolling(window=2).apply(lambda x: abs(x.max() - x.min()))
    atr = high_low.rolling(window=period, min_periods=1).mean().fillna(0)
    return atr

def calculate_bb_width(df, period=20):
    sma = df['price_a'].rolling(window=period).mean()
    std = df['price_a'].rolling(window=period).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    bb_width = (upper_band - lower_band) / sma
    return bb_width.fillna(0)

def calculate_stoch_k(df, period=30):
    low_min = df['price_a'].rolling(window=period).min()
    high_max = df['price_a'].rolling(window=period).max()
    epsilon = 1e-6
    stoch_k = 100 * (df['price_a'] - low_min) / (high_max - low_min + epsilon)
    stoch_k = stoch_k.rolling(window=3).mean().fillna(50).clip(0.001, 99.999)
    return stoch_k

def calculate_ma_crossover(df, short_period=20, long_period=50):
    ma_short = df['price_a'].rolling(window=short_period).mean()
    ma_long = df['price_a'].rolling(window=long_period).mean()
    crossover = ma_short - ma_long
    return crossover.fillna(0)

class TradingController:
    def __init__(self, capital: float = 10000.0):
        self.trading_model = EnsembleModel()
        self.bot_model = BotModel(capital=capital)
        self.view = TradingView()
    
    def load_and_train(self, data: pd.DataFrame):
        try:
            if 'momentum' not in data.columns or data['momentum'].isna().all():
                data["momentum"] = data["price_a"].pct_change(5).fillna(0)
            data["rsi"] = calculate_rsi(data)
            data["atr"] = calculate_atr(data)
            data["bb_width"] = calculate_bb_width(data)
            data["stoch_k"] = calculate_stoch_k(data)
            data["ma_crossover"] = calculate_ma_crossover(data)
            
            print(f"Data columns: {data.columns.tolist()}")
            print(f"Price_a range: min={data['price_a'].min():.2f}, max={data['price_a'].max():.2f}")
            self.trading_model.fit_scaler(data)
            
            train_size = int(len(data) * 0.7)
            val_size = int(len(data) * 0.15)
            train_data = data.iloc[:train_size]
            val_data = data.iloc[train_size:train_size + val_size]
            test_data = data.iloc[train_size + val_size:]
            
            X_train, y_train = self.trading_model.model1.prepare_data(train_data)
            X_val, y_val = self.trading_model.model1.prepare_data(val_data)
            self.trading_model.train_model(X_train, y_train, X_val, y_val)
            
            if len(test_data) > 60:
                X_test, y_test = self.trading_model.model1.prepare_data(test_data)
                self.trading_model.model1.eval()
                with torch.no_grad():
                    test_outputs = self.trading_model.model1.forward(X_test)
                    test_loss = self.trading_model.model1.criterion(test_outputs, y_test).item()
                    test_predictions = self.trading_model.model1.price_scaler.inverse_transform(test_outputs.cpu().numpy())
                    print(f"Sample predictions: {test_predictions[-5:]}")
                    print(f"Actual prices: {test_data[['price_a', 'price_b']].iloc[-5:].values}")
                    print(f"Input features (last row): {X_test[-1, -1, :]}")
                self.view.display_message(f"Test Loss: {test_loss}")
            
            self.view.display_message("Training completed successfully.")
        except Exception as e:
            self.view.display_error(f"Training failed: {str(e)}")
    
    def simulate_trading(self, data: pd.DataFrame):
        try:
            for i in range(60, len(data)):
                recent = data.iloc[i - 60:i]
                curr_price_a = data.iloc[i]["price_a"]
                curr_price_b = data.iloc[i]["price_b"]
                if curr_price_a <= 0:
                    continue
                sentiment = self.bot_model.get_sentiment_score(data.iloc[i]["sentiment"])
                momentum = data.iloc[i]["momentum"]
                rsi = data.iloc[i]["rsi"]
                macd = 0
                macd_signal = 0
                bb_width = data.iloc[i]["bb_width"]
                stoch_k = data.iloc[i]["stoch_k"]
                obv = 0
                ma_crossover = data.iloc[i]["ma_crossover"]
                atr = data.iloc[i]["atr"]
                pred = self.trading_model.predict_next(recent)
                pred_price_a = pred[0]
                strategy = self.bot_model.select_strategy(pred_price_a, curr_price_a, curr_price_b, sentiment, momentum, rsi, macd, macd_signal, bb_width, stoch_k, obv, ma_crossover, atr)
                self.bot_model.execute_trade(curr_price_a, curr_price_b, strategy, atr, bb_width, pred_price_a)
            final_capital = self.bot_model.capital + self.bot_model.position * data.iloc[-1]["price_a"]
            self.view.display_final_results(final_capital, self.bot_model.trade_log, self.bot_model.trade_profits,
                                           self.bot_model.strategy_counts, self.bot_model.strategy_wins)
        except Exception as e:
            self.view.display_error(f"Simulation failed: {str(e)}")