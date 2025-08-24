import pandas as pd
import numpy as np
import torch
import yfinance as yf
from src.model import BetterTradingModel
from src.bot import BotModel
from src.view import TradingView
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.alphaintelligence import AlphaIntelligence
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')

def calculate_atr(df, period=14):
    high_low = df['price_a'].rolling(window=2).apply(lambda x: abs(x.max() - x.min()))
    atr = high_low.rolling(window=period, min_periods=1).mean().fillna(0)
    return atr

def calculate_ma_crossover(df, short_period=20, long_period=50):
    ma_short = df['price_a'].rolling(window=short_period).mean()
    ma_long = df['price_a'].rolling(window=long_period).mean()
    crossover = ma_short - ma_long
    return crossover.fillna(0)

def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    ema_short = df['price_a'].ewm(span=short_period, adjust=False).mean()
    ema_long = df['price_a'].ewm(span=long_period, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

class TradingController:
    def __init__(self, capital: float = 10000.0):
        self.trading_model = BetterTradingModel()
        self.bot_model = BotModel(capital=capital)
        self.view = TradingView()

    def validate_data(self, data: pd.DataFrame):
        required_columns = ['price_a', 'price_b', 'sentiment']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data missing required columns: {required_columns}")
        if len(data) < 60:
            raise ValueError("Data must have at least 60 rows for sequence length")
        if (data['price_a'] <= 0).any() or (data['price_b'] <= 0).any():
            raise ValueError("Prices must be positive")

    def fetch_sentiment_newsapi(self, ticker, from_date, to_date):
        if not NEWSAPI_KEY:
            print("Warning: No NewsAPI key provided, using neutral sentiment")
            return 0.0
        try:
            url = f"https://newsapi.org/v2/everything?q={ticker}&from={from_date}&to={to_date}&language=en&apiKey={NEWSAPI_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            if not articles:
                print(f"No news found for {ticker} from {from_date} to {to_date}, using neutral sentiment")
                return 0.0
            analyzer = SentimentIntensityAnalyzer()
            sentiments = [analyzer.polarity_scores(article['title'])['compound'] for article in articles]
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            print(f"NewsAPI sentiment for {ticker} from {from_date} to {to_date}: {avg_sentiment:.3f}")
            return avg_sentiment
        except Exception as e:
            if '426' in str(e):
                print(f"NewsAPI 426 error for {ticker}: Upgrade required, likely not running on localhost")
            else:
                print(f"Error fetching NewsAPI sentiment for {ticker}: {str(e)}")
            return None

    def fetch_sentiment_alpha_vantage(self, ticker, from_date, to_date):
        if not ALPHA_VANTAGE_KEY:
            print("Warning: No Alpha Vantage key provided, using neutral sentiment")
            return 0.0
        try:
            ai = AlphaIntelligence(key=ALPHA_VANTAGE_KEY)
            data = ai.get_news_sentiment(tickers=ticker, time_from=from_date + 'T0000', time_to=to_date + 'T2359')
            sentiments = [item['overall_sentiment_score'] for item in data.get('feed', [])]
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            print(f"Alpha Vantage sentiment for {ticker} from {from_date} to {to_date}: {avg_sentiment:.3f}")
            return avg_sentiment
        except Exception as e:
            print(f"Error fetching Alpha Vantage sentiment for {ticker}: {str(e)}")
            return 0.0

    def fetch_sentiment(self, ticker, from_date, to_date):
        sentiment = self.fetch_sentiment_newsapi(ticker, from_date, to_date)
        if sentiment is not None:
            return sentiment
        print("Falling back to Alpha Vantage for sentiment")
        sentiment = self.fetch_sentiment_alpha_vantage(ticker, from_date, to_date)
        if sentiment is not None:
            return sentiment
        print(f"Using neutral sentiment for {ticker} from {from_date} to {to_date}")
        return 0.0

    def fetch_real_time_data(self, ticker_a='AAPL', ticker_b=None, start_date='2023-01-01', end_date=None, interval='1d'):
        try:
            hist_data_a = yf.download(ticker_a, start=start_date, end=end_date, interval=interval)
            data = pd.DataFrame()
            data['price_a'] = hist_data_a['Close']
            if ticker_b:
                hist_data_b = yf.download(ticker_b, start=start_date, end=end_date, interval=interval)
                data['price_b'] = hist_data_b['Close']
            else:
                data['price_b'] = hist_data_a['Open']
            data['sentiment'] = 0.0
            for date in data.index:
                from_date = (date - timedelta(days=1)).strftime('%Y-%m-%d')
                to_date = date.strftime('%Y-%m-%d')
                data.loc[date, 'sentiment'] = self.fetch_sentiment(ticker_a, from_date, to_date)
            data['volume'] = hist_data_a['Volume']
            data = data.dropna()
            latest_a = yf.Ticker(ticker_a).info
            current_price_a = latest_a.get('currentPrice', data['price_a'].iloc[-1])
            if ticker_b:
                latest_b = yf.Ticker(ticker_b).info
                current_price_b = latest_b.get('currentPrice', data['price_b'].iloc[-1])
            else:
                current_price_b = latest_a.get('regularMarketOpen', data['price_b'].iloc[-1])
            latest_sentiment = self.fetch_sentiment(ticker_a, (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
            new_row = pd.DataFrame({
                'price_a': [current_price_a],
                'price_b': [current_price_b],
                'sentiment': [latest_sentiment],
                'volume': [latest_a.get('volume', data['volume'].iloc[-1])]
            }, index=[pd.Timestamp.now()])
            data = pd.concat([data, new_row])
            # Compute indicators (some computed by better_model.py, but kept for bot.py)
            data['atr'] = calculate_atr(data)
            data['ma_crossover'] = calculate_ma_crossover(data)
            data['macd'], data['macd_signal'] = calculate_macd(data)
            return data
        except Exception as e:
            raise ValueError(f"Failed to fetch real-time data: {str(e)}")

    def load_and_train(self, data: pd.DataFrame = None, ticker_a='AAPL', ticker_b=None):
        if data is None:
            data = self.fetch_real_time_data(ticker_a=ticker_a, ticker_b=ticker_b)
        try:
            self.validate_data(data)
            train_size = int(len(data) * 0.7)
            val_size = int(len(data) * 0.15)
            train_data = data.iloc[:train_size]
            val_data = data.iloc[train_size:train_size + val_size]
            test_data = data.iloc[train_size + val_size:]
            self.trading_model.train_model(train_df=train_data, val_df=val_data)
            if len(test_data) > 60:
                X_test, yret_test, ydir_test = self.trading_model.make_sequences(test_data, seq_len=60)
                self.trading_model.model.eval()
                with torch.no_grad():
                    outputs = self.trading_model.model(X_test.to(self.trading_model.device))
                    q_loss = self.trading_model.model.criterion(outputs['q'], yret_test.to(self.trading_model.device))
                    bce = nn.BCEWithLogitsLoss()
                    dir_loss = bce(outputs['dir_logits'], ydir_test.to(self.trading_model.device))
                    test_loss = q_loss + 0.2 * dir_loss
                    print(f"Test Quantile Loss: {q_loss.item():.6f}, Direction Loss: {dir_loss.item():.6f}")
                    last_prices = test_data[['price_a', 'price_b']].iloc[-5:].values
                    pred = self.trading_model.predict_next(test_data)
                    print(f"Sample predictions: price_a={pred['price_a_quantiles'][1]}, price_b={pred['price_b_quantiles'][1]}")
                    print(f"Actual prices: {last_prices}")
                self.view.display_message(f"Test Loss: {test_loss.item():.6f}")
            self.view.display_message("Training completed successfully.")
        except Exception as e:
            self.view.display_error(f"Training failed: {str(e)}")

    def simulate_trading(self, data: pd.DataFrame = None, ticker_a='AAPL', ticker_b=None):
        if data is None:
            data = self.fetch_real_time_data(ticker_a=ticker_a, ticker_b=ticker_b)
        try:
            self.validate_data(data)
            for i in range(60, len(data)):
                recent = data.iloc[i - 60:i]
                curr_price_a = data.iloc[i]["price_a"]
                curr_price_b = data.iloc[i]["price_b"]
                if curr_price_a <= 0:
                    continue
                sentiment = self.bot_model.get_sentiment_score(data.iloc[i]["sentiment"])
                momentum = data.iloc[i]["momentum"]
                rsi = data.iloc[i]["rsi"]
                macd = data.iloc[i]["macd"]
                macd_signal = data.iloc[i]["macd_signal"]
                bb_width = data.iloc[i]["bb_width"]
                stoch_k = data.iloc[i]["stoch_k"]
                obv = 0
                ma_crossover = data.iloc[i]["ma_crossover"]
                atr = data.iloc[i]["atr"]
                pred = self.trading_model.predict_next(recent)
                pred_price_a = pred['price_a_quantiles'][1]  # Use Q50
                pred_price_b = pred['price_b_quantiles'][1]  # Use Q50
                prob_up_a = pred['prob_up'][0]
                strategy = self.bot_model.select_strategy(
                    pred_price_a, curr_price_a, curr_price_b, sentiment, momentum, rsi,
                    macd, macd_signal, bb_width, stoch_k, obv, ma_crossover, atr
                )
                # Adjust strategy based on prob_up
                if prob_up_a > 0.7 and strategy != 'buy':
                    strategy = 'buy'
                elif prob_up_a < 0.3 and strategy != 'sell':
                    strategy = 'sell'
                self.bot_model.execute_trade(curr_price_a, curr_price_b, strategy, atr, bb_width, pred_price_a)
            final_capital = self.bot_model.capital + self.bot_model.position * data.iloc[-1]["price_a"]
            self.view.display_final_results(
                final_capital, self.bot_model.trade_log, self.bot_model.trade_profits,
                self.bot_model.strategy_counts, self.bot_model.strategy_wins
            )
        except Exception as e:
            self.view.display_error(f"Simulation failed: {str(e)}")

    def predict_real_time(self, ticker_a='AAPL', ticker_b=None, seq_length=60):
        data = self.fetch_real_time_data(ticker_a=ticker_a, ticker_b=ticker_b)
        recent = data.tail(seq_length + 1)  # BetterTradingModel needs seq_len + 1
        pred = self.trading_model.predict_next(recent)
        print(f"Predicted next price_a (Q10, Q50, Q90): {pred['price_a_quantiles']}")
        print(f"Predicted next price_b (Q10, Q50, Q90): {pred['price_b_quantiles']}")
        print(f"Probability of up move: price_a={pred['prob_up'][0]:.3f}, price_b={pred['prob_up'][1]:.3f}")
        return pred