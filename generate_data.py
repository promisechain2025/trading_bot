import numpy as np
import pandas as pd

np.random.seed(42)
dates = pd.date_range('2025-01-01', periods=300)
base_trend = np.linspace(0, 100, 300)
volatility_a = np.random.randn(300) * 10
volatility_b = np.random.randn(300) * 10 + np.random.uniform(-0.01, 0.01, 300) * base_trend  # Smaller discrepancies
prices_a = np.cumsum(volatility_a) + 100 + base_trend
prices_b = np.cumsum(volatility_b) + 100 + base_trend
sentiments = np.clip(0.5 + 0.5 * np.tanh(prices_a / 50 - 1), 0.2, 0.95)
data = pd.DataFrame({'date': dates, 'price_a': prices_a, 'price_b': prices_b, 'sentiment': sentiments})
data['momentum'] = data['price_a'].pct_change(5).fillna(0)
data['volatility'] = data['price_a'].pct_change().rolling(20).std().fillna(0)
data.to_csv('data/sample_data.csv', index=False)