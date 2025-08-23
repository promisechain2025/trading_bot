import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range(start='2025-01-01', end='2025-10-27', freq='D')
n = len(dates)
price_trend = np.concatenate([
    np.cumsum(np.random.normal(0.15, 3.0, n//4)),  # Stronger uptrend
    np.cumsum(np.random.normal(-0.2, 3.0, n//2)),  # Stronger downtrend
    np.cumsum(np.random.normal(0, 2.5, n - 3*(n//4)))  # Neutral
])
price_a = 80 + price_trend[:n]  # Adjusted base price
price_a = np.clip(price_a, 60, 120)  # Wider range
price_b = price_a * (1 + np.random.normal(0, 0.03, n))
# Correlate sentiment with price movements
sentiment = 0.5 + 0.4 * np.tanh(np.diff(price_a, prepend=price_a[0]) / 3) + np.random.normal(0, 0.1, n)
sentiment = np.clip(sentiment, 0.1, 0.9)
momentum = np.diff(price_a, prepend=price_a[0]) / price_a * 15  # Increased momentum scale
volatility = np.abs(np.diff(price_a, prepend=price_a[0])) / price_a * 3
data = pd.DataFrame({
    'date': dates,
    'price_a': price_a,
    'price_b': price_b,
    'sentiment': sentiment,
    'momentum': momentum,
    'volatility': volatility
})
data.to_csv('data.csv', index=False)
print("Generated data.csv with 300 rows")