import pytest
import pandas as pd
from src.models import TradingModel, BotModel

def test_trading_model_predict():
    model = TradingModel()
    data = pd.DataFrame({
        'price': [100 + i for i in range(20)],
        'sentiment': [0.5] * 20
    })
    pred = model.predict_next(data[-10:])
    assert isinstance(pred, float), "Prediction should be a float"

def test_bot_model_trade():
    bot = BotModel(capital=10000, risk_per_trade=0.02)
    bot.execute_trade(curr_price=100, strategy='trend_follow')
    assert bot.position > 0, "Should hold a position after buy"
    assert len(bot.trade_log) == 1, "Trade log should have one entry"
