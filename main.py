import pandas as pd
from src.controller import TradingController
import matplotlib.pyplot as plt
import time
import schedule
import argparse
from datetime import datetime

def plot_capital(capital_history, timestamps, ticker_a='CSV'):
    """Plot portfolio value over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, capital_history, label='Portfolio Value')
    plt.xlabel('Time')
    plt.ylabel('Capital ($)')
    plt.title(f'Portfolio Value Over Time ({ticker_a})')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def run_trading(ticker_a='AAPL', ticker_b='GOOG', interval='1d', data_file=None, live=False):
    """Run training, simulation, and real-time prediction."""
    controller = TradingController(capital=10000)

    # Load data
    try:
        if data_file:
            print(f"Loading data from {data_file}")
            data = pd.read_csv(data_file, parse_dates=['date'], index_col='date')
            data['price_a'] = data['price_a'].clip(lower=0.01)
            data['price_b'] = data['price_b'].clip(lower=0.01)
            ticker_a = 'CSV'  # For plot title when using CSV
        else:
            print(f"Fetching real-time data for {ticker_a}/{ticker_b if ticker_b else ticker_a} (interval={interval})")
            data = controller.fetch_real_time_data(ticker_a=ticker_a, ticker_b=ticker_b, interval=interval)

        # Train model
        controller.load_and_train(data=data, ticker_a=ticker_a, ticker_b=ticker_b)

        # Simulate trading and track capital
        capital_history = []
        timestamps = []
        for i in range(60, len(data)):
            recent = data.iloc[i - 60:i]
            curr_price_a = data.iloc[i]["price_a"]
            if curr_price_a <= 0:
                continue
            pred = controller.trading_model.predict_next(recent)
            strategy = controller.bot_model.select_strategy(
                pred_price_a=pred[0],
                curr_price_a=curr_price_a,
                curr_price_b=data.iloc[i]["price_b"],
                sentiment=controller.bot_model.get_sentiment_score(data.iloc[i]["sentiment"]),
                momentum=data.iloc[i]["momentum"],
                rsi=data.iloc[i]["rsi"],
                macd=data.iloc[i]["macd"],
                macd_signal=data.iloc[i]["macd_signal"],
                bb_width=data.iloc[i]["bb_width"],
                stoch_k=data.iloc[i]["stoch_k"],
                obv=0,
                ma_crossover=data.iloc[i]["ma_crossover"],
                atr=data.iloc[i]["atr"]
            )
            controller.bot_model.execute_trade(
                curr_price_a=curr_price_a,
                curr_price_b=data.iloc[i]["price_b"],
                strategy=strategy,
                atr=data.iloc[i]["atr"],
                bb_width=data.iloc[i]["bb_width"],
                pred_price_a=pred[0]
            )
            final_capital = controller.bot_model.capital + controller.bot_model.position * curr_price_a
            capital_history.append(final_capital)
            timestamps.append(data.index[i])
        controller.view.display_final_results(
            final_capital,
            controller.bot_model.trade_log,
            controller.bot_model.trade_profits,
            controller.bot_model.strategy_counts,
            controller.bot_model.strategy_wins
        )

        # Plot capital
        plot_capital(capital_history, timestamps, ticker_a)

        # Real-time prediction
        if not live:
            controller.predict_real_time(ticker_a=ticker_a, ticker_b=ticker_b)

        # Live trading loop (uncomment for live mode, use with caution)
        """
        if live:
            def live_predict():
                print(f"\n[{datetime.now()}] Running live prediction...")
                controller.predict_real_time(ticker_a=ticker_a, ticker_b=ticker_b)
            
            schedule.every(5).minutes.do(live_predict)
            print("Starting live trading loop (Ctrl+C to stop)...")
            while True:
                schedule.run_pending()
                time.sleep(60)
        """
    except FileNotFoundError:
        print(f"Error: {data_file} not found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trading system with real-time data or CSV")
    parser.add_argument('--ticker_a', default='AAPL', help='Primary ticker (e.g., AAPL, BTC-USD)')
    parser.add_argument('--ticker_b', default='GOOG', help='Secondary ticker (optional, e.g., GOOG)')
    parser.add_argument('--interval', default='1d', help='Data interval (e.g., 1d, 1h, 1m)')
    parser.add_argument('--data_file', default=None, help='Path to CSV data file (optional, e.g., data.csv)')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode')
    args = parser.parse_args()

    run_trading(
        ticker_a=args.ticker_a,
        ticker_b=args.ticker_b,
        interval=args.interval,
        data_file=args.data_file,
        live=args.live
    )