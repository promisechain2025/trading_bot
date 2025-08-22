import pandas as pd
from src.controller import TradingController

def main():
    # Load data
    try:
        data = pd.read_csv('data/sample_data.csv', parse_dates=['date'])
        # Clip negative prices to avoid invalid trades
        data['price_a'] = data['price_a'].clip(lower=0.01)
        data['price_b'] = data['price_b'].clip(lower=0.01)
        
        # Initialize controller
        controller = TradingController(capital=10000.0)
        
        # Train model
        controller.load_and_train(data)
        # Simulate trading
        controller.simulate_trading(data)
    except FileNotFoundError:
        print("Error: data.csv not found in /Users/luwamtesfalem/Downloads/AASB_Project/")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()