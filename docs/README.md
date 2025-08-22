# Adaptive AI Sentiment Bot (AASB)

A Python-based crypto trading bot using MVC architecture, LSTM for price prediction, and sentiment analysis for strategy selection.

## Setup
1. Clone the repo: `git clone <repo_url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure `data/sample_data.csv` exists (generated or real data).
4. Run: `python main.py`

## Requirements
- Python 3.8+
- See `requirements.txt`

## Usage
- Simulation: Uses `data/sample_data.csv` for backtesting.
- Real-time: Replace data fetch in `controllers.py` with API (e.g., ccxt).
- Warning: Crypto trading is high-risk. Test thoroughly.

## License
MIT License (see LICENSE file).
