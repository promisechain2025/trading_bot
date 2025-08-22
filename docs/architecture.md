# AASB Architecture

The project follows Model-View-Controller (MVC) pattern:
- **Model**: `TradingModel` (LSTM predictions), `BotModel` (trading logic, state).
- **View**: `TradingView` (displays results/errors).
- **Controller**: `TradingController` (orchestrates data flow).

Data flow:
1. Controller loads/trains model with data.
2. Model predicts prices and selects strategies.
3. Controller executes trades via BotModel.
4. View displays results.

Future: Add real-time APIs, UI dashboard, advanced ML.
