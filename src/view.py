class TradingView:
    def display_message(self, msg):
        print(msg)
    
    def display_error(self, err):
        print(err)
    
    def display_final_results(self, final_capital, trade_log, trade_profits, strategy_counts, strategy_wins):
        print(f"Final Capital: {final_capital:.2f}")
        print("Trade Log:")
        for entry in trade_log:
            print(entry)
        wins = sum(p > 0 for p in trade_profits)
        total = len(trade_profits) if trade_profits else 0
        win_rate = wins / total if total > 0 else 0
        avg_profit = sum(p for p in trade_profits if p > 0) / wins if wins > 0 else 0
        avg_loss = sum(p for p in trade_profits if p < 0) / (total - wins) if total > wins else 0
        print(f"Overall Win Rate: {win_rate * 100:.2f}%")
        print(f"Average Profit per Winning Trade: {avg_profit:.2f}")
        print(f"Average Loss per Losing Trade: {avg_loss:.2f}")
        print("Strategy Performance:")
        for strategy in strategy_counts:
            strat_wins = strategy_wins.get(strategy, 0)
            strat_total = strategy_counts.get(strategy, 0)
            strat_win_rate = strat_wins / strat_total if strat_total > 0 else 0
            print(f"{strategy.replace('_', ' ').title()}: {strat_wins}/{strat_total} ({strat_win_rate * 100:.2f}%)")