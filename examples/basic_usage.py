#!/usr/bin/env python3
"""
Basic usage example for CSC Brain backtesting framework.
"""

import pandas as pd
from csc_brain import (
    moving_average_strategy,
    run_backtest,
    backtest,
    plot_equity_curve,
    sharpe_ratio,
    max_drawdown
)

def main():
    """Example of using the backtesting framework."""
    
    # Load data
    try:
        from csc_brain import load_stock_data
        data = load_stock_data("AAPL")
        print(f"Loaded data with {len(data)} rows")
    except FileNotFoundError:
        print("Data file not found. Please ensure DataIngestion/Data/AAPL_historical_data.csv exists.")
        return
    
    # Simple backtest
    print("\n=== Simple Backtest ===")
    result = backtest(data, moving_average_strategy)
    
    # Calculate metrics
    returns = result['Strategy_Returns'].dropna()
    sharpe = sharpe_ratio(returns)
    max_dd = max_drawdown(result['Equity_Curve'])
    
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Max Drawdown: {max_dd:.3f}")
    
    # Plot results
    plot_equity_curve(result, "Simple Moving Average Strategy")
    
    # Run full framework
    print("\n=== Full Framework ===")
    run_backtest()

if __name__ == "__main__":
    main() 