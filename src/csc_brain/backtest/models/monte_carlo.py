import pandas as pd
from .in_sample import backtest

def monte_carlo_simulation(df: pd.DataFrame, strategy_func, n_simulations: int = 100):
    """
    Perform Monte Carlo simulation using bootstrap resampling.
    
    Args:
        df: DataFrame with time series data
        strategy_func: Strategy function to test
        n_simulations: Number of Monte Carlo simulations
        
    Returns:
        List of backtest results for each simulation
    """
    results = []
    daily_returns = df['Close'].pct_change().dropna()
    for i in range(n_simulations):
        bootstrapped_returns = daily_returns.sample(len(daily_returns), replace=True).reset_index(drop=True)
        price_series = 100 * (1 + bootstrapped_returns).cumprod()
        sim_df = pd.DataFrame({'Close': price_series})
        sim_result = backtest(sim_df, strategy_func)
        results.append(sim_result)
        print(f"Monte Carlo simulation {i+1} completed.")
    return results 