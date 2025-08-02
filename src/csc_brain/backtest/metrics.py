import pandas as pd
import numpy as np

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sharpe ratio of a strategy.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        float: Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate the maximum drawdown of an equity curve.
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        float: Maximum drawdown as a percentage
    """
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min() 