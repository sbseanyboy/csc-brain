from typing import Callable, List, Tuple
import pandas as pd
from .in_sample import backtest

def walk_forward_splits(df: pd.DataFrame, train_size: int, test_size: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward splits for time series cross-validation.
    
    Args:
        df: DataFrame with time series data
        train_size: Number of periods for training
        test_size: Number of periods for testing
        
    Returns:
        List of (train, test) DataFrame tuples
    """
    splits = []
    start = 0
    while start + train_size + test_size <= len(df):
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + test_size]
        splits.append((train, test))
        start += test_size
    return splits

def walk_forward_test(df: pd.DataFrame, strategy_func: Callable, train_size: int, test_size: int):
    """
    Perform walk-forward testing on a strategy.
    
    Args:
        df: DataFrame with time series data
        strategy_func: Strategy function to test
        train_size: Number of periods for training
        test_size: Number of periods for testing
        
    Returns:
        List of backtest results for each fold
    """
    results = []
    splits = walk_forward_splits(df, train_size, test_size)
    for i, (train, test) in enumerate(splits):
        # Optional: optimize on train here
        test_result = backtest(test, strategy_func)
        results.append(test_result)
        print(f"Walk-forward fold {i + 1} done.")
    return results 