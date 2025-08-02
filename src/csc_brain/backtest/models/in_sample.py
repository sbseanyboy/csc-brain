import pandas as pd

def backtest(df: pd.DataFrame, strategy_func) -> pd.DataFrame:
    """
    Perform a simple backtest of a strategy.
    
    Args:
        df: DataFrame with OHLCV data
        strategy_func: Strategy function that returns position signals
        
    Returns:
        DataFrame with backtest results including equity curve
    """
    df = df.copy()
    df['Position'] = strategy_func(df)
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Position']
    df['Equity_Curve'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def in_sample_test(df: pd.DataFrame, strategy_func) -> pd.DataFrame:
    """
    Perform in-sample testing of a strategy.
    
    Args:
        df: DataFrame with OHLCV data
        strategy_func: Strategy function that returns position signals
        
    Returns:
        DataFrame with backtest results including equity curve
    """
    return backtest(df, strategy_func) 