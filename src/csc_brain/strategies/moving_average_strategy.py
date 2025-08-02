import pandas as pd

def moving_average_strategy(df: pd.DataFrame) -> pd.Series:
    """
    Simple moving average crossover strategy.
    
    Args:
        df: DataFrame with OHLCV data including 'Close' column
        
    Returns:
        pd.Series: Position signals (1 for long, 0 for neutral)
    """
    sma = df['Close'].rolling(20).mean()
    position = (df['Close'] > sma).astype(int)
    return position.shift(1).fillna(0) 