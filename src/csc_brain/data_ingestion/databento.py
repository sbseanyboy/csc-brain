import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os

class DatabentoDataLoader:
    """
    Loader for Databento limit order book data.
    
    This class handles the ingestion and preprocessing of high-frequency
    market microstructure data for backtesting.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Databento data loader.
        
        Args:
            api_key: Databento API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('DATABENTO_API_KEY')
        if not self.api_key:
            raise ValueError("Databento API key required. Set DATABENTO_API_KEY environment variable.")
        
        # Initialize Databento client (you'll need to install databento package)
        # self.client = databento.Historical(self.api_key)
    
    def load_limit_order_book(self, 
                            symbol: str, 
                            start_date: str, 
                            end_date: str,
                            levels: int = 10) -> pd.DataFrame:
        """
        Load limit order book data for a given symbol and date range.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            levels: Number of price levels to include
            
        Returns:
            DataFrame with LOB data including bid/ask prices and sizes
        """
        # This is a placeholder implementation
        # You'll need to implement actual Databento API calls
        
        # Example structure of LOB data:
        columns = ['timestamp', 'symbol', 'bid_price_1', 'bid_size_1', 'ask_price_1', 'ask_size_1']
        for i in range(2, levels + 1):
            columns.extend([f'bid_price_{i}', f'bid_size_{i}', f'ask_price_{i}', f'ask_size_{i}'])
        
        # Placeholder data - replace with actual API call
        data = self._generate_sample_lob_data(symbol, start_date, end_date, levels)
        
        return pd.DataFrame(data, columns=columns)
    
    def _generate_sample_lob_data(self, symbol: str, start_date: str, end_date: str, levels: int) -> List:
        """
        Generate sample LOB data for development/testing.
        Replace this with actual Databento API calls.
        """
        # Generate sample timestamps
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate data every second during market hours (9:30-16:00)
        timestamps = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # Weekdays only
                for hour in range(9, 16):
                    for minute in range(60):
                        for second in range(60):
                            if hour == 9 and minute < 30:  # Before market open
                                continue
                            if hour == 16 and minute > 0:  # After market close
                                continue
                            timestamps.append(current.replace(hour=hour, minute=minute, second=second))
            current += timedelta(days=1)
        
        # Generate sample LOB data
        data = []
        base_price = 150.0  # Example base price
        
        for ts in timestamps[:1000]:  # Limit to 1000 records for demo
            row = [ts, symbol]
            
            # Generate bid/ask prices and sizes
            for level in range(1, levels + 1):
                spread = 0.01 * level  # Increasing spread for deeper levels
                bid_price = base_price - spread - np.random.normal(0, 0.001)
                ask_price = base_price + spread + np.random.normal(0, 0.001)
                
                bid_size = np.random.randint(100, 10000)
                ask_size = np.random.randint(100, 10000)
                
                row.extend([bid_price, bid_size, ask_price, ask_size])
            
            data.append(row)
        
        return data
    
    def calculate_microstructure_metrics(self, lob_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market microstructure metrics from LOB data.
        
        Args:
            lob_data: DataFrame with limit order book data
            
        Returns:
            DataFrame with calculated microstructure metrics
        """
        df = lob_data.copy()
        
        # Basic metrics
        df['spread'] = df['ask_price_1'] - df['bid_price_1']
        df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
        df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000  # Basis points
        
        # Order book imbalance
        df['bid_volume'] = df['bid_size_1']
        df['ask_volume'] = df['ask_size_1']
        df['order_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        
        # Depth metrics (using first 5 levels)
        for i in range(1, 6):
            if f'bid_size_{i}' in df.columns and f'ask_size_{i}' in df.columns:
                df[f'bid_depth_{i}'] = df[f'bid_size_{i}']
                df[f'ask_depth_{i}'] = df[f'ask_size_{i}']
        
        # Price impact (simplified)
        df['price_impact'] = df['spread'] * df['order_imbalance']
        
        return df
    
    def resample_to_timeframe(self, lob_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample high-frequency LOB data to different timeframes.
        
        Args:
            lob_data: DataFrame with high-frequency LOB data
            timeframe: Target timeframe ('1min', '5min', '1H', '1D')
            
        Returns:
            DataFrame resampled to target timeframe
        """
        df = lob_data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample mid price
        resampled = df['mid_price'].resample(timeframe).ohlc()
        
        # Resample other metrics
        resampled['spread'] = df['spread'].resample(timeframe).mean()
        resampled['order_imbalance'] = df['order_imbalance'].resample(timeframe).mean()
        resampled['volume'] = df['bid_volume'].resample(timeframe).sum() + df['ask_volume'].resample(timeframe).sum()
        
        return resampled 