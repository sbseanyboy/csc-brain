import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings

class VolatilityBreakoutStrategy:
    """
    Strategy that generates signals based on volatility breakouts.
    
    Identifies periods of unusual volatility and trades in the direction
    of the breakout with position sizing based on volatility levels.
    """
    
    def __init__(self, 
                 short_window: int = 10,
                 long_window: int = 50,
                 volatility_threshold: float = 2.0,
                 position_scaling: bool = True):
        """
        Initialize the volatility breakout strategy.
        
        Args:
            short_window: Short-term volatility window
            long_window: Long-term volatility window
            volatility_threshold: Threshold for volatility breakout
            position_scaling: Whether to scale positions by volatility
        """
        self.short_window = short_window
        self.long_window = long_window
        self.volatility_threshold = volatility_threshold
        self.position_scaling = position_scaling
    
    def calculate_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various volatility metrics.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with added volatility metrics
        """
        df = df.copy()
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        
        # Rolling volatility (standard deviation of returns)
        df['volatility_short'] = df['returns'].rolling(self.short_window).std()
        df['volatility_long'] = df['returns'].rolling(self.long_window).std()
        
        # Volatility ratio
        df['volatility_ratio'] = df['volatility_short'] / df['volatility_long']
        
        # Volatility z-score
        df['volatility_zscore'] = (df['volatility_short'] - df['volatility_long']) / \
                                 df['volatility_long'].rolling(self.long_window).std()
        
        # Realized volatility (annualized)
        df['realized_vol'] = df['returns'].rolling(self.short_window).std() * np.sqrt(252)
        
        # Parkinson volatility (using high-low range)
        if 'High' in df.columns and 'Low' in df.columns:
            df['hl_ratio'] = np.log(df['High'] / df['Low'])
            df['parkinson_vol'] = np.sqrt(df['hl_ratio'].rolling(self.short_window).var() / (4 * np.log(2)))
        
        return df
    
    def generate_breakout_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on volatility breakouts.
        
        Args:
            df: DataFrame with volatility metrics
            
        Returns:
            Series with position signals
        """
        signals = pd.Series(0, index=df.index)
        
        # Volatility breakout conditions
        high_volatility = df['volatility_ratio'] > self.volatility_threshold
        low_volatility = df['volatility_ratio'] < (1 / self.volatility_threshold)
        
        # Price momentum during volatility breakouts
        price_momentum = df['Close'].pct_change(self.short_window)
        
        # Generate signals
        # High volatility + positive momentum = long
        long_condition = high_volatility & (price_momentum > 0)
        # High volatility + negative momentum = short
        short_condition = high_volatility & (price_momentum < 0)
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals
    
    def calculate_position_sizes(self, df: pd.DataFrame, base_signals: pd.Series) -> pd.Series:
        """
        Calculate position sizes based on volatility.
        
        Args:
            df: DataFrame with volatility metrics
            base_signals: Base position signals
            
        Returns:
            Series with scaled position sizes
        """
        if not self.position_scaling:
            return base_signals
        
        # Inverse volatility scaling (smaller positions when volatility is high)
        volatility_scale = 1 / (1 + df['volatility_ratio'])
        
        # Normalize to reasonable position sizes
        max_position = 1.0
        scaled_signals = base_signals * volatility_scale * max_position
        
        return scaled_signals
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate complete volatility breakout signals.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Series with position signals
        """
        # Calculate volatility metrics
        df = self.calculate_volatility_metrics(df)
        
        # Generate base signals
        base_signals = self.generate_breakout_signals(df)
        
        # Scale positions by volatility
        final_signals = self.calculate_position_sizes(df, base_signals)
        
        return final_signals

class VolatilityMeanReversionStrategy:
    """
    Strategy that trades mean reversion during high volatility periods.
    """
    
    def __init__(self, 
                 volatility_window: int = 20,
                 reversion_window: int = 10,
                 volatility_threshold: float = 1.5):
        self.volatility_window = volatility_window
        self.reversion_window = reversion_window
        self.volatility_threshold = volatility_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate mean reversion signals during high volatility.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Series with position signals
        """
        df = df.copy()
        signals = pd.Series(0, index=df.index)
        
        # Calculate volatility
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.volatility_window).std()
        df['volatility_ma'] = df['volatility'].rolling(self.volatility_window).mean()
        
        # High volatility periods
        high_vol = df['volatility'] > (df['volatility_ma'] * self.volatility_threshold)
        
        # Price deviation from moving average
        df['price_ma'] = df['Close'].rolling(self.reversion_window).mean()
        df['price_deviation'] = (df['Close'] - df['price_ma']) / df['price_ma']
        
        # Z-score of price deviation
        df['deviation_zscore'] = df['price_deviation'].rolling(self.reversion_window).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        # Generate mean reversion signals during high volatility
        long_condition = high_vol & (df['deviation_zscore'] < -1.5)
        short_condition = high_vol & (df['deviation_zscore'] > 1.5)
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals

class MultiTimeframeVolatilityStrategy:
    """
    Strategy that combines volatility signals across multiple timeframes.
    """
    
    def __init__(self, timeframes: List[str] = ['1min', '5min', '15min', '1H']):
        self.timeframes = timeframes
    
    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to different timeframe.
        
        Args:
            df: Original DataFrame
            timeframe: Target timeframe
            
        Returns:
            Resampled DataFrame
        """
        df_resampled = df.copy()
        df_resampled['timestamp'] = pd.to_datetime(df_resampled.index)
        df_resampled.set_index('timestamp', inplace=True)
        
        # Resample OHLCV data
        resampled = df_resampled.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        return resampled
    
    def calculate_timeframe_volatility(self, df: pd.DataFrame, timeframe: str) -> pd.Series:
        """
        Calculate volatility for a specific timeframe.
        
        Args:
            df: DataFrame with price data
            timeframe: Timeframe to calculate volatility for
            
        Returns:
            Series with volatility values
        """
        df_tf = self.resample_data(df, timeframe)
        returns = df_tf['Close'].pct_change()
        volatility = returns.rolling(20).std()
        
        # Resample back to original frequency
        volatility_original = volatility.reindex(df.index, method='ffill')
        
        return volatility_original
    
    def generate_multi_timeframe_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on volatility across multiple timeframes.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Series with position signals
        """
        signals = pd.Series(0, index=df.index)
        
        # Calculate volatility for each timeframe
        volatility_signals = {}
        
        for tf in self.timeframes:
            vol = self.calculate_timeframe_volatility(df, tf)
            vol_ma = vol.rolling(50).mean()
            vol_zscore = (vol - vol_ma) / vol.rolling(50).std()
            
            # Generate signals for this timeframe
            tf_signals = pd.Series(0, index=df.index)
            tf_signals[vol_zscore > 1.5] = 1
            tf_signals[vol_zscore < -1.5] = -1
            
            volatility_signals[tf] = tf_signals
        
        # Combine signals from all timeframes
        combined_signals = pd.DataFrame(volatility_signals)
        
        # Weight by timeframe (shorter timeframes get higher weight)
        weights = {'1min': 0.4, '5min': 0.3, '15min': 0.2, '1H': 0.1}
        
        weighted_signals = pd.Series(0, index=df.index)
        for tf, weight in weights.items():
            if tf in combined_signals.columns:
                weighted_signals += combined_signals[tf] * weight
        
        # Apply threshold
        signals[weighted_signals > 0.3] = 1
        signals[weighted_signals < -0.3] = -1
        
        return signals

class VolatilityRegimeStrategy:
    """
    Strategy that adapts to different volatility regimes.
    """
    
    def __init__(self, regime_window: int = 100):
        self.regime_window = regime_window
    
    def identify_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Identify the current volatility regime.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Series with regime labels (low, medium, high)
        """
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.regime_window).std()
        
        # Calculate regime thresholds
        vol_33 = df['volatility'].rolling(self.regime_window).quantile(0.33)
        vol_67 = df['volatility'].rolling(self.regime_window).quantile(0.67)
        
        # Identify regimes
        regimes = pd.Series('medium', index=df.index)
        regimes[df['volatility'] < vol_33] = 'low'
        regimes[df['volatility'] > vol_67] = 'high'
        
        return regimes
    
    def generate_regime_adaptive_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals that adapt to volatility regimes.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Series with position signals
        """
        df = df.copy()
        signals = pd.Series(0, index=df.index)
        
        # Identify regimes
        regimes = self.identify_volatility_regime(df)
        
        # Calculate momentum
        df['momentum_short'] = df['Close'].pct_change(5)
        df['momentum_long'] = df['Close'].pct_change(20)
        
        # Generate regime-specific signals
        for regime in ['low', 'medium', 'high']:
            regime_mask = regimes == regime
            
            if regime == 'low':
                # Trend following in low volatility
                signals[regime_mask & (df['momentum_long'] > 0.01)] = 1
                signals[regime_mask & (df['momentum_long'] < -0.01)] = -1
                
            elif regime == 'medium':
                # Mean reversion in medium volatility
                df['price_ma'] = df['Close'].rolling(10).mean()
                df['deviation'] = (df['Close'] - df['price_ma']) / df['price_ma']
                signals[regime_mask & (df['deviation'] < -0.02)] = 1
                signals[regime_mask & (df['deviation'] > 0.02)] = -1
                
            elif regime == 'high':
                # Breakout trading in high volatility
                signals[regime_mask & (df['momentum_short'] > 0.02)] = 1
                signals[regime_mask & (df['momentum_short'] < -0.02)] = -1
        
        return signals 