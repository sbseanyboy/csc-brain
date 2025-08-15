import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats

class MarketMicrostructureStrategy:
    """
    Trading strategy based on market microstructure dynamics.
    
    Uses limit order book data to identify short-term price movements
    based on order flow imbalances, spread dynamics, and market depth.
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 imbalance_threshold: float = 0.3,
                 spread_threshold: float = 0.0005):
        """
        Initialize the microstructure strategy.
        
        Args:
            lookback_period: Period for calculating rolling statistics
            imbalance_threshold: Threshold for order imbalance signal
            spread_threshold: Threshold for spread-based signals
        """
        self.lookback_period = lookback_period
        self.imbalance_threshold = imbalance_threshold
        self.spread_threshold = spread_threshold
    
    def calculate_order_flow_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on order flow imbalance.
        
        Args:
            df: DataFrame with microstructure metrics
            
        Returns:
            Series with position signals (1: long, -1: short, 0: neutral)
        """
        signals = pd.Series(0, index=df.index)
        
        # Calculate rolling order imbalance
        df['imbalance_ma'] = df['order_imbalance'].rolling(self.lookback_period).mean()
        df['imbalance_std'] = df['order_imbalance'].rolling(self.lookback_period).std()
        
        # Z-score of current imbalance
        df['imbalance_zscore'] = (df['order_imbalance'] - df['imbalance_ma']) / df['imbalance_std']
        
        # Generate signals based on extreme imbalances
        long_condition = df['imbalance_zscore'] > self.imbalance_threshold
        short_condition = df['imbalance_zscore'] < -self.imbalance_threshold
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals
    
    def calculate_spread_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on spread dynamics.
        
        Args:
            df: DataFrame with microstructure metrics
            
        Returns:
            Series with position signals
        """
        signals = pd.Series(0, index=df.index)
        
        # Calculate spread changes
        df['spread_change'] = df['spread'].diff()
        df['spread_ma'] = df['spread'].rolling(self.lookback_period).mean()
        
        # Normalize spread changes
        df['spread_normalized'] = (df['spread'] - df['spread_ma']) / df['spread_ma']
        
        # Generate signals based on spread compression/expansion
        # Narrowing spreads often precede price moves
        spread_compression = df['spread_normalized'] < -self.spread_threshold
        spread_expansion = df['spread_normalized'] > self.spread_threshold
        
        signals[spread_compression] = 1  # Potential breakout
        signals[spread_expansion] = -1   # Potential reversal
        
        return signals
    
    def calculate_depth_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on market depth analysis.
        
        Args:
            df: DataFrame with microstructure metrics
            
        Returns:
            Series with position signals
        """
        signals = pd.Series(0, index=df.index)
        
        # Calculate depth imbalance across levels
        depth_columns = [col for col in df.columns if 'depth_' in col]
        
        if len(depth_columns) >= 2:
            # Calculate total depth on each side
            bid_depth_cols = [col for col in depth_columns if 'bid_' in col]
            ask_depth_cols = [col for col in depth_columns if 'ask_' in col]
            
            if bid_depth_cols and ask_depth_cols:
                df['total_bid_depth'] = df[bid_depth_cols].sum(axis=1)
                df['total_ask_depth'] = df[ask_depth_cols].sum(axis=1)
                
                # Depth imbalance
                df['depth_imbalance'] = (df['total_bid_depth'] - df['total_ask_depth']) / \
                                      (df['total_bid_depth'] + df['total_ask_depth'])
                
                # Rolling statistics
                df['depth_ma'] = df['depth_imbalance'].rolling(self.lookback_period).mean()
                df['depth_std'] = df['depth_imbalance'].rolling(self.lookback_period).std()
                
                # Z-score
                df['depth_zscore'] = (df['depth_imbalance'] - df['depth_ma']) / df['depth_std']
                
                # Generate signals
                long_condition = df['depth_zscore'] > self.imbalance_threshold
                short_condition = df['depth_zscore'] < -self.imbalance_threshold
                
                signals[long_condition] = 1
                signals[short_condition] = -1
        
        return signals
    
    def calculate_volume_profile_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on volume profile analysis.
        
        Args:
            df: DataFrame with microstructure metrics
            
        Returns:
            Series with position signals
        """
        signals = pd.Series(0, index=df.index)
        
        # Calculate volume-weighted metrics
        df['vwap'] = (df['mid_price'] * df['volume']).rolling(self.look_period).sum() / \
                     df['volume'].rolling(self.lookback_period).sum()
        
        # Price relative to VWAP
        df['price_vwap_ratio'] = df['mid_price'] / df['vwap']
        
        # Volume momentum
        df['volume_ma'] = df['volume'].rolling(self.lookback_period).mean()
        df['volume_momentum'] = df['volume'] / df['volume_ma']
        
        # Generate signals
        high_volume_above_vwap = (df['price_vwap_ratio'] > 1.001) & (df['volume_momentum'] > 1.5)
        high_volume_below_vwap = (df['price_vwap_ratio'] < 0.999) & (df['volume_momentum'] > 1.5)
        
        signals[high_volume_above_vwap] = 1
        signals[high_volume_below_vwap] = -1
        
        return signals
    
    def generate_combined_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Combine all microstructure signals into a single signal.
        
        Args:
            df: DataFrame with microstructure metrics
            
        Returns:
            Series with combined position signals
        """
        # Get individual signals
        order_flow_sig = self.calculate_order_flow_signals(df)
        spread_sig = self.calculate_spread_signals(df)
        depth_sig = self.calculate_depth_signals(df)
        volume_sig = self.calculate_volume_profile_signals(df)
        
        # Combine signals (simple average)
        combined = (order_flow_sig + spread_sig + depth_sig + volume_sig) / 4
        
        # Apply threshold to get final positions
        final_signals = pd.Series(0, index=df.index)
        final_signals[combined > 0.25] = 1
        final_signals[combined < -0.25] = -1
        
        return final_signals

class OrderFlowReversalStrategy:
    """
    Strategy that identifies potential reversals based on order flow patterns.
    """
    
    def __init__(self, window: int = 50, threshold: float = 2.0):
        self.window = window
        self.threshold = threshold
    
    def detect_reversals(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect potential reversal points based on order flow.
        
        Args:
            df: DataFrame with microstructure metrics
            
        Returns:
            Series with reversal signals
        """
        signals = pd.Series(0, index=df.index)
        
        # Calculate price momentum
        df['price_momentum'] = df['mid_price'].pct_change(self.window)
        
        # Calculate order flow momentum
        df['flow_momentum'] = df['order_imbalance'].rolling(self.window).mean()
        
        # Detect divergences
        # Price up but order flow negative (potential reversal down)
        bearish_divergence = (df['price_momentum'] > 0.01) & (df['flow_momentum'] < -0.1)
        
        # Price down but order flow positive (potential reversal up)
        bullish_divergence = (df['price_momentum'] < -0.01) & (df['flow_momentum'] > 0.1)
        
        signals[bearish_divergence] = -1
        signals[bullish_divergence] = 1
        
        return signals

class LiquidityProvisionStrategy:
    """
    Strategy that provides liquidity based on spread dynamics.
    """
    
    def __init__(self, min_spread_bps: float = 5.0, max_position_size: float = 0.1):
        self.min_spread_bps = min_spread_bps
        self.max_position_size = max_position_size
    
    def generate_liquidity_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals for liquidity provision.
        
        Args:
            df: DataFrame with microstructure metrics
            
        Returns:
            Series with liquidity provision signals
        """
        signals = pd.Series(0, index=df.index)
        
        # Only provide liquidity when spreads are wide enough
        wide_spreads = df['spread_bps'] > self.min_spread_bps
        
        # Calculate optimal position size based on spread width
        position_size = np.minimum(df['spread_bps'] / 100, self.max_position_size)
        
        # Generate signals (smaller positions for liquidity provision)
        signals[wide_spreads] = position_size[wide_spreads] * 0.5
        
        return signals 