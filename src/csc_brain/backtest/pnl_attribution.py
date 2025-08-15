import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class AttributionResult:
    """Container for PnL attribution results."""
    total_pnl: float
    attribution_by_factor: Dict[str, float]
    attribution_by_timeframe: Dict[str, float]
    attribution_by_regime: Dict[str, float]
    attribution_by_signal_type: Dict[str, float]
    factor_contributions: pd.DataFrame
    cumulative_attribution: pd.DataFrame

class PnLAttribution:
    """
    System for attributing profits and losses to different factors.
    
    Breaks down strategy performance by:
    - Timeframes (1min, 5min, 15min, 1H, 1D)
    - Volatility regimes (low, medium, high)
    - Signal types (momentum, mean reversion, breakout)
    - Market conditions (trending, ranging, volatile)
    """
    
    def __init__(self, 
                 timeframes: List[str] = ['1min', '5min', '15min', '1H', '1D'],
                 volatility_regimes: List[str] = ['low', 'medium', 'high']):
        """
        Initialize the PnL attribution system.
        
        Args:
            timeframes: List of timeframes to analyze
            volatility_regimes: List of volatility regimes to analyze
        """
        self.timeframes = timeframes
        self.volatility_regimes = volatility_regimes
    
    def calculate_basic_attribution(self, df: pd.DataFrame) -> AttributionResult:
        """
        Calculate basic PnL attribution by different factors.
        
        Args:
            df: DataFrame with backtest results including positions and returns
            
        Returns:
            AttributionResult with attribution breakdown
        """
        df = df.copy()
        
        # Calculate basic metrics
        df['pnl'] = df['Strategy_Returns']
        df['cumulative_pnl'] = df['pnl'].cumsum()
        total_pnl = df['pnl'].sum()
        
        # Initialize attribution dictionaries
        attribution_by_factor = {}
        attribution_by_timeframe = {}
        attribution_by_regime = {}
        attribution_by_signal_type = {}
        
        # 1. Attribution by timeframes
        timeframe_attribution = self._attribute_by_timeframes(df)
        attribution_by_timeframe.update(timeframe_attribution)
        
        # 2. Attribution by volatility regimes
        regime_attribution = self._attribute_by_volatility_regimes(df)
        attribution_by_regime.update(regime_attribution)
        
        # 3. Attribution by signal types
        signal_attribution = self._attribute_by_signal_types(df)
        attribution_by_signal_type.update(signal_attribution)
        
        # 4. Attribution by market conditions
        market_attribution = self._attribute_by_market_conditions(df)
        attribution_by_factor.update(market_attribution)
        
        # Create factor contributions DataFrame
        factor_contributions = self._create_factor_contributions_df(df)
        
        # Create cumulative attribution DataFrame
        cumulative_attribution = self._create_cumulative_attribution_df(df)
        
        return AttributionResult(
            total_pnl=total_pnl,
            attribution_by_factor=attribution_by_factor,
            attribution_by_timeframe=attribution_by_timeframe,
            attribution_by_regime=attribution_by_regime,
            attribution_by_signal_type=attribution_by_signal_type,
            factor_contributions=factor_contributions,
            cumulative_attribution=cumulative_attribution
        )
    
    def _attribute_by_timeframes(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Attribute PnL by different timeframes.
        
        Args:
            df: DataFrame with backtest results
            
        Returns:
            Dictionary with PnL attribution by timeframe
        """
        attribution = {}
        
        # Calculate volatility for different timeframes
        for tf in self.timeframes:
            # Resample data to timeframe
            df_tf = self._resample_to_timeframe(df, tf)
            
            # Calculate PnL for this timeframe
            tf_pnl = df_tf['pnl'].sum()
            attribution[f'timeframe_{tf}'] = tf_pnl
        
        return attribution
    
    def _attribute_by_volatility_regimes(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Attribute PnL by volatility regimes.
        
        Args:
            df: DataFrame with backtest results
            
        Returns:
            Dictionary with PnL attribution by regime
        """
        df = df.copy()
        attribution = {}
        
        # Calculate volatility
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Identify regimes
        vol_33 = df['volatility'].quantile(0.33)
        vol_67 = df['volatility'].quantile(0.67)
        
        for regime in self.volatility_regimes:
            if regime == 'low':
                mask = df['volatility'] < vol_33
            elif regime == 'medium':
                mask = (df['volatility'] >= vol_33) & (df['volatility'] <= vol_67)
            else:  # high
                mask = df['volatility'] > vol_67
            
            regime_pnl = df.loc[mask, 'pnl'].sum()
            attribution[f'regime_{regime}'] = regime_pnl
        
        return attribution
    
    def _attribute_by_signal_types(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Attribute PnL by signal types.
        
        Args:
            df: DataFrame with backtest results
            
        Returns:
            Dictionary with PnL attribution by signal type
        """
        df = df.copy()
        attribution = {}
        
        # Identify signal types based on position changes
        df['position_change'] = df['Position'].diff()
        
        # Momentum signals (position increases)
        momentum_mask = df['position_change'] > 0
        momentum_pnl = df.loc[momentum_mask, 'pnl'].sum()
        attribution['signal_momentum'] = momentum_pnl
        
        # Mean reversion signals (position decreases)
        reversion_mask = df['position_change'] < 0
        reversion_pnl = df.loc[reversion_mask, 'pnl'].sum()
        attribution['signal_reversion'] = reversion_pnl
        
        # Hold signals (no position change)
        hold_mask = df['position_change'] == 0
        hold_pnl = df.loc[hold_mask, 'pnl'].sum()
        attribution['signal_hold'] = hold_pnl
        
        return attribution
    
    def _attribute_by_market_conditions(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Attribute PnL by market conditions.
        
        Args:
            df: DataFrame with backtest results
            
        Returns:
            Dictionary with PnL attribution by market condition
        """
        df = df.copy()
        attribution = {}
        
        # Calculate market conditions
        df['price_ma_short'] = df['Close'].rolling(10).mean()
        df['price_ma_long'] = df['Close'].rolling(50).mean()
        df['volatility'] = df['Close'].pct_change().rolling(20).std()
        
        # Trending market (price above/below long MA)
        trending_up = df['Close'] > df['price_ma_long']
        trending_down = df['Close'] < df['price_ma_long']
        
        # Ranging market (price near long MA)
        ranging = (df['Close'] >= df['price_ma_long'] * 0.98) & \
                 (df['Close'] <= df['price_ma_long'] * 1.02)
        
        # Volatile market (high volatility)
        volatile = df['volatility'] > df['volatility'].quantile(0.8)
        
        # Calculate PnL for each condition
        attribution['condition_trending_up'] = df.loc[trending_up, 'pnl'].sum()
        attribution['condition_trending_down'] = df.loc[trending_down, 'pnl'].sum()
        attribution['condition_ranging'] = df.loc[ranging, 'pnl'].sum()
        attribution['condition_volatile'] = df.loc[volatile, 'pnl'].sum()
        
        return attribution
    
    def _resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to a specific timeframe.
        
        Args:
            df: Original DataFrame
            timeframe: Target timeframe
            
        Returns:
            Resampled DataFrame
        """
        df_resampled = df.copy()
        df_resampled['timestamp'] = pd.to_datetime(df_resampled.index)
        df_resampled.set_index('timestamp', inplace=True)
        
        # Resample OHLCV and PnL data
        resampled = df_resampled.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Position': 'last',
            'Strategy_Returns': 'sum',
            'pnl': 'sum'
        })
        
        return resampled
    
    def _create_factor_contributions_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create DataFrame showing factor contributions over time.
        
        Args:
            df: DataFrame with backtest results
            
        Returns:
            DataFrame with factor contributions
        """
        df = df.copy()
        
        # Calculate rolling factor contributions
        window = 20
        
        # Volatility regime contribution
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        vol_33 = df['volatility'].quantile(0.33)
        vol_67 = df['volatility'].quantile(0.67)
        
        df['regime_low'] = (df['volatility'] < vol_33).astype(int)
        df['regime_medium'] = ((df['volatility'] >= vol_33) & (df['volatility'] <= vol_67)).astype(int)
        df['regime_high'] = (df['volatility'] > vol_67).astype(int)
        
        # Market condition contribution
        df['price_ma'] = df['Close'].rolling(50).mean()
        df['trending'] = (df['Close'] > df['price_ma']).astype(int)
        df['ranging'] = ((df['Close'] >= df['price_ma'] * 0.98) & 
                        (df['Close'] <= df['price_ma'] * 1.02)).astype(int)
        
        # Calculate rolling contributions
        contributions = pd.DataFrame(index=df.index)
        contributions['regime_low_contribution'] = df['regime_low'] * df['pnl']
        contributions['regime_medium_contribution'] = df['regime_medium'] * df['pnl']
        contributions['regime_high_contribution'] = df['regime_high'] * df['pnl']
        contributions['trending_contribution'] = df['trending'] * df['pnl']
        contributions['ranging_contribution'] = df['ranging'] * df['pnl']
        
        return contributions
    
    def _create_cumulative_attribution_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create DataFrame showing cumulative attribution over time.
        
        Args:
            df: DataFrame with backtest results
            
        Returns:
            DataFrame with cumulative attribution
        """
        factor_contributions = self._create_factor_contributions_df(df)
        cumulative = factor_contributions.cumsum()
        
        # Add total PnL for comparison
        cumulative['total_pnl'] = df['pnl'].cumsum()
        
        return cumulative
    
    def plot_attribution_breakdown(self, result: AttributionResult, 
                                 save_path: Optional[str] = None):
        """
        Plot PnL attribution breakdown.
        
        Args:
            result: AttributionResult object
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PnL Attribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Attribution by timeframe
        timeframe_data = {k.replace('timeframe_', ''): v for k, v in result.attribution_by_timeframe.items()}
        axes[0, 0].bar(timeframe_data.keys(), timeframe_data.values())
        axes[0, 0].set_title('Attribution by Timeframe')
        axes[0, 0].set_ylabel('PnL')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Attribution by regime
        regime_data = {k.replace('regime_', ''): v for k, v in result.attribution_by_regime.items()}
        axes[0, 1].bar(regime_data.keys(), regime_data.values())
        axes[0, 1].set_title('Attribution by Volatility Regime')
        axes[0, 1].set_ylabel('PnL')
        
        # 3. Attribution by signal type
        signal_data = {k.replace('signal_', ''): v for k, v in result.attribution_by_signal_type.items()}
        axes[1, 0].bar(signal_data.keys(), signal_data.values())
        axes[1, 0].set_title('Attribution by Signal Type')
        axes[1, 0].set_ylabel('PnL')
        
        # 4. Attribution by market condition
        condition_data = {k.replace('condition_', ''): v for k, v in result.attribution_by_factor.items() 
                         if k.startswith('condition_')}
        axes[1, 1].bar(condition_data.keys(), condition_data.values())
        axes[1, 1].set_title('Attribution by Market Condition')
        axes[1, 1].set_ylabel('PnL')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cumulative_attribution(self, result: AttributionResult,
                                  save_path: Optional[str] = None):
        """
        Plot cumulative attribution over time.
        
        Args:
            result: AttributionResult object
            save_path: Optional path to save the plot
        """
        cumulative = result.cumulative_attribution
        
        plt.figure(figsize=(15, 8))
        
        # Plot cumulative contributions
        for col in cumulative.columns:
            if col != 'total_pnl':
                plt.plot(cumulative.index, cumulative[col], label=col.replace('_contribution', ''))
        
        # Plot total PnL
        plt.plot(cumulative.index, cumulative['total_pnl'], 
                label='Total PnL', linewidth=2, color='black')
        
        plt.title('Cumulative PnL Attribution Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative PnL', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_attribution_report(self, result: AttributionResult) -> str:
        """
        Generate a text report of PnL attribution.
        
        Args:
            result: AttributionResult object
            
        Returns:
            String containing the attribution report
        """
        report = []
        report.append("=" * 60)
        report.append("PnL ATTRIBUTION REPORT")
        report.append("=" * 60)
        report.append(f"Total PnL: ${result.total_pnl:,.2f}")
        report.append("")
        
        # Timeframe attribution
        report.append("ATTRIBUTION BY TIMEFRAME:")
        report.append("-" * 30)
        for timeframe, pnl in result.attribution_by_timeframe.items():
            tf_name = timeframe.replace('timeframe_', '')
            percentage = (pnl / result.total_pnl * 100) if result.total_pnl != 0 else 0
            report.append(f"{tf_name:>10}: ${pnl:>10,.2f} ({percentage:>6.1f}%)")
        report.append("")
        
        # Regime attribution
        report.append("ATTRIBUTION BY VOLATILITY REGIME:")
        report.append("-" * 35)
        for regime, pnl in result.attribution_by_regime.items():
            reg_name = regime.replace('regime_', '')
            percentage = (pnl / result.total_pnl * 100) if result.total_pnl != 0 else 0
            report.append(f"{reg_name:>10}: ${pnl:>10,.2f} ({percentage:>6.1f}%)")
        report.append("")
        
        # Signal type attribution
        report.append("ATTRIBUTION BY SIGNAL TYPE:")
        report.append("-" * 30)
        for signal, pnl in result.attribution_by_signal_type.items():
            sig_name = signal.replace('signal_', '')
            percentage = (pnl / result.total_pnl * 100) if result.total_pnl != 0 else 0
            report.append(f"{sig_name:>10}: ${pnl:>10,.2f} ({percentage:>6.1f}%)")
        report.append("")
        
        # Market condition attribution
        report.append("ATTRIBUTION BY MARKET CONDITION:")
        report.append("-" * 35)
        for condition, pnl in result.attribution_by_factor.items():
            if condition.startswith('condition_'):
                cond_name = condition.replace('condition_', '')
                percentage = (pnl / result.total_pnl * 100) if result.total_pnl != 0 else 0
                report.append(f"{cond_name:>15}: ${pnl:>10,.2f} ({percentage:>6.1f}%)")
        
        return "\n".join(report) 