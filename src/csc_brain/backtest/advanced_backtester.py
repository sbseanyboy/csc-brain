import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..data_ingestion.databento import DatabentoDataLoader
from ..strategies.microstructure_models import (
    MarketMicrostructureStrategy, 
    OrderFlowReversalStrategy,
    LiquidityProvisionStrategy
)
from ..strategies.volatility_strategies import (
    VolatilityBreakoutStrategy,
    VolatilityMeanReversionStrategy,
    MultiTimeframeVolatilityStrategy,
    VolatilityRegimeStrategy
)
from .pnl_attribution import PnLAttribution, AttributionResult
from .models.in_sample import backtest
from .visualization import plot_equity_curve, plot_strategy_analysis

class AdvancedBacktester:
    """
    Advanced backtesting framework that integrates:
    - Limit order book data and market microstructure analysis
    - Volatility-based signal generation across multiple timeframes
    - Comprehensive PnL attribution analysis
    """
    
    def __init__(self, 
                 databento_api_key: Optional[str] = None,
                 use_sample_data: bool = True):
        """
        Initialize the advanced backtester.
        
        Args:
            databento_api_key: API key for Databento (optional for sample data)
            use_sample_data: Whether to use sample data instead of real API calls
        """
        self.use_sample_data = use_sample_data
        
        if not use_sample_data:
            self.data_loader = DatabentoDataLoader(databento_api_key)
        
        # Initialize strategies
        self.microstructure_strategy = MarketMicrostructureStrategy()
        self.order_flow_strategy = OrderFlowReversalStrategy()
        self.liquidity_strategy = LiquidityProvisionStrategy()
        
        self.volatility_breakout = VolatilityBreakoutStrategy()
        self.volatility_mean_reversion = VolatilityMeanReversionStrategy()
        self.multi_timeframe = MultiTimeframeVolatilityStrategy()
        self.regime_strategy = VolatilityRegimeStrategy()
        
        # Initialize PnL attribution
        self.pnl_attribution = PnLAttribution()
    
    def load_microstructure_data(self, 
                               symbol: str, 
                               start_date: str, 
                               end_date: str) -> pd.DataFrame:
        """
        Load and process microstructure data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with microstructure metrics
        """
        if self.use_sample_data:
            # Use sample data for development
            data_loader = DatabentoDataLoader()
            lob_data = data_loader.load_limit_order_book(symbol, start_date, end_date)
        else:
            # Use real Databento data
            lob_data = self.data_loader.load_limit_order_book(symbol, start_date, end_date)
        
        # Calculate microstructure metrics
        if self.use_sample_data:
            data_loader = DatabentoDataLoader()
            microstructure_data = data_loader.calculate_microstructure_metrics(lob_data)
        else:
            microstructure_data = self.data_loader.calculate_microstructure_metrics(lob_data)
        
        return microstructure_data
    
    def run_microstructure_backtest(self, 
                                  symbol: str,
                                  start_date: str,
                                  end_date: str,
                                  strategy_type: str = 'combined') -> pd.DataFrame:
        """
        Run backtest using microstructure strategies.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            strategy_type: Type of strategy ('order_flow', 'spread', 'depth', 'combined')
            
        Returns:
            DataFrame with backtest results
        """
        # Load microstructure data
        microstructure_data = self.load_microstructure_data(symbol, start_date, end_date)
        
        # Generate signals based on strategy type
        if strategy_type == 'order_flow':
            signals = self.microstructure_strategy.calculate_order_flow_signals(microstructure_data)
        elif strategy_type == 'spread':
            signals = self.microstructure_strategy.calculate_spread_signals(microstructure_data)
        elif strategy_type == 'depth':
            signals = self.microstructure_strategy.calculate_depth_signals(microstructure_data)
        elif strategy_type == 'combined':
            signals = self.microstructure_strategy.generate_combined_signals(microstructure_data)
        elif strategy_type == 'reversal':
            signals = self.order_flow_strategy.detect_reversals(microstructure_data)
        elif strategy_type == 'liquidity':
            signals = self.liquidity_strategy.generate_liquidity_signals(microstructure_data)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        # Add signals to data
        microstructure_data['Position'] = signals
        
        # Run backtest
        results = backtest(microstructure_data, lambda df: df['Position'])
        
        return results
    
    def run_volatility_backtest(self,
                               df: pd.DataFrame,
                               strategy_type: str = 'breakout') -> pd.DataFrame:
        """
        Run backtest using volatility-based strategies.
        
        Args:
            df: DataFrame with price data
            strategy_type: Type of volatility strategy
            
        Returns:
            DataFrame with backtest results
        """
        # Generate signals based on strategy type
        if strategy_type == 'breakout':
            signals = self.volatility_breakout.generate_signals(df)
        elif strategy_type == 'mean_reversion':
            signals = self.volatility_mean_reversion.generate_signals(df)
        elif strategy_type == 'multi_timeframe':
            signals = self.multi_timeframe.generate_multi_timeframe_signals(df)
        elif strategy_type == 'regime':
            signals = self.regime_strategy.generate_regime_adaptive_signals(df)
        else:
            raise ValueError(f"Unknown volatility strategy type: {strategy_type}")
        
        # Add signals to data
        df['Position'] = signals
        
        # Run backtest
        results = backtest(df, lambda df: df['Position'])
        
        return results
    
    def run_combined_backtest(self,
                            symbol: str,
                            start_date: str,
                            end_date: str,
                            microstructure_weight: float = 0.5,
                            volatility_weight: float = 0.5) -> pd.DataFrame:
        """
        Run combined backtest using both microstructure and volatility strategies.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            microstructure_weight: Weight for microstructure signals
            volatility_weight: Weight for volatility signals
            
        Returns:
            DataFrame with combined backtest results
        """
        # Load data
        microstructure_data = self.load_microstructure_data(symbol, start_date, end_date)
        
        # Generate microstructure signals
        micro_signals = self.microstructure_strategy.generate_combined_signals(microstructure_data)
        
        # Generate volatility signals
        vol_signals = self.volatility_breakout.generate_signals(microstructure_data)
        
        # Combine signals
        combined_signals = (micro_signals * microstructure_weight + 
                          vol_signals * volatility_weight)
        
        # Normalize to [-1, 1] range
        combined_signals = np.clip(combined_signals, -1, 1)
        
        # Add signals to data
        microstructure_data['Position'] = combined_signals
        
        # Run backtest
        results = backtest(microstructure_data, lambda df: df['Position'])
        
        return results
    
    def run_comprehensive_analysis(self,
                                 symbol: str,
                                 start_date: str,
                                 end_date: str) -> Dict[str, any]:
        """
        Run comprehensive analysis with all strategies and PnL attribution.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with all analysis results
        """
        results = {}
        
        # 1. Microstructure strategies
        micro_strategies = ['order_flow', 'spread', 'depth', 'combined', 'reversal', 'liquidity']
        for strategy in micro_strategies:
            try:
                micro_results = self.run_microstructure_backtest(symbol, start_date, end_date, strategy)
                results[f'microstructure_{strategy}'] = micro_results
            except Exception as e:
                print(f"Error running microstructure strategy {strategy}: {e}")
        
        # 2. Volatility strategies
        vol_strategies = ['breakout', 'mean_reversion', 'multi_timeframe', 'regime']
        for strategy in vol_strategies:
            try:
                # Use microstructure data for volatility strategies
                microstructure_data = self.load_microstructure_data(symbol, start_date, end_date)
                vol_results = self.run_volatility_backtest(microstructure_data, strategy)
                results[f'volatility_{strategy}'] = vol_results
            except Exception as e:
                print(f"Error running volatility strategy {strategy}: {e}")
        
        # 3. Combined strategy
        try:
            combined_results = self.run_combined_backtest(symbol, start_date, end_date)
            results['combined'] = combined_results
        except Exception as e:
            print(f"Error running combined strategy: {e}")
        
        # 4. PnL Attribution for combined strategy
        if 'combined' in results:
            try:
                attribution_result = self.pnl_attribution.calculate_basic_attribution(results['combined'])
                results['attribution'] = attribution_result
            except Exception as e:
                print(f"Error calculating PnL attribution: {e}")
        
        return results
    
    def generate_comprehensive_report(self, results: Dict[str, any]) -> str:
        """
        Generate a comprehensive report of all backtest results.
        
        Args:
            results: Dictionary with backtest results
            
        Returns:
            String containing the comprehensive report
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE BACKTESTING REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Strategy performance summary
        report.append("STRATEGY PERFORMANCE SUMMARY:")
        report.append("-" * 40)
        
        for strategy_name, strategy_results in results.items():
            if isinstance(strategy_results, pd.DataFrame) and 'Strategy_Returns' in strategy_results.columns:
                total_return = strategy_results['Strategy_Returns'].sum()
                sharpe_ratio = (strategy_results['Strategy_Returns'].mean() / 
                              strategy_results['Strategy_Returns'].std()) * np.sqrt(252)
                max_drawdown = (strategy_results['Equity_Curve'] / 
                              strategy_results['Equity_Curve'].cummax() - 1).min()
                
                report.append(f"{strategy_name:>25}: Return={total_return:>8.2%}, "
                            f"Sharpe={sharpe_ratio:>6.2f}, MaxDD={max_drawdown:>8.2%}")
        
        report.append("")
        
        # PnL Attribution report
        if 'attribution' in results:
            report.append("PnL ATTRIBUTION ANALYSIS:")
            report.append("-" * 30)
            attribution_report = self.pnl_attribution.generate_attribution_report(results['attribution'])
            report.append(attribution_report)
        
        return "\n".join(report)
    
    def plot_comprehensive_results(self, results: Dict[str, any], save_dir: Optional[str] = None):
        """
        Plot comprehensive results from all strategies.
        
        Args:
            results: Dictionary with backtest results
            save_dir: Optional directory to save plots
        """
        # Create save directory if specified
        if save_dir:
            Path(save_dir).mkdir(exist_ok=True)
        
        # Plot equity curves for all strategies
        plt.figure(figsize=(15, 10))
        
        for strategy_name, strategy_results in results.items():
            if isinstance(strategy_results, pd.DataFrame) and 'Equity_Curve' in strategy_results.columns:
                plt.plot(strategy_results.index, strategy_results['Equity_Curve'], 
                        label=strategy_name, linewidth=2)
        
        plt.title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_dir:
            plt.savefig(f"{save_dir}/strategy_comparison.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Plot PnL attribution if available
        if 'attribution' in results:
            self.pnl_attribution.plot_attribution_breakdown(results['attribution'])
            if save_dir:
                plt.savefig(f"{save_dir}/pnl_attribution.png", dpi=300, bbox_inches='tight')
            
            self.pnl_attribution.plot_cumulative_attribution(results['attribution'])
            if save_dir:
                plt.savefig(f"{save_dir}/cumulative_attribution.png", dpi=300, bbox_inches='tight')

def run_advanced_backtest(symbol: str = "AAPL",
                         start_date: str = "2023-01-01",
                         end_date: str = "2023-12-31",
                         save_results: bool = True):
    """
    Convenience function to run the complete advanced backtesting framework.
    
    Args:
        symbol: Stock symbol to test
        start_date: Start date for backtest
        end_date: End date for backtest
        save_results: Whether to save results and plots
    """
    print(f"Running advanced backtest for {symbol} from {start_date} to {end_date}")
    
    # Initialize backtester
    backtester = AdvancedBacktester(use_sample_data=True)
    
    # Run comprehensive analysis
    results = backtester.run_comprehensive_analysis(symbol, start_date, end_date)
    
    # Generate report
    report = backtester.generate_comprehensive_report(results)
    print(report)
    
    # Save report
    if save_results:
        with open(f"advanced_backtest_report_{symbol}.txt", "w") as f:
            f.write(report)
    
    # Plot results
    save_dir = f"results_{symbol}" if save_results else None
    backtester.plot_comprehensive_results(results, save_dir)
    
    return results 