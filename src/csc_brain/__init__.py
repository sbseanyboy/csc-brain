"""
CSC Brain - Trading Strategy Backtesting Framework

A comprehensive framework for backtesting trading strategies with support for
walk-forward testing, Monte Carlo simulations, and performance analysis.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .strategies.moving_average_strategy import moving_average_strategy
from .backtest.backtester import run_backtest
from .backtest.models.in_sample import backtest, in_sample_test
from .backtest.models.walk_forward import walk_forward_test
from .backtest.models.monte_carlo import monte_carlo_simulation
from .backtest.metrics import sharpe_ratio, max_drawdown
from .backtest.visualization import plot_equity_curve, plot_strategy_analysis
from .data_ingestion.yahoo import download_stock_data, download_multiple_tickers, load_stock_data

# Advanced backtesting components
from .backtest.advanced_backtester import AdvancedBacktester, run_advanced_backtest
from .backtest.pnl_attribution import PnLAttribution, AttributionResult
from .data_ingestion.databento import DatabentoDataLoader
from .strategies.microstructure_models import (
    MarketMicrostructureStrategy, 
    OrderFlowReversalStrategy,
    LiquidityProvisionStrategy
)
from .strategies.volatility_strategies import (
    VolatilityBreakoutStrategy,
    VolatilityMeanReversionStrategy,
    MultiTimeframeVolatilityStrategy,
    VolatilityRegimeStrategy
)

__all__ = [
    "moving_average_strategy",
    "run_backtest",
    "backtest",
    "in_sample_test", 
    "walk_forward_test",
    "monte_carlo_simulation",
    "sharpe_ratio",
    "max_drawdown",
    "plot_equity_curve",
    "plot_strategy_analysis",
    "download_stock_data",
    "download_multiple_tickers", 
    "load_stock_data",
    # Advanced components
    "AdvancedBacktester",
    "run_advanced_backtest",
    "PnLAttribution",
    "AttributionResult",
    "DatabentoDataLoader",
    "MarketMicrostructureStrategy",
    "OrderFlowReversalStrategy",
    "LiquidityProvisionStrategy",
    "VolatilityBreakoutStrategy",
    "VolatilityMeanReversionStrategy",
    "MultiTimeframeVolatilityStrategy",
    "VolatilityRegimeStrategy",
] 