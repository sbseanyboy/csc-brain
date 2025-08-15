# CSC Brain - Advanced Trading Strategy Backtesting Framework

A comprehensive Python framework for backtesting trading strategies with support for walk-forward testing, Monte Carlo simulations, market microstructure analysis, volatility-based signal generation, and comprehensive PnL attribution.

## Features

- **Multiple Backtesting Methods**: Historical backtesting, walk-forward testing, and Monte Carlo simulations
- **Market Microstructure Analysis**: Limit order book data processing and order flow analysis
- **Volatility-Based Strategies**: Multi-timeframe volatility signal generation and regime detection
- **PnL Attribution**: Comprehensive profit/loss attribution across timeframes and market conditions
- **Strategy Framework**: Easy-to-use interface for implementing custom trading strategies
- **Performance Analysis**: Built-in metrics and visualization tools
- **Production Ready**: Proper package structure with CLI interface

## Advanced Features (Resume Points Implementation)

This framework now implements the sophisticated features mentioned in trading resumes:

### Point #2: Limit Order Book Data & Market Microstructure
- **Databento Integration**: Load and process high-frequency limit order book data
- **Microstructure Metrics**: Spread analysis, order imbalance, market depth, price impact
- **Order Flow Strategies**: Signal generation based on order flow patterns and market microstructure dynamics
- **Liquidity Provision**: Strategies for providing liquidity based on spread dynamics

### Point #3: Volatility-Based Signals & PnL Attribution
- **Multi-Timeframe Analysis**: Volatility signals across 1min, 5min, 15min, 1H, 1D timeframes
- **Volatility Regime Detection**: Adaptive strategies for low/medium/high volatility environments
- **Comprehensive PnL Attribution**: Break down profits/losses by:
  - Timeframes (which timeframes contribute most to PnL)
  - Volatility regimes (performance in different market conditions)
  - Signal types (momentum vs mean reversion vs breakout)
  - Market conditions (trending vs ranging vs volatile)

## Installation

### Development Installation

```bash
# Clone the repository
git clone <repository-url>
cd csc-brain

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Production Installation

```bash
pip install csc-brain
```

## Quick Start

### Using the CLI

```bash
# Run with default settings
csc-backtest

# Run with custom parameters
csc-backtest --data-file "path/to/data.csv" --strategy "moving_average" --output-dir "results"
```

### Using Python API

```python
from csc_brain import run_backtest

# Run the basic backtesting framework
run_backtest()

# Run advanced backtesting with microstructure and volatility analysis
from csc_brain import run_advanced_backtest

results = run_advanced_backtest(
    symbol="AAPL",
    start_date="2023-01-01", 
    end_date="2023-12-31"
)
```

### Advanced Usage Examples

```python
from csc_brain import (
    AdvancedBacktester,
    DatabentoDataLoader,
    MarketMicrostructureStrategy,
    VolatilityBreakoutStrategy,
    PnLAttribution
)

# 1. Market Microstructure Analysis
data_loader = DatabentoDataLoader()
lob_data = data_loader.load_limit_order_book("AAPL", "2023-01-01", "2023-12-31")
microstructure_data = data_loader.calculate_microstructure_metrics(lob_data)

# 2. Generate Microstructure Signals
strategy = MarketMicrostructureStrategy()
signals = strategy.generate_combined_signals(microstructure_data)

# 3. Volatility-Based Strategies
vol_strategy = VolatilityBreakoutStrategy()
vol_signals = vol_strategy.generate_signals(price_data)

# 4. PnL Attribution Analysis
attribution = PnLAttribution()
attribution_result = attribution.calculate_basic_attribution(backtest_results)
report = attribution.generate_attribution_report(attribution_result)
```

### Implementing Custom Strategies

```python
import pandas as pd

def my_custom_strategy(df: pd.DataFrame) -> pd.Series:
    """
    Custom trading strategy implementation.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        pd.Series: Position signals (1 for long, 0 for neutral, -1 for short)
    """
    # Your strategy logic here
    return position_signals
```

## Project Structure

```
csc-brain/
├── src/csc_brain/          # Main package source
│   ├── strategies/         # Trading strategies
│   ├── backtest/          # Backtesting engine
│   ├── data_ingestion/    # Data loading utilities
│   └── utils/             # Utility functions
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── examples/              # Example notebooks
├── pyproject.toml        # Modern Python project config
├── setup.py              # Legacy setup (for compatibility)
└── README.md             # This file
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=csc_brain

# Run specific test file
pytest tests/test_strategies.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build docs
cd docs && make html
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with pandas, numpy, matplotlib, and seaborn
- Inspired by modern backtesting frameworks
