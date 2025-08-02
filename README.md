# CSC Brain - Trading Strategy Backtesting Framework

A comprehensive Python framework for backtesting trading strategies with support for walk-forward testing, Monte Carlo simulations, and performance analysis.

## Features

- **Multiple Backtesting Methods**: Historical backtesting, walk-forward testing, and Monte Carlo simulations
- **Strategy Framework**: Easy-to-use interface for implementing custom trading strategies
- **Performance Analysis**: Built-in metrics and visualization tools
- **Production Ready**: Proper package structure with CLI interface

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

# Run the backtesting framework
run_backtest()
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
