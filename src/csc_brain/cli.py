#!/usr/bin/env python3
"""
Command-line interface for CSC Brain backtesting framework.
"""

import argparse
import sys
from pathlib import Path

from .backtest.backtester import run_backtest


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CSC Brain - Trading Strategy Backtesting Framework"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="DataIngestion/Data/AAPL_historical_data.csv",
        help="Path to the data file to use for backtesting",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="moving_average",
        help="Strategy to use for backtesting",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results and plots",
    )

    args = parser.parse_args()

    # Validate data file exists
    if not Path(args.data_file).exists():
        print(f"Error: Data file '{args.data_file}' not found.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(exist_ok=True)

    try:
        run_backtest()
        print("Backtesting completed successfully!")
    except Exception as e:
        print(f"Error during backtesting: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 