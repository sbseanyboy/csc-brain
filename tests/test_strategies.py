"""
Tests for trading strategies.
"""

import pytest
import pandas as pd
import numpy as np

from csc_brain.strategies.moving_average_strategy import moving_average_strategy


class TestMovingAverageStrategy:
    """Test cases for the moving average strategy."""

    def test_moving_average_strategy_returns_series(self):
        """Test that the strategy returns a pandas Series."""
        # Create sample data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "Close": np.random.randn(100).cumsum() + 100,
            },
            index=dates,
        )

        result = moving_average_strategy(data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_moving_average_strategy_values_are_binary(self):
        """Test that the strategy returns only 0 and 1 values."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "Close": np.random.randn(100).cumsum() + 100,
            },
            index=dates,
        )

        result = moving_average_strategy(data)
        assert result.isin([0, 1]).all()

    def test_moving_average_strategy_handles_empty_data(self):
        """Test that the strategy handles empty data gracefully."""
        empty_data = pd.DataFrame({"Close": []})
        
        with pytest.raises(Exception):
            moving_average_strategy(empty_data) 