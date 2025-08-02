import pandas as pd
from ..strategies.moving_average_strategy import moving_average_strategy
from .models.walk_forward import walk_forward_test
from .models.monte_carlo import monte_carlo_simulation
from .models.in_sample import in_sample_test
from .visualization import plot_equity_curve, plot_strategy_analysis


def run_backtest():
    """
    Run the complete backtesting framework with multiple testing methods.
    """
    from ..data_ingestion.yahoo import load_stock_data
    data = load_stock_data("AAPL")

    # Run historical backtest
    hist_result = in_sample_test(data, moving_average_strategy)
    print("Historical backtest done.")
    plot_strategy_analysis(hist_result, "Moving Average Strategy - Full Analysis")

    # Run walk-forward test
    wf_results = walk_forward_test(data, moving_average_strategy, train_size=500, test_size=100)
    print(f"Walk-forward test completed with {len(wf_results)} folds.")

    # Plot first fold equity curve as example
    plot_equity_curve(wf_results[0], "Walk-forward Test Fold 1")

    # Run Monte Carlo simulations
    mc_results = monte_carlo_simulation(data, moving_average_strategy, n_simulations=20)
    print(f"Monte Carlo simulation completed with {len(mc_results)} runs.")

    # Plot first Monte Carlo run as example
    plot_equity_curve(mc_results[0], "Monte Carlo Simulation Run 1")


if __name__ == "__main__":
    run_backtest() 