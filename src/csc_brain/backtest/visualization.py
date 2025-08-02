import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_equity_curve(df: pd.DataFrame, title: str = "Equity Curve"):
    """
    Plot the equity curve of a trading strategy.
    
    Args:
        df: DataFrame containing equity curve data
        title: Title for the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot equity curve
    plt.plot(df.index, df['Equity_Curve'], label='Strategy Performance', linewidth=2, color='blue')
    
    # Add baseline (buy and hold)
    if 'Close' in df.columns:
        buy_hold = df['Close'] / df['Close'].iloc[0]
        plt.plot(df.index, buy_hold, label='Buy & Hold', linewidth=1, color='gray', alpha=0.7)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (1.0 = 100%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y-1)))
    
    plt.tight_layout()
    plt.show()

def plot_strategy_analysis(df: pd.DataFrame, title: str = "Strategy Analysis"):
    """
    Create a comprehensive strategy analysis plot with multiple subplots.
    
    Args:
        df: DataFrame containing backtest results
        title: Title for the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Equity Curve
    axes[0, 0].plot(df.index, df['Equity_Curve'], label='Strategy', linewidth=2, color='blue')
    if 'Close' in df.columns:
        buy_hold = df['Close'] / df['Close'].iloc[0]
        axes[0, 0].plot(df.index, buy_hold, label='Buy & Hold', linewidth=1, color='gray', alpha=0.7)
    axes[0, 0].set_title('Equity Curve')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Returns Distribution
    returns = df['Strategy_Returns'].dropna()
    axes[0, 1].hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_title('Returns Distribution')
    axes[0, 1].set_xlabel('Daily Returns')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Drawdown
    equity = df['Equity_Curve']
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    axes[1, 0].fill_between(df.index, drawdown, 0, alpha=0.3, color='red')
    axes[1, 0].plot(df.index, drawdown, color='red', linewidth=1)
    axes[1, 0].set_title('Drawdown')
    axes[1, 0].set_ylabel('Drawdown %')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Position Signals
    if 'Position' in df.columns:
        axes[1, 1].plot(df.index, df['Position'], linewidth=1, color='green')
        axes[1, 1].set_title('Position Signals')
        axes[1, 1].set_ylabel('Position (0/1)')
        axes[1, 1].set_ylim(-0.1, 1.1)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 