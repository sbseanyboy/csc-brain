import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

def download_multiple_tickers(tickers: list[str]) -> dict:
    """
    Download historical data for multiple tickers.
    
    Args:
        tickers: List of stock ticker symbols
        
    Returns:
        dict: Dictionary mapping tickers to their DataFrames
    """
    data = {}
    BASE_DIR = Path(__file__).parent.parent.parent.parent  # Go to project root
    output_dir = BASE_DIR / "DataIngestion" / "Data"
    output_dir.mkdir(exist_ok=True)

    for ticker in tickers:
        output_path = output_dir / f"{ticker}_historical_data.csv"
        try: 
            data[ticker] = download_stock_data(ticker, output_path)
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")

    return data

def download_stock_data(ticker: str, output_path: str = None) -> pd.DataFrame:
    """
    Download daily OHLCV data for a given ticker from 2010 to 2024.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        output_path (str, optional): Path to save the CSV file. If None, returns DataFrame without saving.
    
    Returns:
        pd.DataFrame: DataFrame containing the historical stock data
    """
    # Create a Ticker object
    stock = yf.Ticker(ticker)
    
    # Download data from 2010 to current date
    start_date = '2010-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download the data
    df = stock.history(start=start_date, end=end_date)
    
    # Save to CSV if output_path is provided
    if output_path:
        df.to_csv(output_path)
        print(f"Data saved to {output_path}")
    
    return df

def load_stock_data(ticker: str) -> pd.DataFrame:
    """
    Load stock data from the Data directory.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        pd.DataFrame: Historical stock data
    """
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    data_path = BASE_DIR / "DataIngestion" / "Data" / f"{ticker}_historical_data.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file for {ticker} not found. Run download_stock_data first.")
    
    return pd.read_csv(data_path, index_col=0, parse_dates=True)

if __name__ == "__main__":
    # Example usage with 10 major tech stocks
    tech_tickers = [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "GOOGL",  # Alphabet (Google)
        "AMZN",   # Amazon
        "META",   # Meta Platforms (Facebook)
        "NVDA",   # NVIDIA
        "TSLA",   # Tesla
        "PLTR",   # Palantir
        "ADBE",   # Adobe
        "CRM"     # Salesforce
    ]
    data = download_multiple_tickers(tech_tickers)
    print(f"Downloaded data for: {', '.join(data.keys())}") 