import yfinance as yf
import pandas as pd
from datetime import datetime

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

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    output_file = f"data/{ticker}_historical_data.csv"
    df = download_stock_data(ticker, output_file)
    print(f"Downloaded {len(df)} days of data for {ticker}")
