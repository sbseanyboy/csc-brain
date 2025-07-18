import yfinance as yf
import pandas as pd
from datetime import datetime

def download_multiple_tickers(tickers: list[str]) -> pd.DataFrame:
    data = {}
    for ticker in tickers:
        data[ticker] =  download_stock_data(ticker, f"data/{ticker}_historical_data.csv")
    return data;

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

