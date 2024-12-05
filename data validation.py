# Import necessary libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Fetch historical data for a stock
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"No data found for ticker: {ticker}")
        else:
            print(f"Data fetched successfully for ticker: {ticker}")
        return stock_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Step 2: Visualize the stock's closing prices
def visualize_stock_data(stock_data, ticker):
    if stock_data is not None and not stock_data.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data.index, stock_data['Close'], label=f"{ticker} Closing Prices")
        plt.title(f"{ticker} Historical Closing Prices")
        plt.xlabel("Date")
        plt.ylabel("Close Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No data to visualize.")

# Example usage
if __name__ == "__main__":
    # Input parameters
    ticker = "AAPL"  # Example ticker
    start_date = "2023-01-01"  # Start date for data
    end_date = "2023-12-01"    # End date for data

    # Fetch data
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    # Visualize data
    visualize_stock_data(stock_data, ticker)
