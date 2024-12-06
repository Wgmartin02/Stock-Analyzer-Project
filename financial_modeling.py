import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime

# Global variables
stock_data = None
betas = {}

# Function to fetch stock data using yfinance
def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data for a list of tickers.
    """
    try:
        print(f"Fetching data for tickers: {tickers} from {start_date} to {end_date}")
        data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker", auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.stack(level=0).reset_index()
            data.rename(columns={"level_1": "Ticker"}, inplace=True)
        else:
            data["Ticker"] = tickers[0]
        data = data.reset_index()  # Ensure 'Date' is a column
        print(f"Data fetched successfully for {tickers}.")
        return data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch data: {e}")
        return None

# Function to fetch today's closing price
def fetch_todays_closing_prices():
    tickers = ticker_entry.get().strip().split(",")
    if not tickers or tickers == [""]:
        messagebox.showwarning("Input Error", "Please enter at least one stock ticker.")
        return

    try:
        latest_prices = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            todays_data = stock.history(period="1d")
            if not todays_data.empty:
                latest_prices[ticker] = todays_data.iloc[-1]["Close"]
            else:
                latest_prices[ticker] = "No Data"

        closing_text = "\n".join([f"{ticker}: ${price:.2f}" if isinstance(price, (int, float)) else f"{ticker}: {price}" for ticker, price in latest_prices.items()])
        messagebox.showinfo("Today's Closing Prices", f"Today's Closing Prices:\n\n{closing_text}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch today's closing prices: {e}")

# Function to calculate daily returns
def calculate_daily_returns(data):
    data['Daily Return'] = data.groupby('Ticker')['Close'].pct_change()
    return data

# Function to calculate cumulative returns
def calculate_cumulative_returns(data):
    data['Cumulative Return'] = (1 + data['Daily Return']).groupby(data['Ticker']).cumprod()
    return data

# Function to calculate correlation matrix
def calculate_correlations(data):
    pivot_data = data.pivot(index="Date", columns="Ticker", values="Close")
    correlations = pivot_data.corr()
    return correlations

# Function to calculate beta values
def calculate_betas(data, market_ticker="^GSPC"):
    """
    Calculate beta values for selected stocks relative to the market.
    """
    global betas
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    # Fetch market data
    print(f"Fetching market data for: {market_ticker}")
    market_data = fetch_stock_data([market_ticker], start_date, end_date)
    if market_data is None or market_data.empty:
        messagebox.showerror("Error", f"Market data for {market_ticker} could not be fetched.")
        return {}

    # Calculate daily returns for the market
    market_data = calculate_daily_returns(market_data)
    market_returns = market_data[['Date', 'Daily Return']].rename(columns={'Daily Return': 'Market Return'})

    # Ensure proper date alignment and remove time zones
    if 'Date' not in data.columns or 'Date' not in market_returns.columns:
        messagebox.showerror("Error", "Date column is missing in the data.")
        return {}

    market_returns['Date'] = pd.to_datetime(market_returns['Date'], errors='coerce').dt.tz_localize(None)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.tz_localize(None)

    # Drop rows with invalid dates
    market_returns = market_returns.dropna(subset=['Date'])
    data = data.dropna(subset=['Date'])

    # Initialize a dictionary to store beta values
    betas = {}
    for ticker in data['Ticker'].unique():
        print(f"Calculating beta for ticker: {ticker}")
        stock_returns = data[data['Ticker'] == ticker][['Date', 'Daily Return']]

        # Merge stock returns with market returns on Date
        merged = pd.merge(stock_returns, market_returns, on='Date', how='inner')
        print(f"Merged Data (first 5 rows) for {ticker}:\n", merged.head())

        # Drop rows with NaN or infinite values
        merged = merged.dropna()
        merged = merged[~merged.isin([np.inf, -np.inf]).any(axis=1)]

        # Check if merged data is empty
        if merged.empty:
            print(f"No valid overlapping data for {ticker}. Skipping beta calculation.")
            betas[ticker] = np.nan
            continue

        # Perform linear regression to calculate beta
        X = merged['Market Return']
        Y = merged['Daily Return']
        X = sm.add_constant(X)  # Add a constant for the intercept
        try:
            model = sm.OLS(Y, X).fit()
            betas[ticker] = model.params['Market Return']
            print(f"Beta for {ticker}: {betas[ticker]}")
        except Exception as e:
            print(f"Error calculating beta for {ticker}: {e}")
            betas[ticker] = np.nan

    print("Final calculated betas:", betas)  # Debugging statement
    return betas

# Function to plot historical prices
def plot_prices(data):
    if 'Date' not in data.columns:
        messagebox.showerror("Error", "'Date' column is missing in the data.")
        return
    
    plt.figure(figsize=(10, 6))
    for ticker in data['Ticker'].unique():
        stock_data = data[data['Ticker'] == ticker]
        plt.plot(stock_data['Date'], stock_data['Close'], label=ticker)
    plt.legend()
    plt.title("Historical Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

# Function to plot correlation heatmap
def plot_correlation_heatmap(correlations):
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlations, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

# Validate date input
def validate_date(date_str):
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

# Fetch data button action
def fetch_data():
    global stock_data, betas
    tickers = ticker_entry.get().strip().split(",")
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    # Validate date range
    if not start_date or not end_date:
        messagebox.showwarning("Input Error", "Please enter both start and end dates.")
        return

    if not validate_date(start_date) or not validate_date(end_date):
        messagebox.showwarning("Input Error", "Invalid date format. Please use YYYY-MM-DD.")
        return

    if not tickers or tickers == [""]:
        messagebox.showwarning("Input Error", "Please enter at least one stock ticker.")
        return

    data = fetch_stock_data(tickers, start_date, end_date)
    if data is not None and not data.empty:
        data = calculate_daily_returns(data)
        data = calculate_cumulative_returns(data)
        stock_data = data

        # Calculate betas for the selected stocks
        betas = calculate_betas(stock_data)
        if betas:
            messagebox.showinfo("Success", "Data fetched successfully! Betas calculated.")
        else:
            messagebox.showwarning("Warning", "Betas could not be calculated.")
    else:
        stock_data = None
        messagebox.showerror("Error", "Failed to fetch stock data.")

# Show metrics button action
def show_metrics():
    if stock_data is None:
        messagebox.showwarning("No Data", "Please fetch stock data first.")
        return

    correlations = calculate_correlations(stock_data)
    plot_correlation_heatmap(correlations)

# Show visualizations button action
def show_visualizations():
    if stock_data is None:
        messagebox.showwarning("No Data", "Please fetch stock data first.")
        return

    plot_prices(stock_data)

# Show betas button action
def show_betas():
    if not betas:
        messagebox.showwarning("No Data", "Please fetch stock data first to calculate betas.")
        return

    beta_text = "\n".join([f"{ticker}: {beta:.2f}" for ticker, beta in betas.items()])
    messagebox.showinfo("Betas", f"Beta values for selected stocks:\n\n{beta_text}")

# Tkinter UI setup
root = tk.Tk()
root.title("Stock Performance Analysis")
root.geometry("500x450")

# UI Elements
ttk.Label(root, text="Enter Stock Tickers (comma-separated):").pack(pady=10)
ticker_entry = ttk.Entry(root, width=50)
ticker_entry.pack(pady=5)

ttk.Label(root, text="Start Date (YYYY-MM-DD):").pack(pady=10)
start_date_entry = ttk.Entry(root, width=20)
start_date_entry.pack(pady=5)

ttk.Label(root, text="End Date (YYYY-MM-DD):").pack(pady=10)
end_date_entry = ttk.Entry(root, width=20)
end_date_entry.pack(pady=5)

# Buttons
ttk.Button(root, text="Fetch Data", command=fetch_data).pack(pady=5)
ttk.Button(root, text="Show Metrics (Correlations)", command=show_metrics).pack(pady=5)
ttk.Button(root, text="Show Visualizations (Prices)", command=show_visualizations).pack(pady=5)
ttk.Button(root, text="Show Betas", command=show_betas).pack(pady=5)
ttk.Button(root, text="Show Today's Closing Prices", command=fetch_todays_closing_prices).pack(pady=5)

# Start the Tkinter event loop
root.mainloop()
