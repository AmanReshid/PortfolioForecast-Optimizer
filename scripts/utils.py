import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")

def load_data(tickers, start_date="2015-01-01", end_date="2024-01-01"):
    """
    Load historical data from YFinance for given tickers.
    """
    try:
        logging.info(f"Loading data for tickers: {tickers}")
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return pd.DataFrame()

def clean_data(data):
    """
    Handle missing values by forward filling.
    """
    missing_values_before = data.isnull().sum().sum()
    logging.info(f"Missing values before cleaning: {missing_values_before}")
    data.ffill(inplace=True)
    missing_values_after = data.isnull().sum().sum()
    logging.info(f"Missing values after cleaning: {missing_values_after}")
    return data

def normalize_data(data):
    """
    Normalize data for comparison by setting all assets to 100% at the start.
    """
    logging.info("Normalizing data for comparison.")
    return data / data.iloc[0] * 100

def calculate_daily_returns(data):
    """
    Calculate daily returns for each asset.
    """
    logging.info("Calculating daily returns.")
    return data.pct_change().dropna()

def plot_normalized_data(data):
    """
    Plot normalized price data for visualization.
    """
    logging.info("Plotting normalized price data.")
    data.plot(figsize=(12, 6), title="Normalized Price of Assets")
    plt.ylabel("Normalized Price (%)")
    plt.show()

def plot_daily_returns(daily_returns):
    """
    Plot daily returns to show volatility.
    """
    logging.info("Plotting daily returns.")
    daily_returns.plot(figsize=(12, 6), title="Daily Returns of Assets")
    plt.ylabel("Daily Return (%)")
    plt.show()

def rolling_statistics(daily_returns, window=30):
    """
    Calculate rolling mean and standard deviation for volatility analysis.
    """
    logging.info(f"Calculating rolling mean and standard deviation with window size {window}.")
    rolling_mean = daily_returns.rolling(window=window).mean()
    rolling_std = daily_returns.rolling(window=window).std()
    return rolling_mean, rolling_std

def plot_rolling_stats(rolling_mean, rolling_std):
    """
    Plot rolling mean and standard deviation for volatility analysis.
    """
    logging.info("Plotting rolling mean and standard deviation.")
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    rolling_mean.plot(ax=ax[0], title="30-Day Rolling Mean of Daily Returns")
    rolling_std.plot(ax=ax[1], title="30-Day Rolling Standard Deviation (Volatility)")
    ax[0].set_ylabel("Mean Return (%)")
    ax[1].set_ylabel("Volatility (%)")
    plt.show()

def detect_outliers(daily_returns, threshold=3):
    """
    Detect outliers using a z-score threshold.
    """
    logging.info(f"Detecting outliers using a threshold of {threshold}.")
    z_scores = np.abs((daily_returns - daily_returns.mean()) / daily_returns.std())
    outliers = daily_returns[(z_scores > threshold).any(axis=1)]
    logging.info(f"Number of outliers detected: {len(outliers)}")
    return outliers

def plot_outliers(daily_returns, outliers):
    """
    Plot daily returns with identified outliers.

    Parameters:
        daily_returns (pd.DataFrame): DataFrame containing daily returns of assets.
        outliers (pd.DataFrame): DataFrame containing only the identified outliers.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    # Plot all daily returns
    for column in daily_returns.columns:
        plt.plot(daily_returns.index, daily_returns[column], label=f"{column} Daily Returns")

        # Plot outliers for the specific column
        if column in outliers.columns:
            plt.scatter(
                outliers.index, 
                outliers[column], 
                color='red', 
                label=f"{column} Outliers", 
                marker='o'
            )

    plt.title("Daily Returns with Outliers")
    plt.xlabel("Date")
    plt.ylabel("Daily Return (%)")
    plt.legend()
    plt.show()


def decompose_time_series(data, ticker, period=365):
    """
    Decompose time series data into trend, seasonal, and residual components.
    """
    logging.info(f"Decomposing time series for ticker: {ticker}")
    try:
        decomposition = seasonal_decompose(data[ticker].dropna(), model='multiplicative', period=period)
        return decomposition
    except Exception as e:
        logging.error(f"Error in decomposition: {e}")
        return None

def plot_decomposition(decomposition):
    """
    Plot seasonal decomposition of time series data.
    """
    if decomposition is not None:
        logging.info("Plotting seasonal decomposition.")
        decomposition.plot()
        plt.suptitle("Seasonal Decomposition")
        plt.show()
    else:
        logging.error("Decomposition data is None. Skipping plot.")

def calculate_var_sharpe_ratio(daily_returns, ticker, confidence_level=0.05):
    """
    Calculate Value at Risk (VaR) and Sharpe Ratio for a given asset.
    """
    logging.info(f"Calculating VaR and Sharpe Ratio for {ticker} with {confidence_level*100}% confidence.")
    VaR = np.percentile(daily_returns[ticker], confidence_level * 100)
    sharpe_ratio = (daily_returns[ticker].mean() / daily_returns[ticker].std()) * np.sqrt(252)  # Annualized
    logging.info(f"VaR: {VaR:.4f}, Sharpe Ratio: {sharpe_ratio:.2f}")
    return VaR, sharpe_ratio
