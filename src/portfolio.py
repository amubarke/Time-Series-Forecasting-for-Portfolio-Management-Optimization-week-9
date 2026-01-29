import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pypfopt import plotting
from pypfopt import EfficientFrontier, risk_models, expected_returns



def calculate_expected_returns(tsla_forecasted_prices, bnd_prices, spy_prices):
    # TSLA expected return from forecast
    tsla_return_forecast = (tsla_forecasted_prices[-1] - tsla_forecasted_prices[0]) / tsla_forecasted_prices[0]

    # Historical returns for BND and SPY
    bnd_returns = bnd_prices.pct_change().dropna()
    spy_returns = spy_prices.pct_change().dropna()
    
    bnd_expected_return = bnd_returns.mean() * 252  # annualize
    spy_expected_return = spy_returns.mean() * 252  # annualize

    # Combine into a vector
    expected_returns = pd.Series({
        'TSLA': tsla_return_forecast,
        'BND': bnd_expected_return,
        'SPY': spy_expected_return
    })

    return expected_returns

def compute_covariance_matrix(tsla_prices, bnd_prices, spy_prices):
    # Daily returns
    tsla_returns = tsla_prices.pct_change().dropna()
    bnd_returns = bnd_prices.pct_change().dropna()
    spy_returns = spy_prices.pct_change().dropna()
    
    returns_df = pd.concat([tsla_returns, bnd_returns, spy_returns], axis=1)
    returns_df.columns = ['TSLA', 'BND', 'SPY']

    # Covariance matrix (annualized)
    cov_matrix = returns_df.cov() * 252
    return cov_matrix


def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate=0.04):
    # Initialize Efficient Frontier
    ef = EfficientFrontier(expected_returns, cov_matrix)

    # Max Sharpe Ratio Portfolio
    max_sharpe_weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    max_sharpe_perf = ef.portfolio_performance(verbose=True)

    # Min Volatility Portfolio
    ef_minvol = EfficientFrontier(expected_returns, cov_matrix)
    min_vol_weights = ef_minvol.min_volatility()
    min_vol_perf = ef_minvol.portfolio_performance(verbose=True)

    return {
        "max_sharpe": {"weights": max_sharpe_weights, "performance": max_sharpe_perf},
        "min_vol": {"weights": min_vol_weights, "performance": min_vol_perf}
    }



def plot_efficient_frontier(expected_returns, cov_matrix, max_sharpe, min_vol):
    ef = EfficientFrontier(expected_returns, cov_matrix)
    fig, ax = plt.subplots(figsize=(10,6))
    
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
    
    # Mark max Sharpe
    ax.scatter(max_sharpe['performance'][1], max_sharpe['performance'][0], marker="*", color="r", s=200, label="Max Sharpe")
    
    # Mark min vol
    ax.scatter(min_vol['performance'][1], min_vol['performance'][0], marker="X", color="g", s=200, label="Min Volatility")
    
    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Expected Return")
    ax.legend()
    plt.show()


def plot_covariance_heatmap(cov_matrix):
    plt.figure(figsize=(6,5))
    sns.heatmap(cov_matrix, annot=True, fmt=".4f", cmap="coolwarm")
    plt.title("Covariance Matrix Heatmap")
    plt.show()
