# BacktestEngine

This repository contains `BacktestEngine`, a Python-based backtesting framework designed for evaluating asset allocation strategies. The engine handles rolling-window data ingestion, daily mark-to-market valuations, transaction logging, and comprehensive risk metric calculations.

## Code Architecture

The codebase is structured around three core components:

* **`MarketSnapshot`**: A frozen dataclass that captures the state of the market at a specific rebalancing timestamp. It provides external strategy objects with historical price windows, lagged volatility metrics, and portfolio equity states required for dynamic weight calculations.
* **`BenchmarkCalculator`**: A utility class containing static methods to compute allocation weights for standard empirical baselines based on a rolling historical data window.
* **`BacktestEngine`**: The central execution environment. It simulates the daily evolution of portfolio wealth, evaluates external strategies against built-in benchmarks, executes rebalancing logic at discrete intervals, and logs all transaction details for post-trade analysis.

---

## Supported Benchmark Methods

The engine natively evaluates any custom strategy against three widely recognized empirical baselines, computed internally by the `BenchmarkCalculator`:

* **Equal Weight (EW)**
A naive diversification approach allocating capital equally across all available assets regardless of historical market dynamics. The weight for each asset is defined as $w_i = 1/N$, where $N$ is the total number of risky assets.
* **Inverse Variance (IV)**
A simplified risk-parity mechanism that allocates capital inversely proportional to the historical variance of each asset, inherently assuming zero cross-asset correlation. Weights are computed over a 20-day rolling window:

$$w_i = \frac{\sigma_i^{-2}}{\sum_{j=1}^N \sigma_j^{-2}}$$


* **Global Minimum Variance (GMV)**
A risk-centric portfolio utilizing the full historical covariance matrix $\Sigma$ over a 20-day rolling window. It captures historical cross-asset correlations while remaining agnostic to expected returns:

$$w = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top \Sigma^{-1}\mathbf{1}}$$



*Note: If `allow_leverage=False` is passed to the engine, the GMV weights are normalized by their absolute sum to enforce cash constraints.*

---

## Performance and Risk Metrics

The engine automatically computes a comprehensive suite of metrics based on the daily portfolio net asset value (NAV) trajectories ($x(t)$). Assuming 252 trading days per year and letting $r_t = \ln(x(t)/x(t-\Delta t))$ represent the daily logarithmic return:

* **Annualized Return:** The continuous compound growth rate throughout the investment horizon.
* **Annualized Volatility:** The overall statistical dispersion of portfolio returns.
* **Sharpe Ratio:** Evaluates risk-adjusted performance by measuring the absolute return generated per unit of aggregate risk ($\text{Return} / \text{Vol}$).
* **Sortino Ratio:** A refined variation of the Sharpe metric that penalizes only downside volatility, measuring the genuine risk of capital erosion.
* **Maximum Drawdown (MDD):** The most severe peak-to-trough percentage decline in the portfolio's historical wealth.
* **Recovery Time (RT):** The maximum number of consecutive trading sessions the portfolio remains "underwater" before reclaiming its previous high-water mark.
* **Calmar Ratio:** Contrasts the overall return against the maximum drawdown ($\text{Return} / \vert{}\text{MDD}\vert{}$).
* **Marginal Expected Shortfall at 5% (MES 5%):** A measure of systemic tail-risk exposure. It calculates the expected average return of a given strategy exclusively across the worst 5% of trading days experienced by the aggregate market baseline (using the Equal Weight portfolio as the market proxy).

---
## Directory Contents

The `/Trading_test` directory includes the following files to facilitate the empirical evaluation:

### Core Engine & Strategies
* **`DeepBSDEBacktestEngine_lnSRE.py`**: The core framework containing the `BacktestEngine`, `MarketSnapshot`, and `BenchmarkCalculator` classes described above.
* **`DeepBSDE_Strategy_LogSpace.py`**: The concrete strategy implementation that loads the trained DeepBSDE model and interfaces with the backtest engine to output dynamic optimal weights based on real-time market snapshots.
* **`DBDP_Strategy_LogSpace.py`**: The corresponding strategy implementation that wraps the trained DBDP2 model for backtesting evaluation.

### Data & Pre-trained Weights
* **`SP500_ETFs_data.csv`**: The empirical historical market dataset used for 4 ETFs with S&P500 index.
* **`sre_deepbsde_model_step5.pt`**: The finalized, pre-trained PyTorch neural network weights for the DeepBSDE solver.
* **`value_net_final_complete_set2.pt`**: The finalized, pre-trained PyTorch neural network weights for the DBDP2 solver.

### Execution & Verification
* **`Trading_test_DeepBSDE_2020.ipynb`**: A Jupyter Notebook demonstrating the DeepBSDE strategy for the year 2020.
* **`Trading_test_DBDP2_2020.ipynb`**: A parallel Jupyter Notebook dedicated to the DBDP2 strategy for the year 2020.
