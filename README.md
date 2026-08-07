[DOI](https://zenodo.org/badge/1326598410.svg)](https://doi.org/10.5281/zenodo.21838225)

This repository contains the implementation for the paper: **"Dynamic Mean–Variance Portfolio Selection under Multifactor Stochastic Volatility: A Computational Framework for Stochastic Riccati Equations"**.

**Authors:** Zhecheng Huang, Guojiang Shao, Lei Wang, Qi Zhang

This codebase provides deep learning-based solvers (DeepBSDE and DBDP2) for Stochastic Riccati Equations (SREs) in continuous-time portfolio selection problems under the Heston stochastic volatility model. The implementation is calibrated to empirical market parameters (2015-2019) for the S&P 500 and four major sector ETFs (XLK, XLF, XLE, XLV).

## 📂 Repository Structure

The repository is modularized into three main components: the DeepBSDE solver, the DBDP2 solver, and the out-of-sample backtesting engine.

### 1. DeepBSDE Method (Root Directory)

The root directory contains the implementation of the DeepBSDE approach, alongside the core market dynamics models shared across both methods.

* `StochasticVolatilityModel.py`: Constructs the Heston model for the four sector ETFs (XLK, XLF, XLE, XLV) and the market index.
* `lnSRE_Equation.py`: Defines the backward equation for the Stochastic Riccati Equation after the log transformation.
* `LambdaNetwork.py`: Neural network architecture specifically designed for the DeepBSDE methodology.
* `DeepLnSRESolver.py`: The learning solver executing the DeepBSDE algorithm.
* `main.py`: The primary execution script. It includes the empirical market parameters from 2015-2019 and serves as an example for hyperparameter configuration and training execution.

### 2. DBDP2 Method (`/DBDP2` Directory)

The DBDP2 folder contains the implementation for the DBDP2 solver. Note that this method directly imports and reuses `StochasticVolatilityModel.py` and `lnSRE_Equation.py` from the root directory to ensure rigorous methodological comparison.

* `DBDP2/ValueNetwork.py`: Neural network architecture for the value equation of the lnSRE under the DBDP2 approach.
* `DBDP2/DBDPLnSRESolver.py`: The learning solver executing the DBDP2 algorithm.

### 3. Real Market Backtesting (`/Trading_test` Directory)

To evaluate the practical asset allocation performance of the trained neural network models under **real-world market dynamics**, this repository includes a backtesting engine driven by actual historical market data, alongside pre-trained model weights.

* **Backtest Engine & Strategies**: Includes the core evaluation framework (`DeepBSDEBacktestEngine_lnSRE.py`) and log-space strategy implementations for both the DeepBSDE and DBDP2 methods.
* **Historical Data & Results**: Contains the real historical dataset (`SP500_ETFs_data.csv`), pre-trained neural network weights (`.pt` files), and Jupyter Notebooks demonstrating the practical execution of both methods using the actual market trajectory for the year 2020.
* **Detailed Documentation**: For a comprehensive breakdown of the engine architecture, supported empirical benchmarks (e.g., Equal Weight, Inverse Variance, Global Minimum Variance), and detailed performance/risk metrics, please refer to the dedicated documentation: **[Trading_test/README_for_trading_test.md](./Trading_test/README_for_trading_test.md)**.
---

## 💻 Computational Cost & Hardware Requirements

All benchmarks and experimental results in the paper were executed using Google Colab with an NVIDIA L4 GPU. Expected convergence times are as follows:

* **DeepBSDE Method**: ~ 1 to 2 hours (depending on exact model parameters).
* **DBDP2 Method**: > 3 hours (varies based on early stopping precision and maximum epoch configurations).

---

## 🤝 Acknowledgments

The authors would like to thank Weiran Xiong for helpful discussions on the use of deep learning methods forsolving backward stochastic differential equations (BSDEs).
