# Demand Forecasting & Inventory Optimization Tool

An interactive analytics tool that forecasts product demand using 5 statistical and ML methods, calculates optimal inventory parameters, and presents insights through a Streamlit dashboard — built on 1.7M+ real retail sales records.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Live Demo

[View Dashboard →](https://demand-forecast-9hjy8sujmqrpxzcxnlgacb.streamlit.app/)

---

## Problem

Supply chain teams need to answer two critical questions daily: how much demand should we expect, and how much inventory should we carry? Getting these wrong means either stockouts (lost sales, unhappy customers) or overstocking (wasted capital, holding costs). This tool automates both answers using data.

## Solution

An end-to-end analytics pipeline that ingests historical sales data, runs multiple forecasting models, evaluates accuracy, and calculates optimal inventory parameters — all accessible through an interactive web dashboard with no coding required by the end user.

---

## Dataset

**Kaggle: Store Sales — Time Series Forecasting**
- 3M+ sales records across 54 stores and 33 product families
- 1,782 days of history (2013–2017)
- Includes promotions, oil prices, holidays, store metadata
- Source: Real retail data from Corporación Favorita (Ecuador)

---

## Features

### Data Pipeline
Loads and merges 5 CSV files (sales, stores, oil prices, holidays, transactions). Handles missing values with forward-fill interpolation. Engineers 10+ features including day-of-week, payday flags, weekend indicators, seasonality markers, and promotion flags.

### Exploratory Analysis
Interactive charts filtered by product family and time aggregation (daily/weekly/monthly). Analyzes sales trends, day-of-week patterns, monthly seasonality, and promotion impact with quantified lift percentages.

### Demand Forecasting (5 Methods)
Compares these models side by side on an 80/20 train/test split:

| Method | Type | What It Captures |
|--------|------|-----------------|
| Moving Average | Statistical | Short-term trends |
| Holt-Winters | Exponential Smoothing | Trend + seasonality |
| ARIMA | Statistical | Autoregressive patterns |
| Prophet | Facebook/Meta | Holidays + changepoints |
| XGBoost | Machine Learning | Non-linear relationships with lag features |

Each model is evaluated on MAE, RMSE, and MAPE. The best model is highlighted automatically. Forecast vs actuals plotted for visual comparison.

### Inventory Optimization
Calculates for any product with adjustable parameters:
- **EOQ** — Economic Order Quantity (minimizes total ordering + holding cost)
- **Safety Stock** — Buffer for demand variability at target service level
- **Reorder Point** — When to trigger the next purchase order
- **Total Cost Curve** — Visual showing cost trade-offs across order quantities
- **Inventory Turnover** — How efficiently stock is being cycled

### ABC Analysis
Pareto classification of all 33 product families:
- **A items** — Top 80% of revenue (vital few, tight control)
- **B items** — Next 15% of revenue (moderate control)
- **C items** — Bottom 5% of revenue (loose control)

Includes Pareto chart with cumulative percentage overlay.

### What-If Simulator
Compare inventory scenarios across multiple lead times and service levels simultaneously. Visualizes how safety stock and total cost change as you adjust parameters. Answers questions like "What happens to our costs if lead time doubles?" or "How much more stock do we need for 99% vs 95% fill rate?"

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11+ | Core language |
| pandas, NumPy | Data manipulation |
| plotly | Interactive visualization |
| statsmodels | ARIMA, Exponential Smoothing |
| Prophet | Holiday-aware forecasting |
| XGBoost, scikit-learn | ML forecasting + evaluation |
| SciPy | Statistical calculations (z-scores) |
| Streamlit | Interactive web dashboard |
| Streamlit Cloud | Free hosting |

---

## Project Structure

```
demand-forecast/
├── data/
│   ├── train.csv
│   ├── stores.csv
│   ├── oil.csv
│   ├── holidays_events.csv
│   └── transactions.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Data loading, cleaning, feature engineering, ABC analysis
│   ├── forecasting.py       # 5 forecasting models + evaluation metrics
│   └── inventory.py         # EOQ, safety stock, ROP, what-if analysis, cost curves
├── app.py                   # Streamlit dashboard (6 pages)
├── requirements.txt
└── README.md
```

---

## Key Metrics & Results

- **Dataset:** 3M+ records, 54 stores, 33 product families, 4.5 years
- **Best Forecast Accuracy:** 8–12% MAPE depending on product family
- **Forecasting Methods:** 5 models compared with automated best-model selection
- **Inventory Parameters:** EOQ, safety stock, and ROP calculated at configurable service levels (90/95/99%)
- **ABC Classification:** 33 families classified with Pareto analysis

---

## Getting Started

### Prerequisites
- Python 3.11+
- Dataset from [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

### Installation

```bash
git clone https://github.com/Vanshsingh1203/demand-forecast.git
cd demand-forecast
pip install -r requirements.txt
```

### Download Data
Download the dataset from Kaggle and place all CSV files in the `data/` folder.

### Run Locally

```bash
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`

---

## Supply Chain Concepts Applied

- **Time Series Decomposition** — Separating trend, seasonality, and residuals
- **Demand Variability** — Standard deviation of demand driving safety stock
- **Economic Order Quantity** — Balancing ordering cost vs holding cost
- **Service Level Targeting** — Translating fill rate goals into inventory buffers
- **ABC/Pareto Analysis** — Focusing control effort on high-value items
- **Lead Time Impact** — Quantifying how supplier responsiveness affects stock needs
- **Total Cost of Inventory** — Holistic view beyond just unit price

---

## Interview Questions This Project Prepares You For

- How would you forecast demand for a seasonal product?
- What's the difference between ARIMA and exponential smoothing?
- How do you calculate safety stock and what drives it?
- What factors go into setting a reorder point?
- How would you choose between a statistical model and ML for forecasting?
- What is ABC analysis and how does it inform inventory policy?
- How do you balance service level against holding cost?
- Walk me through an EOQ calculation and its assumptions.

---

## Author

**Vansh Singh**
- MS in Engineering Management — Northeastern University
- BE in Mechanical Engineering — Vellore Institute of Technology

## License

MIT — free to use, modify, and distribute.