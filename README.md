# Sales-Dashboard-of-Hotel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Step 1: Simulating Financial Dataset
np.random.seed(42)
date_range = pd.date_range(start="2020-01-01", end="2023-12-31", freq='D')
data = {
    "date": date_range,
    "revenue": np.random.uniform(5000, 15000, len(date_range)).round(2),
    "expenses": np.random.uniform(2000, 7000, len(date_range)).round(2),
    "stock_price": np.random.uniform(100, 300, len(date_range)).round(2),
}

df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])
df["profit"] = df["revenue"] - df["expenses"]
df["profit_margin"] = (df["profit"] / df["revenue"] * 100).round(2)
df["year_month"] = df["date"].dt.to_period("M")

# Step 2: Exploratory Data Analysis
sns.set(style="whitegrid", palette="muted")
plt.figure(figsize=(15, 10))

# Revenue and Expenses over Time
plt.subplot(2, 1, 1)
sns.lineplot(data=df, x="date", y="revenue", label="Revenue", color="blue")
sns.lineplot(data=df, x="date", y="expenses", label="Expenses", color="orange")
plt.title("Revenue and Expenses Over Time", fontsize=16)
plt.ylabel("Amount (₹)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend()

# Profit Margin Distribution
plt.subplot(2, 2, 3)
sns.histplot(df["profit_margin"], kde=True, color="green")
plt.title("Profit Margin Distribution", fontsize=16)
plt.xlabel("Profit Margin (%)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

# Correlation Heatmap
correlation_matrix = df[["revenue", "expenses", "profit", "profit_margin", "stock_price"]].corr()
plt.subplot(2, 2, 4)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={'shrink': 0.8})
plt.title("Correlation Matrix", fontsize=16)

plt.tight_layout()
plt.show()

# Step 3: Time-Series Forecasting
monthly_revenue = df.groupby("year_month")["revenue"].sum()
monthly_revenue.index = monthly_revenue.index.to_timestamp()

# Step 4: Stationarity Check using ADF Test
result = adfuller(monthly_revenue)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# If p-value > 0.05, the series is non-stationary and needs differencing
if result[1] > 0.05:
    print("The series is non-stationary. Differencing the data.")
    monthly_revenue = monthly_revenue.diff().dropna()

# Step 5: Plotting ACF and PACF to identify AR and MA orders
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(monthly_revenue, lags=12, ax=plt.gca())  # Adjusted to 12 lags
plt.title("ACF (Auto-Correlation Function)")

plt.subplot(1, 2, 2)
plot_pacf(monthly_revenue, lags=12, ax=plt.gca())  # Adjusted to 12 lags
plt.title("PACF (Partial Auto-Correlation Function)")

plt.tight_layout()
plt.show()

# Step 6: ARIMA Model - Forecasting next 12 months
try:
    # Fit the ARIMA model (adjusting p, d, q based on ACF/PACF)
    model = ARIMA(monthly_revenue, order=(1, 1, 1))  # p=1, d=1, q=1 based on earlier ACF/PACF plots
    model_fit = model.fit()

    # Forecasting
    forecast_steps = 12
    forecast = model_fit.forecast(steps=forecast_steps)

    # Creating a future time index
    future_dates = pd.date_range(start=monthly_revenue.index[-1] + pd.DateOffset(months=1),
                                 periods=forecast_steps, freq='MS')

    # Combine the forecast with the future dates
    forecast_df = pd.DataFrame({"Forecasted Revenue": forecast.round(2)}, index=future_dates)

    # Plotting historical and forecasted data
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_revenue, label="Historical Revenue", color="blue")
    plt.plot(forecast_df, label="Forecasted Revenue", color="orange", linestyle="--")
    plt.title("Revenue Forecast (Next 12 Months)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Revenue (₹)", fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

    # Displaying Forecast Data
    print(forecast_df.head())
except Exception as e:
    print(f"Error in ARIMA model fitting or forecasting: {e}")
