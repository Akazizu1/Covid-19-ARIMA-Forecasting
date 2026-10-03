import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. LOAD AND CLEAN THE DATA
# =====================================================
data = pd.read_csv("full_grouped.csv")
print("Original shape:", data.shape)

data["Date"] = pd.to_datetime(data["Date"])            # text -> date
data["Country/Region"] = data["Country/Region"].str.strip()
data = data.drop_duplicates()                           # remove duplicate rows
data = data.dropna()                                    # remove rows with missing values

# a few columns have small negative numbers (data corrections), set them to 0
number_cols = ["Confirmed", "Deaths", "Recovered", "Active",
               "New cases", "New deaths", "New recovered"]
for col in number_cols:
    data[col] = data[col].clip(lower=0)

data = data.sort_values(["Country/Region", "Date"]).reset_index(drop=True)
print("Cleaned shape:", data.shape)

# =====================================================
# 2. EXPLORE THE DATA
# =====================================================
print("\n--- First 5 rows ---")
print(data.head())
print("\n--- Summary statistics ---")
print(data[number_cols].describe().round(1))
print("\nDate range:", data["Date"].min().date(), "to", data["Date"].max().date())
print("Number of countries:", data["Country/Region"].nunique())
print("WHO regions:", list(data["WHO Region"].unique()))

# =====================================================
# 3. ANALYZE THE DATA
# =====================================================
column = input("\nWhich column do you want to analyze? (Confirmed / Deaths / Recovered / Active): ").strip().capitalize()
while column not in ["Confirmed", "Deaths", "Recovered", "Active"]:
    column = input("Invalid column, try again: ").strip().capitalize()

last_day = data["Date"].max()
latest = data[data["Date"] == last_day]                 # numbers on the last day

top10 = latest.sort_values(column, ascending=False).head(10)
print(f"\n--- Top 10 countries by {column} (on {last_day.date()}) ---")
print(top10[["Country/Region", column]].to_string(index=False))

region_total = latest.groupby("WHO Region")[column].sum().sort_values(ascending=False)
print(f"\n--- {column} by WHO region ---")
print(region_total)

# death rate = deaths per 100 confirmed cases
latest = latest[latest["Confirmed"] > 1000].copy()      # ignore countries with very few cases
latest["Death Rate"] = latest["Deaths"] / latest["Confirmed"] * 100
top_rate = latest.sort_values("Death Rate", ascending=False).head(10)

# =====================================================
# 4. VISUALIZE THE DATA
# =====================================================
# Chart 1: top 10 countries
plt.figure(figsize=(10, 5))
plt.bar(top10["Country/Region"], top10[column], color="steelblue")
plt.title(f"Top 10 countries by {column} cases")
plt.xlabel("Country")
plt.ylabel(column)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Chart 2: world trend over time
# combine all countries to get the worldwide total
world = data.groupby("Date")[column].sum()

plt.figure(figsize=(10, 5))
plt.plot(world, color="darkred")
plt.title(f"World {column} cases over time")
plt.xlabel("Date")
plt.ylabel(column)
plt.show()

# Chart 3: WHO regions
plt.figure(figsize=(10, 5))
plt.bar(region_total.index, region_total.values, color="seagreen")
plt.title(f"{column} cases by WHO region")
plt.xlabel("WHO Region")
plt.ylabel(column)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Chart 4: highest death rates
plt.figure(figsize=(10, 5))
plt.bar(top_rate["Country/Region"], top_rate["Death Rate"], color="orange")
plt.title("Top 10 death rates (countries with more than 1000 cases)")
plt.xlabel("Country")
plt.ylabel("Deaths per 100 confirmed cases")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Chart 5: country comparison (only if the user wants it)
answer = input("\nDo you want to compare countries? (yes/no): ").strip().lower()
if answer == "yes":
    names = input("Enter country names separated by commas (e.g. US, India, Brazil): ")
    plt.figure(figsize=(10, 5))
    for name in names.split(","):
        name = name.strip()
        country_data = data[data["Country/Region"] == name]
        if country_data.empty:
            print("Country not found, skipping:", name)
            continue
        plt.plot(country_data["Date"], country_data[column], label=name)
    plt.title(f"{column} cases comparison")
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.legend()
    plt.show()

# =====================================================
# 5. PREDICTIVE MODEL (ARIMA)
# =====================================================
print("\n--- ARIMA Forecasting ---")
country = input("Enter country name for forecasting (or 'World'): ").strip()
while country != "World" and country not in data["Country/Region"].values:
    country = input("Country not found, try again: ").strip()

if country == "World":
    # combine all countries for worldwide totals
    series = data.groupby("Date")[column].sum()
else:
    series = data[data["Country/Region"] == country].set_index("Date")[column]

series = series.asfreq("D")

# make the series stationary (ADF test + differencing)
d = 0
test_series = series.copy()
while adfuller(test_series.dropna())[1] > 0.05 and d < 2:
    test_series = test_series.diff()
    d += 1

print("Differencing order d =", d)

# try small p and q values and select the one with the lowest AIC
best_aic = np.inf
best_order = (1, d, 1)

for p in range(3):
    for q in range(3):
        try:
            aic = ARIMA(series, order=(p, d, q)).fit().aic
        except Exception:
            continue

        if aic < best_aic:
            best_aic = aic
            best_order = (p, d, q)

print("Best ARIMA order:", best_order)

# =====================================================
# 6. TRAIN AND TEST THE MODEL
# =====================================================

# split the data into training and testing
train_size = int(len(series) * 0.8)

train = series[:train_size]
test = series[train_size:]

# train the model using training data only
model = ARIMA(train, order=best_order).fit()

# predict the test period
predictions = model.forecast(steps=len(test))

# calculate model evaluation metrics
actual = test
fitted = predictions

# remove zero values because MAPE cannot be calculated for zero
valid = actual > 0
actual = actual[valid]
fitted = fitted[valid]

mae = np.mean(np.abs(actual - fitted))
rmse = np.sqrt(np.mean((actual - fitted) ** 2))
mape = np.mean(np.abs((actual - fitted) / actual)) * 100

print("\n--- Model Evaluation ---")
print("MAE     :", round(mae, 2))
print("RMSE    :", round(rmse, 2))
print("MAPE    :", round(mape, 2), "%")

# plot actual vs predicted test data
plt.figure(figsize=(10, 5))
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, predictions, label="Predicted", color="red", linestyle="--")
plt.title(f"Actual vs predicted - {country}")
plt.xlabel("Date")
plt.ylabel(column)
plt.legend()
plt.show()

# =====================================================
# 7. FINAL MODEL AND 30-DAY FORECAST
# =====================================================

# retrain the model using all available data
final_model = ARIMA(series, order=best_order).fit()

# forecast the next 30 days
forecast = final_model.get_forecast(steps=30)
mean = forecast.predicted_mean
ci = forecast.conf_int()

plt.figure(figsize=(10, 5))
plt.plot(series, label="Historical")
plt.plot(mean, label="30-day forecast", color="red")
plt.fill_between(
    ci.index,
    ci.iloc[:, 0],
    ci.iloc[:, 1],
    color="red",
    alpha=0.2
)

plt.title(f"30-day forecast of {column} cases - {country}")
plt.xlabel("Date")
plt.ylabel(column)
plt.legend()
plt.show()

print("\nForecasted values:")
print(mean.round(0).astype(int))