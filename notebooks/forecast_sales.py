from pathlib import Path
import pandas as pd
from prophet import Prophet

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "retail_merged.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "sales_forecast.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

monthly = (
    df.dropna(subset=["order_date"])
      .assign(month=lambda x: x["order_date"].dt.to_period("M").dt.to_timestamp())
      .groupby("month", as_index=False)["sales"]
      .sum()
      .sort_values("month")
)

monthly = monthly.rename(columns={"month": "ds", "sales": "y"})

model = Prophet()
model.fit(monthly)

future = model.make_future_dataframe(periods=6, freq="ME")
forecast = model.predict(future)

forecast_output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
forecast_output.to_csv(OUTPUT_PATH, index=False)

print("Forecast complete.")
print(forecast_output.tail(10))