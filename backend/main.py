from pathlib import Path
import pandas as pd
from fastapi import FastAPI

app = FastAPI(title="Retail Intelligence API")


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "retail_merged.csv"
FORECAST_PATH = BASE_DIR / "data" / "processed" / "sales_forecast.csv"


df = pd.read_csv(DATA_PATH, low_memory=False)
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["order_id"] = df["order_id"].astype(str)
df["invoice_no"] = df["invoice_no"].astype(str)
df["country"] = df["country"].astype(str)


@app.get("/")
def home():
    return {"message": "Retail Intelligence API is running"}


@app.get("/kpi")
def get_kpi():
    total_sales = float(df["sales"].sum())
    total_orders = int(df["order_id"].nunique())
    total_customers = int(df["customer_id"].nunique())

    return {
        "total_sales": total_sales,
        "total_orders": total_orders,
        "total_customers": total_customers,
    }


@app.get("/top-products")
def top_products(limit: int = 10):
    result = (
        df.dropna(subset=["description"])
        .groupby("description", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
        .head(limit)
    )
    return result.to_dict(orient="records")


@app.get("/sales-by-country")
def sales_by_country():
    result = (
        df.dropna(subset=["country"])
        .groupby("country", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
    )
    return result.to_dict(orient="records")


@app.get("/sales-trend")
def sales_trend():
    monthly = (
        df.dropna(subset=["order_date"])
        .assign(month=lambda x: x["order_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["sales"]
        .sum()
        .sort_values("month")
    )

    monthly["month"] = monthly["month"].astype(str)
    return monthly.to_dict(orient="records")

@app.get("/forecaast")
def get_forecast():
    forecast_df = pd.read_csv(FORECAST_PATH)
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"], errors="coerce")
    forecast_df["ds"] = forecast_df["ds"].astype(str)
    return forecast_df.to_dict(orient="records")