"""
main.py  —  Upgraded Retail Intelligence API
=============================================
New endpoints over v1:
  • /kpi              — enhanced with MoM delta & profit metrics
  • /top-products     — unchanged
  • /sales-by-country — unchanged
  • /sales-trend      — now includes profit trend
  • /forecast         — now includes type (historical/forecast) + trend
  • /rfm              — RFM segments with customer counts & monetary value
  • /rfm/detail       — full per-customer RFM table (paginated)
  • /profit-analysis  — profit & margin by category / sub-category
  • /returns          — return rates by product & country
  • /cohort           — monthly cohort retention matrix
  • /data-quality     — ETL data quality report
  • /forecast-metrics — Prophet cross-validation metrics
"""

from pathlib import Path
from functools import lru_cache
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Retail Intelligence API",
    description="Professional retail analytics API powering the BI dashboard.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def _path(name: str) -> Path:
    return PROCESSED_DIR / name


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS  (cached so CSVs are read once per process)
# ══════════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=1)
def _load_merged() -> pd.DataFrame:
    df = pd.read_csv(_path("retail_merged.csv"), low_memory=False)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_id"]   = df["order_id"].astype(str)
    df["invoice_no"] = df["invoice_no"].astype(str)
    df["country"]    = df["country"].astype(str)
    if "is_return" not in df.columns:
        df["is_return"] = False
    if "cohort_month" in df.columns:
        df["cohort_month"] = pd.to_datetime(df["cohort_month"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def _load_rfm() -> pd.DataFrame:
    return pd.read_csv(_path("rfm_segments.csv"))


@lru_cache(maxsize=1)
def _load_forecast() -> pd.DataFrame:
    df = pd.read_csv(_path("sales_forecast.csv"))
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def _load_returns() -> pd.DataFrame:
    p = _path("product_return_rates.csv")
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@lru_cache(maxsize=1)
def _load_dq() -> pd.DataFrame:
    p = _path("data_quality_report.csv")
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@lru_cache(maxsize=1)
def _load_forecast_metrics() -> pd.DataFrame:
    p = _path("forecast_metrics.csv")
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def _sales_df() -> pd.DataFrame:
    """Merged data with returns excluded."""
    df = _load_merged()
    return df[df["is_return"] == False].copy()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _safe_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-safe records (handles NaN, NaT, Timestamps)."""
    return (
        df.replace({np.nan: None})
        .assign(**{
            c: df[c].astype(str)
            for c in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        })
        .to_dict(orient="records")
    )


def _mom_delta(series: pd.Series) -> Optional[float]:
    """Month-over-month % change from last two values."""
    vals = series.dropna().values
    if len(vals) < 2 or vals[-2] == 0:
        return None
    return round((vals[-1] - vals[-2]) / vals[-2] * 100, 2)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def home():
    return {
        "message": "Retail Intelligence API v2 is running",
        "endpoints": [
            "/kpi", "/top-products", "/sales-by-country", "/sales-trend",
            "/forecast", "/forecast-metrics",
            "/rfm", "/rfm/detail",
            "/profit-analysis", "/returns", "/cohort",
            "/data-quality",
        ],
    }


# ── KPI ───────────────────────────────────────────────────────────────────────
@app.get("/kpi", tags=["Overview"])
def get_kpi():
    df = _sales_df()

    total_sales     = float(df["sales"].sum())
    total_orders    = int(df["order_id"].nunique())
    total_customers = int(df["customer_id"].nunique())
    total_profit    = float(df["profit"].sum()) if "profit" in df.columns else None
    avg_order_value = round(total_sales / total_orders, 2) if total_orders else 0

    # MoM deltas from monthly trend
    monthly = (
        df.dropna(subset=["order_date"])
        .assign(month=lambda x: x["order_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")["sales"]
        .sum()
        .sort_index()
    )
    sales_mom = _mom_delta(monthly)

    profit_margin_pct = None
    if total_profit is not None and total_sales > 0:
        profit_margin_pct = round(total_profit / total_sales * 100, 2)

    return {
        "total_sales":        round(total_sales, 2),
        "total_orders":       total_orders,
        "total_customers":    total_customers,
        "total_profit":       round(total_profit, 2) if total_profit is not None else None,
        "avg_order_value":    avg_order_value,
        "profit_margin_pct":  profit_margin_pct,
        "sales_mom_delta_pct": sales_mom,
    }


# ── TOP PRODUCTS ──────────────────────────────────────────────────────────────
@app.get("/top-products", tags=["Products"])
def top_products(limit: int = Query(10, ge=1, le=100)):
    df = _sales_df()
    result = (
        df.dropna(subset=["description"])
        .groupby("description", as_index=False)
        .agg(sales=("sales", "sum"), orders=("order_id", "count"))
        .sort_values("sales", ascending=False)
        .head(limit)
    )
    result["sales"] = result["sales"].round(2)
    return _safe_records(result)


# ── SALES BY COUNTRY ──────────────────────────────────────────────────────────
@app.get("/sales-by-country", tags=["Geography"])
def sales_by_country(limit: int = Query(50, ge=1, le=200)):
    df = _sales_df()
    result = (
        df.dropna(subset=["country"])
        .groupby("country", as_index=False)
        .agg(sales=("sales", "sum"), orders=("order_id", "count"))
        .sort_values("sales", ascending=False)
        .head(limit)
    )
    result["sales"] = result["sales"].round(2)
    return _safe_records(result)


# ── SALES TREND ───────────────────────────────────────────────────────────────
@app.get("/sales-trend", tags=["Overview"])
def sales_trend():
    df = _sales_df()
    monthly = (
        df.dropna(subset=["order_date"])
        .assign(month=lambda x: x["order_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(sales=("sales", "sum"), profit=("profit", "sum"), orders=("order_id", "nunique"))
        .sort_values("month")
    )
    monthly["month"]  = monthly["month"].astype(str)
    monthly["sales"]  = monthly["sales"].round(2)
    monthly["profit"] = monthly["profit"].round(2)
    return _safe_records(monthly)


# ── FORECAST ──────────────────────────────────────────────────────────────────
@app.get("/forecast", tags=["Forecast"])
def get_forecast(type: Optional[str] = Query(None, description="Filter: 'historical' or 'forecast'")):
    df = _load_forecast().copy()
    if type in ("historical", "forecast") and "type" in df.columns:
        df = df[df["type"] == type]
    df["ds"] = df["ds"].astype(str)
    return _safe_records(df)


@app.get("/forecast-metrics", tags=["Forecast"])
def get_forecast_metrics():
    df = _load_forecast_metrics()
    if df.empty:
        raise HTTPException(status_code=404, detail="forecast_metrics.csv not found. Run forecast.py first.")
    df["horizon"] = df["horizon"].astype(str)
    return _safe_records(df)


# ── RFM ───────────────────────────────────────────────────────────────────────
@app.get("/rfm", tags=["Customers"])
def get_rfm_summary():
    """Aggregated RFM segment counts and monetary totals."""
    df = _load_rfm()
    summary = (
        df.groupby("segment", as_index=False)
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            total_monetary=("monetary", "sum"),
            avg_monetary=("monetary", "mean"),
        )
        .sort_values("total_monetary", ascending=False)
    )
    for col in ["avg_recency", "avg_frequency", "total_monetary", "avg_monetary"]:
        summary[col] = summary[col].round(2)
    return _safe_records(summary)


@app.get("/rfm/detail", tags=["Customers"])
def get_rfm_detail(
    segment: Optional[str] = Query(None, description="Filter by segment name"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Paginated per-customer RFM table."""
    df = _load_rfm()
    if segment:
        df = df[df["segment"].str.lower() == segment.lower()]
    df = df.sort_values("monetary", ascending=False).iloc[offset: offset + limit]
    return {
        "total": int(_load_rfm().shape[0] if not segment else len(df)),
        "offset": offset,
        "limit": limit,
        "data": _safe_records(df),
    }


# ── PROFIT ANALYSIS ───────────────────────────────────────────────────────────
@app.get("/profit-analysis", tags=["Profit"])
def profit_analysis(group_by: str = Query("category", description="'category' or 'sub_category'")):
    df = _sales_df()
    if group_by not in ("category", "sub_category"):
        raise HTTPException(status_code=400, detail="group_by must be 'category' or 'sub_category'")

    df = df.dropna(subset=[group_by, "sales", "profit"])
    result = (
        df.groupby(group_by, as_index=False)
        .agg(
            sales=("sales", "sum"),
            profit=("profit", "sum"),
            orders=("order_id", "count"),
        )
        .sort_values("profit", ascending=False)
    )
    result["profit_margin_pct"] = (result["profit"] / result["sales"] * 100).round(2)
    result["sales"]             = result["sales"].round(2)
    result["profit"]            = result["profit"].round(2)
    return _safe_records(result)


# ── RETURNS ───────────────────────────────────────────────────────────────────
@app.get("/returns", tags=["Returns"])
def get_returns(limit: int = Query(20, ge=1, le=200)):
    df = _load_returns()
    if df.empty:
        raise HTTPException(status_code=404, detail="product_return_rates.csv not found. Run etl_retail.py first.")
    return _safe_records(df.head(limit))


@app.get("/returns/by-country", tags=["Returns"])
def returns_by_country():
    """Return volume and value by country."""
    merged = _load_merged()
    returns = merged[merged["is_return"] == True].copy()
    if returns.empty:
        return []
    result = (
        returns.dropna(subset=["country"])
        .groupby("country", as_index=False)
        .agg(
            return_orders=("order_id", "count"),
            return_value=("sales", "sum"),
        )
        .sort_values("return_orders", ascending=False)
    )
    result["return_value"] = result["return_value"].abs().round(2)
    return _safe_records(result)


# ── COHORT RETENTION ─────────────────────────────────────────────────────────
@app.get("/cohort", tags=["Customers"])
def cohort_retention():
    """
    Monthly cohort retention matrix.
    Returns list of {cohort_month, period_number, customers, retention_pct}.
    """
    df = _sales_df()

    if "cohort_month" not in df.columns:
        raise HTTPException(
            status_code=404,
            detail="cohort_month column missing. Re-run etl_retail.py first."
        )

    df = df.dropna(subset=["customer_id", "order_date", "cohort_month"])
    df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()

    cohort_data = (
        df.groupby(["cohort_month", "order_month"])["customer_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_id": "customers"})
    )

    cohort_data["period_number"] = (
        (cohort_data["order_month"].dt.to_period("M") -
         cohort_data["cohort_month"].dt.to_period("M"))
        .apply(lambda x: x.n)
    )

    # Cohort sizes (period 0)
    cohort_sizes = (
        cohort_data[cohort_data["period_number"] == 0]
        .set_index("cohort_month")["customers"]
        .rename("cohort_size")
    )

    cohort_data = cohort_data.join(cohort_sizes, on="cohort_month")
    cohort_data["retention_pct"] = (
        cohort_data["customers"] / cohort_data["cohort_size"] * 100
    ).round(2)

    cohort_data["cohort_month"] = cohort_data["cohort_month"].astype(str)
    cohort_data["order_month"]  = cohort_data["order_month"].astype(str)

    return _safe_records(
        cohort_data[["cohort_month", "order_month", "period_number",
                     "customers", "cohort_size", "retention_pct"]]
        .sort_values(["cohort_month", "period_number"])
    )


# ── DATA QUALITY ──────────────────────────────────────────────────────────────
@app.get("/data-quality", tags=["Admin"])
def data_quality():
    df = _load_dq()
    if df.empty:
        raise HTTPException(status_code=404, detail="data_quality_report.csv not found. Run etl_retail.py first.")
    return _safe_records(df)