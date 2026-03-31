"""
forecast.py  —  Upgraded Sales Forecasting Pipeline
=====================================================
Improvements over v1:
  • 12-month forecast horizon (was 6)
  • Yearly + weekly seasonality tuning
  • Changepoint sensitivity tuning (0.3)
  • Holiday effects (United Kingdom)
  • MAPE & RMSE evaluation via cross-validation
  • Forecast saved with evaluation metadata
  • Separate output: forecast_metrics.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[1]
DATA_PATH    = BASE_DIR / "data" / "processed" / "retail_merged.csv"
OUTPUT_PATH  = BASE_DIR / "data" / "processed" / "sales_forecast.csv"
METRICS_PATH = BASE_DIR / "data" / "processed" / "forecast_metrics.csv"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & AGGREGATE TO MONTHLY
# ══════════════════════════════════════════════════════════════════════════════
def load_monthly() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Exclude returns from sales signal
    if "is_return" in df.columns:
        df = df[df["is_return"] == False]

    monthly = (
        df.dropna(subset=["order_date"])
        .assign(month=lambda x: x["order_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["sales"]
        .sum()
        .sort_values("month")
        .rename(columns={"month": "ds", "sales": "y"})
    )

    # Remove incomplete trailing months that skew forecasts
    monthly = monthly[monthly["ds"] < monthly["ds"].max()]

    print(f"Training data: {len(monthly)} months  "
          f"({monthly['ds'].min().date()} → {monthly['ds'].max().date()})")
    print(f"Avg monthly sales: £{monthly['y'].mean():,.0f}  |  "
          f"Max: £{monthly['y'].max():,.0f}  |  "
          f"Min: £{monthly['y'].min():,.0f}")

    return monthly


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD & FIT PROPHET MODEL
# ══════════════════════════════════════════════════════════════════════════════
def build_model() -> Prophet:
    """
    Tuned Prophet configuration:
    - changepoint_prior_scale=0.3   → more flexible trend changes
    - seasonality_prior_scale=15    → stronger seasonal fit
    - yearly_seasonality=True       → capture annual patterns
    - weekly_seasonality=False      → monthly data, not relevant
    - daily_seasonality=False       → monthly data, not relevant
    - country_holidays='GB'         → UK public holidays
    """
    model = Prophet(
        changepoint_prior_scale=0.3,
        seasonality_prior_scale=15,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # better for retail with growth
        interval_width=0.90,                # 90% confidence interval
    )

    # Add UK public holidays (dominant market in Online Retail dataset)
    model.add_country_holidays(country_name="GB")

    # Custom monthly seasonality for retail peaks (Q4)
    model.add_seasonality(
        name="quarterly",
        period=91.25,
        fourier_order=5,
    )

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3. CROSS-VALIDATION & METRICS
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model: Prophet, df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward cross-validation.
    initial  = first 60% of data used for first training window
    period   = re-train every 2 months
    horizon  = evaluate 3 months ahead
    """
    n_months     = len(df)
    initial_days = int(n_months * 0.60) * 30
    period_days  = 60
    horizon_days = 90

    print(f"\nRunning cross-validation "
          f"(initial={initial_days}d, period={period_days}d, horizon={horizon_days}d) ...")

    try:
        cv_results = cross_validation(
            model,
            initial=f"{initial_days} days",
            period=f"{period_days} days",
            horizon=f"{horizon_days} days",
            parallel=None,
        )
        metrics = performance_metrics(cv_results)

        mape = metrics["mape"].mean() * 100
        rmse = metrics["rmse"].mean()
        mae  = metrics["mae"].mean()

        print(f"  MAPE : {mape:.2f}%")
        print(f"  RMSE : £{rmse:,.0f}")
        print(f"  MAE  : £{mae:,.0f}")

        return metrics[["horizon", "mape", "rmse", "mae"]].copy()

    except Exception as e:
        print(f"  Cross-validation skipped: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 4. FORECAST
# ══════════════════════════════════════════════════════════════════════════════
def run_forecast():
    print("=" * 60)
    print("  Retail Intelligence Forecast — starting")
    print("=" * 60)

    # Load data
    monthly = load_monthly()

    # Build & fit
    model = build_model()
    print("\nFitting Prophet model ...")
    model.fit(monthly)
    print("Model fitted.")

    # Evaluate
    metrics_df = evaluate_model(model, monthly)

    # Forecast 12 months ahead
    print("\nGenerating 12-month forecast ...")
    future   = model.make_future_dataframe(periods=12, freq="ME")
    forecast = model.predict(future)

    # Clip negative lower bounds (sales can't be < 0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat"]       = forecast["yhat"].clip(lower=0)

    # ── Annotate historical vs forecast rows ────────────────────────────────
    last_train_date = monthly["ds"].max()
    forecast["type"] = np.where(forecast["ds"] <= last_train_date, "historical", "forecast")

    # ── Select & round output columns ───────────────────────────────────────
    output_cols = ["ds", "yhat", "yhat_lower", "yhat_upper",
                   "trend", "yearly", "type"]
    # Some columns may not exist if seasonality was skipped
    output_cols = [c for c in output_cols if c in forecast.columns]
    forecast_out = forecast[output_cols].copy()
    for col in ["yhat", "yhat_lower", "yhat_upper", "trend"]:
        if col in forecast_out.columns:
            forecast_out[col] = forecast_out[col].round(2)

    # ── Save ────────────────────────────────────────────────────────────────
    forecast_out.to_csv(OUTPUT_PATH, index=False)
    if not metrics_df.empty:
        metrics_df.to_csv(METRICS_PATH, index=False)

    # ── Summary ─────────────────────────────────────────────────────────────
    future_rows = forecast_out[forecast_out["type"] == "forecast"]
    print("\n" + "=" * 60)
    print("  Forecast complete — files written to data/processed/")
    print("=" * 60)
    print(f"  sales_forecast.csv    {len(forecast_out):>6} rows  "
          f"({len(future_rows)} forecast months)")
    if not metrics_df.empty:
        print(f"  forecast_metrics.csv  {len(metrics_df):>6} rows")

    print("\n  12-Month Forecast Preview:")
    print(
        future_rows[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .rename(columns={"ds": "month", "yhat": "forecast",
                         "yhat_lower": "lower_90", "yhat_upper": "upper_90"})
        .to_string(index=False)
    )


if __name__ == "__main__":
    run_forecast()