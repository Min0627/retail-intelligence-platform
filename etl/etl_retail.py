"""
etl_retail.py  —  Upgraded Retail Intelligence ETL Pipeline
============================================================
Improvements over v1:
  • Data quality report (null rates, duplicate counts, outlier flags)
  • Returned orders kept as dim_returns instead of dropped
  • Profit margin % column on fact table
  • RFM scoring (Recency, Frequency, Monetary) per customer
  • Cohort month column for retention analysis
  • Product return rate output table
  • All outputs saved to data/processed/
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[1]
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    superstore = pd.read_csv(RAW_DIR / "SampleSuperstore.csv")
    online     = pd.read_csv(RAW_DIR / "online_retail.csv", encoding="latin1")
    return superstore, online


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN
# ══════════════════════════════════════════════════════════════════════════════
def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    return df


def clean_superstore(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalise_cols(df.copy()).drop_duplicates()

    df["source"]      = "superstore"
    df["order_id"]    = ["SS_" + str(i).zfill(6) for i in range(1, len(df) + 1)]
    df["invoice_no"]  = df["order_id"]
    df["order_date"]  = pd.to_datetime(df.get("order_date"), errors="coerce")
    df["customer_id"] = df.get("customer_id", np.nan)
    df["stock_code"]  = np.nan
    df["description"] = np.nan
    df["unit_price"]  = np.where(df["quantity"] > 0, df["sales"] / df["quantity"], np.nan)
    df["is_return"]   = False

    KEEP = [
        "source", "order_id", "invoice_no", "order_date", "customer_id",
        "stock_code", "description", "category", "sub_category", "segment",
        "ship_mode", "country", "city", "state", "region", "postal_code",
        "quantity", "unit_price", "sales", "discount", "profit", "is_return",
    ]
    for c in KEEP:
        if c not in df.columns:
            df[c] = np.nan
    df = df[KEEP]

    for c in ["quantity", "unit_price", "sales", "discount", "profit"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def clean_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalise_cols(df.copy()).drop_duplicates()

    df["description"] = df["description"].astype(str).str.strip()
    df["country"]     = df["country"].astype(str).str.strip()
    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")

    df = df.dropna(subset=["invoiceno", "stockcode", "quantity", "unitprice", "country"])

    # ── Separate returns before dropping them ──────────────────────────────
    returns_mask = df["invoiceno"].astype(str).str.startswith("C")
    df_returns   = df[returns_mask].copy()
    df           = df[~returns_mask].copy()

    df = df[(df["quantity"] > 0) & (df["unitprice"] > 0)]

    def _build(src: pd.DataFrame, is_return: bool) -> pd.DataFrame:
        out = src.copy()
        out["source"]      = "online_retail"
        out["order_id"]    = out["invoiceno"].astype(str)
        out["invoice_no"]  = out["invoiceno"].astype(str)
        out["order_date"]  = out["invoicedate"]
        out["customer_id"] = out.get("customerid", np.nan)
        out["stock_code"]  = out["stockcode"]
        out["category"]    = "Uncategorized"
        out["sub_category"]= "Uncategorized"
        out["segment"]     = np.nan
        out["ship_mode"]   = np.nan
        out["city"]        = np.nan
        out["state"]       = np.nan
        out["region"]      = np.nan
        out["postal_code"] = np.nan
        out["sales"]       = out["quantity"] * out["unitprice"]
        out["discount"]    = 0.0
        out["profit"]      = np.nan
        out["is_return"]   = is_return
        out = out.rename(columns={"unitprice": "unit_price"})
        KEEP = [
            "source", "order_id", "invoice_no", "order_date", "customer_id",
            "stock_code", "description", "category", "sub_category", "segment",
            "ship_mode", "country", "city", "state", "region", "postal_code",
            "quantity", "unit_price", "sales", "discount", "profit", "is_return",
        ]
        for c in KEEP:
            if c not in out.columns:
                out[c] = np.nan
        out = out[KEEP]
        for c in ["quantity", "unit_price", "sales", "discount", "profit"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    return _build(df, False), _build(df_returns, True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════
def data_quality_report(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = len(df)
    report = []
    for col in df.columns:
        nulls   = df[col].isna().sum()
        uniq    = df[col].nunique(dropna=True)
        dtype   = str(df[col].dtype)
        outliers = 0
        if pd.api.types.is_numeric_dtype(df[col]):
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        report.append({
            "dataset":      name,
            "column":       col,
            "dtype":        dtype,
            "null_count":   int(nulls),
            "null_pct":     round(nulls / rows * 100, 2),
            "unique_values":int(uniq),
            "outlier_count":outliers,
        })
    return pd.DataFrame(report)


# ══════════════════════════════════════════════════════════════════════════════
# 4. ENRICH MERGED DATA
# ══════════════════════════════════════════════════════════════════════════════
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Profit margin %
    df["profit_margin_pct"] = np.where(
        df["sales"] > 0,
        (df["profit"] / df["sales"] * 100).round(2),
        np.nan,
    )

    # Cohort month (first purchase month per customer)
    cohort = (
        df.dropna(subset=["customer_id", "order_date"])
        .groupby("customer_id")["order_date"]
        .min()
        .dt.to_period("M")
        .dt.to_timestamp()
        .rename("cohort_month")
        .reset_index()
    )
    df = df.merge(cohort, on="customer_id", how="left")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. RFM SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute RFM scores for customers with known customer_id and order_date.
    Segments: Champions, Loyal, Potential Loyalist, At Risk, Lost, New Customer.
    """
    base = df.dropna(subset=["customer_id", "order_date", "sales"]).copy()
    base = base[base["is_return"] == False]

    rfm = (
        base.groupby("customer_id")
        .agg(
            last_purchase=("order_date", "max"),
            frequency=("order_id", "nunique"),
            monetary=("sales", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (snapshot_date - rfm["last_purchase"]).dt.days

    # Quintile scoring (5 = best)
    rfm["r_score"] = pd.qcut(rfm["recency"],   5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"),  5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)

    def segment(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        avg = (r + f + m) / 3
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal"
        elif r >= 3 and f <= 2:
            return "Potential Loyalist"
        elif r == 2:
            return "At Risk"
        elif r == 1 and avg >= 3:
            return "Can't Lose Them"
        elif r == 1:
            return "Lost"
        else:
            return "New Customer"

    rfm["segment"] = rfm.apply(segment, axis=1)
    rfm["monetary"] = rfm["monetary"].round(2)
    rfm["recency"]  = rfm["recency"].astype(int)
    return rfm


# ══════════════════════════════════════════════════════════════════════════════
# 6. PRODUCT RETURN RATES
# ══════════════════════════════════════════════════════════════════════════════
def compute_return_rates(df_sales: pd.DataFrame, df_returns: pd.DataFrame) -> pd.DataFrame:
    """Return rate per product description."""
    sales_count = (
        df_sales.dropna(subset=["description"])
        .groupby("description")
        .agg(total_orders=("order_id", "count"), total_sales=("sales", "sum"))
        .reset_index()
    )
    return_count = (
        df_returns.dropna(subset=["description"])
        .groupby("description")
        .agg(return_orders=("order_id", "count"), return_value=("sales", "sum"))
        .reset_index()
    )
    merged = sales_count.merge(return_count, on="description", how="left").fillna(0)
    merged["return_rate_pct"] = (
        merged["return_orders"] / (merged["total_orders"] + merged["return_orders"]) * 100
    ).round(2)
    merged["return_value"] = merged["return_value"].abs().round(2)
    return merged.sort_values("return_rate_pct", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# 7. STAR SCHEMA DIMENSIONS + FACT
# ══════════════════════════════════════════════════════════════════════════════
def create_star_schema(df: pd.DataFrame):
    dim_products = (
        df[["stock_code", "description", "category", "sub_category"]]
        .drop_duplicates().reset_index(drop=True)
    )
    dim_products["product_key"] = range(1, len(dim_products) + 1)

    dim_customers = (
        df[["customer_id", "country", "city", "state", "region", "postal_code", "segment"]]
        .drop_duplicates().reset_index(drop=True)
    )
    dim_customers["customer_key"] = range(1, len(dim_customers) + 1)

    fact = (
        df.merge(dim_products, on=["stock_code", "description", "category", "sub_category"], how="left")
          .merge(dim_customers, on=["customer_id", "country", "city", "state", "region", "postal_code", "segment"], how="left")
    )

    fact_cols = [
        "source", "order_id", "invoice_no", "order_date", "cohort_month",
        "customer_key", "product_key", "ship_mode",
        "quantity", "unit_price", "sales", "discount", "profit",
        "profit_margin_pct", "is_return",
    ]
    for c in fact_cols:
        if c not in fact.columns:
            fact[c] = np.nan

    return dim_products, dim_customers, fact[fact_cols]


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN ETL
# ══════════════════════════════════════════════════════════════════════════════
def run_etl():
    print("=" * 60)
    print("  Retail Intelligence ETL — starting")
    print("=" * 60)

    # Load
    superstore_raw, online_raw = load_data()
    print(f"Loaded Superstore: {len(superstore_raw):,} rows")
    print(f"Loaded Online Retail: {len(online_raw):,} rows")

    # Clean
    superstore_clean               = clean_superstore(superstore_raw)
    online_clean, online_returns   = clean_online_retail(online_raw)
    print(f"Online returns (C-invoices): {len(online_returns):,} rows")

    # Merge sales
    merged = pd.concat([superstore_clean, online_clean], ignore_index=True).drop_duplicates()
    merged["order_date"] = pd.to_datetime(merged["order_date"], errors="coerce")

    # Data quality
    dq = pd.concat([
        data_quality_report(superstore_clean, "superstore"),
        data_quality_report(online_clean,     "online_retail"),
    ], ignore_index=True)

    # Enrich
    merged = enrich(merged)

    # RFM
    snapshot = merged["order_date"].max()
    rfm = compute_rfm(merged, snapshot)
    print(f"RFM segments computed for {len(rfm):,} customers (snapshot: {snapshot.date()})")

    # Return rates
    return_rates = compute_return_rates(merged, online_returns)

    # Star schema
    dim_products, dim_customers, fact_sales = create_star_schema(merged)

    # ── Save ────────────────────────────────────────────────────────────────
    merged.to_csv(PROCESSED_DIR / "retail_merged.csv",       index=False)
    dim_products.to_csv(PROCESSED_DIR / "dim_products.csv",  index=False)
    dim_customers.to_csv(PROCESSED_DIR / "dim_customers.csv",index=False)
    fact_sales.to_csv(PROCESSED_DIR / "fact_sales.csv",      index=False)
    rfm.to_csv(PROCESSED_DIR / "rfm_segments.csv",           index=False)
    online_returns.to_csv(PROCESSED_DIR / "dim_returns.csv", index=False)
    return_rates.to_csv(PROCESSED_DIR / "product_return_rates.csv", index=False)
    dq.to_csv(PROCESSED_DIR / "data_quality_report.csv",     index=False)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ETL complete — files written to data/processed/")
    print("=" * 60)
    print(f"  retail_merged.csv          {len(merged):>10,} rows")
    print(f"  fact_sales.csv             {len(fact_sales):>10,} rows")
    print(f"  dim_products.csv           {len(dim_products):>10,} rows")
    print(f"  dim_customers.csv          {len(dim_customers):>10,} rows")
    print(f"  rfm_segments.csv           {len(rfm):>10,} rows")
    print(f"  dim_returns.csv            {len(online_returns):>10,} rows")
    print(f"  product_return_rates.csv   {len(return_rates):>10,} rows")
    print(f"  data_quality_report.csv    {len(dq):>10,} rows")

    # RFM segment breakdown
    print("\n  RFM Segment Distribution:")
    seg_counts = rfm["segment"].value_counts()
    for seg, cnt in seg_counts.items():
        print(f"    {seg:<25} {cnt:>6,}")

    print("\n  Top 5 highest-return-rate products:")
    print(return_rates[["description", "return_rate_pct", "return_orders"]].head())


if __name__ == "__main__":
    run_etl()