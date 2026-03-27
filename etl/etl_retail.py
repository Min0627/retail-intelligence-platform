from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    superstore = pd.read_csv(RAW_DIR / "SampleSuperstore.csv")
    online = pd.read_csv(RAW_DIR / "online_retail.csv", encoding="latin1")
    return superstore, online


def clean_superstore(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    df = df.drop_duplicates()

    df["source"] = "superstore"
    df["order_id"] = ["SS_" + str(i).zfill(6) for i in range(1, len(df) + 1)]
    df["invoice_no"] = df["order_id"]
    df["order_date"] = pd.NaT
    df["customer_id"] = np.nan
    df["stock_code"] = np.nan
    df["description"] = np.nan
    df["unit_price"] = np.where(df["quantity"] > 0, df["sales"] / df["quantity"], np.nan)

    keep_cols = [
        "source",
        "order_id",
        "invoice_no",
        "order_date",
        "customer_id",
        "stock_code",
        "description",
        "category",
        "sub_category",
        "segment",
        "ship_mode",
        "country",
        "city",
        "state",
        "region",
        "postal_code",
        "quantity",
        "unit_price",
        "sales",
        "discount",
        "profit",
    ]

    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[keep_cols]

    numeric_cols = ["quantity", "unit_price", "sales", "discount", "profit"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    df = df.drop_duplicates()

    df["description"] = df["description"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")

    df = df.dropna(subset=["invoiceno", "stockcode", "quantity", "unitprice", "country"])
    df = df[~df["invoiceno"].astype(str).str.startswith("C")]
    df = df[(df["quantity"] > 0) & (df["unitprice"] > 0)]

    df["source"] = "online_retail"
    df["order_id"] = df["invoiceno"].astype(str)
    df["invoice_no"] = df["invoiceno"].astype(str)
    df["order_date"] = df["invoicedate"]
    df["customer_id"] = df["customerid"]
    df["stock_code"] = df["stockcode"]
    df["category"] = "Uncategorized"
    df["sub_category"] = "Uncategorized"
    df["segment"] = np.nan
    df["ship_mode"] = np.nan
    df["city"] = np.nan
    df["state"] = np.nan
    df["region"] = np.nan
    df["postal_code"] = np.nan
    df["sales"] = df["quantity"] * df["unitprice"]
    df["discount"] = 0.0
    df["profit"] = np.nan

    df = df.rename(columns={"unitprice": "unit_price"})

    keep_cols = [
        "source",
        "order_id",
        "invoice_no",
        "order_date",
        "customer_id",
        "stock_code",
        "description",
        "category",
        "sub_category",
        "segment",
        "ship_mode",
        "country",
        "city",
        "state",
        "region",
        "postal_code",
        "quantity",
        "unit_price",
        "sales",
        "discount",
        "profit",
    ]

    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[keep_cols]

    numeric_cols = ["quantity", "unit_price", "sales", "discount", "profit"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def create_dimensions_and_fact(df: pd.DataFrame):
    data = df.copy()

    dim_products = (
        data[["stock_code", "description", "category", "sub_category"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    dim_products["product_key"] = range(1, len(dim_products) + 1)

    dim_customers = (
        data[["customer_id", "country", "city", "state", "region", "postal_code", "segment"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    dim_customers["customer_key"] = range(1, len(dim_customers) + 1)

    fact = data.merge(
        dim_products,
        on=["stock_code", "description", "category", "sub_category"],
        how="left"
    ).merge(
        dim_customers,
        on=["customer_id", "country", "city", "state", "region", "postal_code", "segment"],
        how="left"
    )

    fact_sales = fact[
        [
            "source",
            "order_id",
            "invoice_no",
            "order_date",
            "customer_key",
            "product_key",
            "ship_mode",
            "quantity",
            "unit_price",
            "sales",
            "discount",
            "profit",
        ]
    ].copy()

    return dim_products, dim_customers, fact_sales


def run_etl():
    superstore_raw, online_raw = load_data()

    superstore_clean = clean_superstore(superstore_raw)
    online_clean = clean_online_retail(online_raw)

    merged = pd.concat([superstore_clean, online_clean], ignore_index=True)
    merged = merged.drop_duplicates()
    merged["order_date"] = pd.to_datetime(merged["order_date"], errors="coerce")

    dim_products, dim_customers, fact_sales = create_dimensions_and_fact(merged)

    merged.to_csv(PROCESSED_DIR / "retail_merged.csv", index=False)
    dim_products.to_csv(PROCESSED_DIR / "dim_products.csv", index=False)
    dim_customers.to_csv(PROCESSED_DIR / "dim_customers.csv", index=False)
    fact_sales.to_csv(PROCESSED_DIR / "fact_sales.csv", index=False)

    print("ETL complete.")
    print(f"Merged rows: {len(merged):,}")
    print(f"Products: {len(dim_products):,}")
    print(f"Customers: {len(dim_customers):,}")
    print(f"Fact rows: {len(fact_sales):,}")


if __name__ == "__main__":
    run_etl()