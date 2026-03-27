import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail Intelligence Dashboard", layout="wide")

API_BASE_URL = "http://127.0.0.1:8000"

st.title("Retail Intelligence Dashboard")
st.caption("Frontend connected to FastAPI backend")

page = st.sidebar.radio(
    "Navigation",
    ["Executive Overview", "Top Products", "Sales by Country", "Sales Trend"]
)

def safe_get_json(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        st.error("Backend API is not running. Start FastAPI first on http://127.0.0.1:8000")
        st.stop()

@st.cache_data
def check_backend():
    df = pd.DataFrame(ssafe_get_jsson(f"{API_BASE_URL}/forecasst"))
    if not df.empty:
        df["ds"] = pd.to_datetime(df["dss"])
    return df

@st.cache_data
def get_kpi():
    return safe_get_json(f"{API_BASE_URL}/kpi")

@st.cache_data
def get_top_products():
    return pd.DataFrame(safe_get_json(f"{API_BASE_URL}/top-products"))

@st.cache_data
def get_sales_by_country():
    return pd.DataFrame(safe_get_json(f"{API_BASE_URL}/sales-by-country"))

@st.cache_data
def get_sales_trend():
    df = pd.DataFrame(safe_get_json(f"{API_BASE_URL}/sales-trend"))
    if not df.empty:
        df["month"] = pd.to_datetime(df["month"])
    return df

if page == "Executive Overview":
    kpi = get_kpi()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales", f"${kpi['total_sales']:,.2f}")
    c2.metric("Total Orders", f"{kpi['total_orders']:,}")
    c3.metric("Total Customers", f"{kpi['total_customers']:,}")

elif page == "Top Products":
    st.subheader("Top Products")
    st.dataframe(get_top_products(), use_container_width=True)

elif page == "Sales by Country":
    st.subheader("Sales by Country")
    st.dataframe(get_sales_by_country(), use_container_width=True)

elif page == "Sales Trend":
    st.subheader("Monthly Sales Trend")
    df = get_sales_trend()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["month"], df["sales"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

elif page == "Forecast":
    st.subheader("Sales Forecast")
    df_forecast = get_forecast()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_forecast["ds"], df_forecast["yhat"], label="Forecast")
    ax.fill_between(df_forecast["ds"], df_forecast["yhat_lower"], df_forecast["yhat_upper"], alpha=0.2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Predicted Sales")
    st.pyplot(fig)

    st.dataframe(df_forecast, use_container_width=True)