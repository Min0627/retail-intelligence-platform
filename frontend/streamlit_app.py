import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Retail Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://127.0.0.1:8000"

# ---------- THEME ----------
TEXT = "#111827"
MUTED = "#6b7280"
BG = "#f5f7fb"
CARD = "#ffffff"
BORDER = "#e5e7eb"
GRID = "#dbe2ea"
NAVY = "#0f172a"
BLUE = "#2563eb"
TEAL = "#0f766e"
RED = "#dc2626"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}

.stApp {{
    background-color: {BG};
}}

.block-container {{
    max-width: 1320px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}}

section[data-testid="stSidebar"] {{
    background: #ffffff;
    border-right: 1px solid {BORDER};
}}

section[data-testid="stSidebar"] * {{
    color: {TEXT} !important;
}}

.hero {{
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 24px 28px;
    border-radius: 20px;
    color: white;
    margin-bottom: 1rem;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.10);
}}

.hero-title {{
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
    color: #ffffff;
}}

.hero-sub {{
    font-size: 0.96rem;
    color: #cbd5e1;
}}

.metric-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}}

.metric-label {{
    font-size: 0.82rem;
    color: {MUTED};
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    letter-spacing: .04em;
}}

.metric-value {{
    font-size: 1.9rem;
    font-weight: 800;
    color: {TEXT};
    line-height: 1.1;
}}

.metric-sub {{
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}}

.card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 14px 14px 8px 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}}

.section-title {{
    font-size: 1.08rem;
    font-weight: 700;
    color: {TEXT};
    margin-bottom: 0.7rem;
}}

[data-baseweb="tab-list"] {{
    gap: 10px;
}}

button[data-baseweb="tab"] {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 10px 16px;
}}

button[data-baseweb="tab"] p {{
    color: {TEXT} !important;
    font-weight: 600;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    background: {NAVY};
    border-color: {NAVY};
}}

button[data-baseweb="tab"][aria-selected="true"] p {{
    color: #ffffff !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
def safe_get_json(url: str):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        st.error("Backend API is not running. Start FastAPI first on http://127.0.0.1:8000")
        st.stop()

def style_fig(fig, height=400):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, size=13),
        legend=dict(font=dict(color=TEXT, size=12), title=dict(text=""))
    )
    fig.update_xaxes(
        title_font=dict(color=TEXT),
        tickfont=dict(color=TEXT),
        showgrid=False,
        zeroline=False
    )
    fig.update_yaxes(
        title_font=dict(color=TEXT),
        tickfont=dict(color=TEXT),
        gridcolor=GRID,
        zeroline=False
    )
    return fig

@st.cache_data
def get_kpi():
    return safe_get_json(f"{API_BASE_URL}/kpi")

@st.cache_data
def get_top_products():
    return pd.DataFrame(safe_get_json(f"{API_BASE_URL}/top-products?limit=20"))

@st.cache_data
def get_sales_by_country():
    return pd.DataFrame(safe_get_json(f"{API_BASE_URL}/sales-by-country"))

@st.cache_data
def get_sales_trend():
    df = pd.DataFrame(safe_get_json(f"{API_BASE_URL}/sales-trend"))
    if not df.empty:
        df["month"] = pd.to_datetime(df["month"])
    return df

@st.cache_data
def get_forecast():
    df = pd.DataFrame(safe_get_json(f"{API_BASE_URL}/forecast"))
    if not df.empty:
        df["ds"] = pd.to_datetime(df["ds"])
    return df

# ---------- LOAD ----------
kpi = get_kpi()
df_products = get_top_products()
df_country = get_sales_by_country()
df_trend = get_sales_trend()
df_forecast = get_forecast()

# ---------- SIDEBAR ----------
st.sidebar.markdown("## Filters")
top_n = st.sidebar.slider("Top N", 5, 20, 10)
chart_height = st.sidebar.slider("Chart Height", 320, 520, 400, 20)

st.sidebar.markdown("---")
st.sidebar.caption("Frontend: Streamlit")
st.sidebar.caption("Backend: FastAPI")
st.sidebar.caption("Forecasting: Prophet")

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <div class="hero-title">Retail Intelligence Platform</div>
    <div class="hero-sub">Clean analytics dashboard for retail KPI tracking, product performance, country insights, and forecasting.</div>
</div>
""", unsafe_allow_html=True)

# ---------- KPI CARDS ----------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Sales</div>
        <div class="metric-value">${kpi['total_sales']:,.0f}</div>
        <div class="metric-sub">Merged retail data</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Orders</div>
        <div class="metric-value">{kpi['total_orders']:,}</div>
        <div class="metric-sub">Unique order records</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Customers</div>
        <div class="metric-value">{kpi['total_customers']:,}</div>
        <div class="metric-sub">Unique identified customers</div>
    </div>
    """, unsafe_allow_html=True)

avg_top_sales = df_products.head(top_n)["sales"].mean() if not df_products.empty else 0
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Top Product Sales</div>
        <div class="metric-value">${avg_top_sales:,.0f}</div>
        <div class="metric-sub">Current Top N selection</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ---------- TABS ----------
tab_overview, tab_products, tab_forecast = st.tabs(
    ["Overview", "Products", "Forecast"]
)

# ---------- OVERVIEW ----------
with tab_overview:
    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("<div class='section-title'>Monthly Sales Trend</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        fig_trend = px.line(df_trend, x="month", y="sales", markers=True)
        fig_trend.update_traces(line=dict(color=BLUE, width=3), marker=dict(size=7, color=BLUE))
        fig_trend.update_layout(xaxis_title="Month", yaxis_title="Sales")
        fig_trend = style_fig(fig_trend, chart_height)
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-title'>Top Countries by Sales</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        top_country = df_country.head(top_n).sort_values("sales", ascending=True)
        fig_country = px.bar(
            top_country,
            x="sales",
            y="country",
            orientation="h",
            text_auto=".2s"
        )
        fig_country.update_traces(marker_color=BLUE, textposition="outside")
        fig_country.update_layout(xaxis_title="Sales", yaxis_title="")
        fig_country = style_fig(fig_country, chart_height)
        fig_country.update_yaxes(showgrid=False)
        st.plotly_chart(fig_country, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------- PRODUCTS ----------
with tab_products:
    st.markdown("<div class='section-title'>Top Product Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    chart_df = df_products.head(top_n).sort_values("sales", ascending=True)
    fig_products = px.bar(
        chart_df,
        x="sales",
        y="description",
        orientation="h",
        text_auto=".2s"
    )
    fig_products.update_traces(marker_color=TEAL, textposition="outside")
    fig_products.update_layout(xaxis_title="Sales", yaxis_title="")
    fig_products = style_fig(fig_products, chart_height + 40)
    fig_products.update_yaxes(showgrid=False)
    st.plotly_chart(fig_products, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- FORECAST ----------
with tab_forecast:
    st.markdown("<div class='section-title'>Sales Forecast</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=df_forecast["ds"],
        y=df_forecast["yhat"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color=RED, width=3)
    ))
    fig_fc.add_trace(go.Scatter(
        x=df_forecast["ds"],
        y=df_forecast["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))
    fig_fc.add_trace(go.Scatter(
        x=df_forecast["ds"],
        y=df_forecast["yhat_lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        name="Confidence Range",
        fillcolor="rgba(220,38,38,0.15)"
    ))
    fig_fc.update_layout(xaxis_title="Month", yaxis_title="Predicted Sales")
    fig_fc = style_fig(fig_fc, chart_height + 20)
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)