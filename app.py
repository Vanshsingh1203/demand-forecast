import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_all_data, get_family_sales, get_abc_classification, get_aggregated_sales
from src.forecasting import train_test_split_ts, run_all_forecasts, forecast_moving_average, forecast_exp_smoothing, forecast_arima, forecast_prophet, forecast_xgboost, evaluate_model
from src.inventory import full_inventory_analysis, what_if_analysis, cost_curve_data

# ─── Page Config ───
st.set_page_config(page_title="Demand Forecast & Inventory Optimizer", page_icon="📦", layout="wide")

# ─── Custom CSS ───
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    return load_all_data()


def main():
    # ─── Sidebar ───
    st.sidebar.title("📦 Demand Forecast")
    st.sidebar.caption("Inventory Optimization Tool")

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Exploratory Analysis", "Forecasting", "Inventory Optimization", "ABC Analysis", "What-If Simulator"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Built by Vansh Singh**")
    st.sidebar.caption("MS Engineering Management\nNortheastern University")

    # Load data
    df = load_data()

    families = sorted(df["family"].unique())
    stores = sorted(df["store_nbr"].unique())

    # ═══════════════════════════════════════════
    # OVERVIEW
    # ═══════════════════════════════════════════
    if page == "Overview":
        st.title("Demand Forecasting & Inventory Optimization")
        st.caption("Analyzing 1.7M+ sales records across 54 stores and 33 product families")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Stores", df["store_nbr"].nunique())
        col3.metric("Product Families", df["family"].nunique())
        col4.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days:,} days")
        col5.metric("Total Sales", f"${df['sales'].sum():,.0f}")

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Monthly Sales Trend")
            monthly = get_aggregated_sales(df, "M")
            fig = px.line(monthly, x="date", y="total_sales", labels={"total_sales": "Sales", "date": "Date"})
            fig.update_layout(height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Top 10 Product Families")
            top_fam = df.groupby("family")["sales"].sum().sort_values(ascending=True).tail(10).reset_index()
            fig = px.bar(top_fam, x="sales", y="family", orientation="h", labels={"sales": "Total Sales", "family": "Product Family"})
            fig.update_layout(height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Sales by Store Type")
            store_type = df.groupby("type")["sales"].sum().reset_index()
            fig = px.pie(store_type, values="sales", names="type", hole=0.4)
            fig.update_layout(height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            st.subheader("Oil Price vs Sales (Monthly)")
            monthly = get_aggregated_sales(df, "M")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=monthly["date"], y=monthly["total_sales"], name="Sales", line=dict(color="#6366f1")), secondary_y=False)
            fig.add_trace(go.Scatter(x=monthly["date"], y=monthly["avg_oil_price"], name="Oil Price", line=dict(color="#f59e0b", dash="dash")), secondary_y=True)
            fig.update_layout(height=350, margin=dict(t=10, b=10))
            fig.update_yaxes(title_text="Sales", secondary_y=False)
            fig.update_yaxes(title_text="Oil Price ($)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════
    # EXPLORATORY ANALYSIS
    # ═══════════════════════════════════════════
    elif page == "Exploratory Analysis":
        st.title("Exploratory Data Analysis")

        sel_family = st.selectbox("Select Product Family", families)
        freq = st.radio("Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True)
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}

        fam_data = get_family_sales(df, sel_family, freq_map[freq])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Sales", f"${fam_data['sales'].sum():,.0f}")
        c2.metric("Avg Daily Sales", f"${df[df['family']==sel_family]['sales'].mean():,.1f}")
        c3.metric("Peak Sales", f"${fam_data['sales'].max():,.0f}")

        st.subheader(f"{sel_family} — Sales Over Time ({freq})")
        fig = px.line(fam_data, x="date", y="sales")
        fig.update_layout(height=400, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Day of Week Pattern")
            fam_daily = df[df["family"] == sel_family].copy()
            dow = fam_daily.groupby("day_name")["sales"].mean().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index()
            fig = px.bar(dow, x="day_name", y="sales", labels={"day_name": "Day", "sales": "Avg Sales"})
            fig.update_layout(height=300, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Monthly Seasonality")
            monthly_avg = fam_daily.groupby("month")["sales"].mean().reset_index()
            fig = px.bar(monthly_avg, x="month", y="sales", labels={"month": "Month", "sales": "Avg Sales"})
            fig.update_layout(height=300, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Promotion Impact")
        promo = fam_daily.groupby("onpromotion")["sales"].mean()
        if len(promo) >= 2:
            promo_lift = ((promo.get(1, 0) - promo.get(0, 0)) / promo.get(0, 1)) * 100
            st.metric("Promo Sales Lift", f"{promo_lift:.1f}%", delta=f"{'↑' if promo_lift > 0 else '↓'}")
        else:
            st.info("Not enough promotion data for this family.")

    # ═══════════════════════════════════════════
    # FORECASTING
    # ═══════════════════════════════════════════
    elif page == "Forecasting":
        st.title("Demand Forecasting")

        sel_family = st.selectbox("Select Product Family", families)
        freq = st.radio("Forecast Frequency", ["Weekly", "Monthly"], horizontal=True)
        freq_map = {"Weekly": "W", "Monthly": "M"}

        fam_data = get_family_sales(df, sel_family, freq_map[freq])

        if len(fam_data) < 20:
            st.warning("Not enough data points for reliable forecasting. Try a different frequency.")
        else:
            train, test = train_test_split_ts(fam_data, "date", "sales", test_ratio=0.2)

            st.info(f"Training on {len(train)} periods, testing on {len(test)} periods")

            with st.spinner("Running 5 forecasting models... This may take a moment."):
                results = run_all_forecasts(train, test, "sales", "date")

            # Metrics table
            st.subheader("Model Comparison")
            display_cols = ["model", "MAE", "RMSE", "MAPE"]
            st.dataframe(
                results[display_cols].style.highlight_min(subset=["MAE", "RMSE", "MAPE"], color="#d1fae5"),
                use_container_width=True,
                hide_index=True,
            )

            best = results.iloc[0]
            st.success(f"Best model: **{best['model']}** with MAPE of **{best['MAPE']:.2f}%**")

            # Plot forecast vs actuals
            st.subheader("Forecast vs Actuals")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train["date"], y=train["sales"], name="Training Data", line=dict(color="#94a3b8")))
            fig.add_trace(go.Scatter(x=test["date"], y=test["sales"], name="Actual", line=dict(color="#0f172a", width=2)))

            colors = {"Moving": "#f59e0b", "Holt": "#3b82f6", "ARIMA": "#10b981", "Prophet": "#8b5cf6", "XGBoost": "#e11d48"}
            for _, row in results.iterrows():
                preds = row["predictions"]
                if len(preds) == len(test):
                    color = next((v for k, v in colors.items() if k in row["model"]), "#6366f1")
                    fig.add_trace(go.Scatter(x=test["date"], y=preds, name=f"{row['model']} (MAPE: {row['MAPE']:.1f}%)", line=dict(color=color, dash="dash")))

            fig.update_layout(height=500, margin=dict(t=10), legend=dict(orientation="h", yanchor="bottom", y=-0.3))
            st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════
    # INVENTORY OPTIMIZATION
    # ═══════════════════════════════════════════
    elif page == "Inventory Optimization":
        st.title("Inventory Optimization")

        sel_family = st.selectbox("Select Product Family", families)

        c1, c2, c3, c4 = st.columns(4)
        ordering_cost = c1.number_input("Ordering Cost ($)", value=50, min_value=1)
        unit_cost = c2.number_input("Unit Cost ($)", value=10.0, min_value=0.1)
        holding_pct = c3.number_input("Holding Cost (%)", value=25, min_value=1, max_value=100) / 100
        lead_time = c4.number_input("Lead Time (days)", value=7, min_value=1)

        service_level = st.slider("Target Service Level", min_value=0.85, max_value=0.99, value=0.95, step=0.01, format="%.0f%%")

        daily_sales = df[df["family"] == sel_family].groupby("date")["sales"].sum()

        result = full_inventory_analysis(daily_sales, ordering_cost, holding_pct, unit_cost, lead_time, service_level)

        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EOQ (units)", f"{result['eoq']:,.0f}")
        c2.metric("Safety Stock", f"{result['safety_stock']:,.0f}")
        c3.metric("Reorder Point", f"{result['reorder_point']:,.0f}")
        c4.metric("Total Annual Cost", f"${result['total_annual_cost']:,.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Avg Daily Demand", f"{result['avg_daily_demand']:,.1f}")
        c6.metric("Orders/Year", f"{result['orders_per_year']:.1f}")
        c7.metric("Days Between Orders", f"{result['days_between_orders']:.1f}")
        c8.metric("Inventory Turnover", f"{result['inventory_turnover']:.1f}x")

        # EOQ Cost Curve
        st.subheader("Total Cost Curve (EOQ Analysis)")
        cost_data = cost_curve_data(result["annual_demand"], ordering_cost, unit_cost * holding_pct, result["eoq"])

        if not cost_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cost_data["order_qty"], y=cost_data["ordering_cost"], name="Ordering Cost", line=dict(color="#f59e0b", dash="dash")))
            fig.add_trace(go.Scatter(x=cost_data["order_qty"], y=cost_data["holding_cost"], name="Holding Cost", line=dict(color="#3b82f6", dash="dash")))
            fig.add_trace(go.Scatter(x=cost_data["order_qty"], y=cost_data["total_cost"], name="Total Cost", line=dict(color="#e11d48", width=3)))
            fig.add_vline(x=result["eoq"], line_dash="dot", line_color="#10b981", annotation_text=f"EOQ = {result['eoq']:,.0f}")
            fig.update_layout(height=400, xaxis_title="Order Quantity", yaxis_title="Annual Cost ($)", margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════
    # ABC ANALYSIS
    # ═══════════════════════════════════════════
    elif page == "ABC Analysis":
        st.title("ABC Classification")
        st.caption("Classifying product families by revenue contribution (Pareto Principle)")

        abc = get_abc_classification(df)

        c1, c2, c3 = st.columns(3)
        c1.metric("A Items (Top 80%)", f"{len(abc[abc['abc_class']=='A'])} families")
        c2.metric("B Items (Next 15%)", f"{len(abc[abc['abc_class']=='B'])} families")
        c3.metric("C Items (Bottom 5%)", f"{len(abc[abc['abc_class']=='C'])} families")

        # Pareto chart
        st.subheader("Pareto Chart")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        colors = abc["abc_class"].map({"A": "#6366f1", "B": "#f59e0b", "C": "#94a3b8"}).tolist()
        fig.add_trace(go.Bar(x=abc["family"], y=abc["total_sales"], name="Sales", marker_color=colors), secondary_y=False)
        fig.add_trace(go.Scatter(x=abc["family"], y=abc["cumulative_pct"], name="Cumulative %", line=dict(color="#e11d48", width=2)), secondary_y=True)
        fig.add_hline(y=80, line_dash="dash", line_color="#10b981", secondary_y=True, annotation_text="80%")
        fig.add_hline(y=95, line_dash="dash", line_color="#f59e0b", secondary_y=True, annotation_text="95%")
        fig.update_layout(height=500, margin=dict(t=10), xaxis_tickangle=-45)
        fig.update_yaxes(title_text="Total Sales", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Classification Table")
        display = abc[["family", "total_sales", "cumulative_pct", "abc_class"]].copy()
        display["total_sales"] = display["total_sales"].apply(lambda x: f"${x:,.0f}")
        display["cumulative_pct"] = display["cumulative_pct"].apply(lambda x: f"{x:.1f}%")
        display.columns = ["Product Family", "Total Sales", "Cumulative %", "Class"]
        st.dataframe(display, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════
    # WHAT-IF SIMULATOR
    # ═══════════════════════════════════════════
    elif page == "What-If Simulator":
        st.title("What-If Simulator")
        st.caption("See how changing parameters affects inventory costs and stock levels")

        sel_family = st.selectbox("Select Product Family", families)

        c1, c2 = st.columns(2)
        ordering_cost = c1.number_input("Ordering Cost ($)", value=50, min_value=1, key="wi_oc")
        unit_cost = c1.number_input("Unit Cost ($)", value=10.0, min_value=0.1, key="wi_uc")
        holding_pct = c2.number_input("Holding Cost (%)", value=25, min_value=1, max_value=100, key="wi_hp") / 100

        lead_times = st.multiselect("Lead Times to Compare (days)", [1, 3, 5, 7, 10, 14, 21, 30], default=[3, 7, 14])
        service_levels = st.multiselect("Service Levels to Compare", [0.85, 0.90, 0.95, 0.97, 0.99], default=[0.90, 0.95, 0.99])

        if lead_times and service_levels:
            daily_sales = df[df["family"] == sel_family].groupby("date")["sales"].sum()

            wi_results = what_if_analysis(daily_sales, lead_times, service_levels, ordering_cost, holding_pct, unit_cost)

            st.subheader("Scenario Comparison")
            st.dataframe(wi_results, use_container_width=True, hide_index=True)

            st.subheader("Safety Stock by Scenario")
            fig = px.bar(wi_results, x="lead_time", y="safety_stock", color="service_level", barmode="group", labels={"lead_time": "Lead Time (days)", "safety_stock": "Safety Stock (units)"})
            fig.update_layout(height=400, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Total Cost by Scenario")
            fig = px.bar(wi_results, x="lead_time", y="total_cost", color="service_level", barmode="group", labels={"lead_time": "Lead Time (days)", "total_cost": "Total Annual Cost ($)"})
            fig.update_layout(height=400, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Select at least one lead time and one service level.")


if __name__ == "__main__":
    main()