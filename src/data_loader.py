import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_all_data():
    """Load and merge all CSV files into a single clean dataframe."""

    # Load raw files
    sales = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
    oil = pd.read_csv(os.path.join(DATA_DIR, "oil.csv"), parse_dates=["date"])
    holidays = pd.read_csv(os.path.join(DATA_DIR, "holidays_events.csv"), parse_dates=["date"])
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"), parse_dates=["date"])

    print(f"Sales records: {len(sales):,}")
    print(f"Stores: {len(stores)}")
    print(f"Oil prices: {len(oil)}")
    print(f"Holidays: {len(holidays)}")
    print(f"Transactions: {len(transactions)}")

    # --- Clean oil prices (fill missing with forward fill) ---
    oil = oil.rename(columns={"dcoilwtico": "oil_price"})
    oil["oil_price"] = oil["oil_price"].ffill().bfill()

    # --- Clean holidays ---
    # Keep only national and regional holidays, remove transferred/bridge
    holidays_clean = holidays[
        (holidays["transferred"] == False) &
        (holidays["type"].isin(["Holiday", "Event"]))
    ][["date", "description"]].drop_duplicates(subset=["date"])
    holidays_clean["is_holiday"] = 1

    # --- Merge everything ---
    df = sales.merge(stores, on="store_nbr", how="left")
    df = df.merge(oil, on="date", how="left")
    df = df.merge(holidays_clean[["date", "is_holiday"]], on="date", how="left")
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")

    # Fill missing values
    df["oil_price"] = df["oil_price"].ffill().bfill()
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    df["transactions"] = df["transactions"].fillna(0).astype(int)
    df["onpromotion"] = df["onpromotion"].fillna(0).astype(int)

    # --- Feature Engineering ---
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_name"] = df["date"].dt.day_name()
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["quarter"] = df["date"].dt.quarter

    # Payday flag (15th and last day of month are common paydays in Ecuador)
    df["is_payday"] = ((df["date"].dt.day == 15) | df["date"].dt.is_month_end).astype(int)

    print(f"\nFinal merged dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Stores: {df['store_nbr'].nunique()}, Families: {df['family'].nunique()}")

    return df


def get_aggregated_sales(df, freq="W"):
    """
    Aggregate sales by date at a given frequency.
    freq: 'D' = daily, 'W' = weekly, 'M' = monthly
    """
    agg = df.groupby(pd.Grouper(key="date", freq=freq)).agg(
        total_sales=("sales", "sum"),
        avg_sales=("sales", "mean"),
        total_transactions=("transactions", "sum"),
        avg_oil_price=("oil_price", "mean"),
        promo_count=("onpromotion", "sum"),
    ).reset_index()
    return agg


def get_family_sales(df, family, freq="D"):
    """Get aggregated sales for a specific product family."""
    family_df = df[df["family"] == family].copy()
    agg = family_df.groupby(pd.Grouper(key="date", freq=freq)).agg(
        sales=("sales", "sum"),
        transactions=("transactions", "sum"),
        onpromotion=("onpromotion", "sum"),
    ).reset_index()
    return agg


def get_store_sales(df, store_nbr, freq="D"):
    """Get aggregated sales for a specific store."""
    store_df = df[df["store_nbr"] == store_nbr].copy()
    agg = store_df.groupby(pd.Grouper(key="date", freq=freq)).agg(
        sales=("sales", "sum"),
        transactions=("transactions", "sum"),
    ).reset_index()
    return agg


def get_abc_classification(df):
    """
    ABC Analysis: Classify product families by revenue contribution.
    A = top 80% of revenue (vital few)
    B = next 15% of revenue
    C = bottom 5% of revenue
    """
    family_revenue = df.groupby("family")["sales"].sum().sort_values(ascending=False).reset_index()
    family_revenue.columns = ["family", "total_sales"]
    family_revenue["cumulative_sales"] = family_revenue["total_sales"].cumsum()
    family_revenue["cumulative_pct"] = family_revenue["cumulative_sales"] / family_revenue["total_sales"].sum() * 100

    family_revenue["abc_class"] = "C"
    family_revenue.loc[family_revenue["cumulative_pct"] <= 80, "abc_class"] = "A"
    family_revenue.loc[
        (family_revenue["cumulative_pct"] > 80) & (family_revenue["cumulative_pct"] <= 95), "abc_class"
    ] = "B"

    return family_revenue


if __name__ == "__main__":
    # Quick test
    df = load_all_data()
    print("\nColumns:", list(df.columns))
    print("\nSample:")
    print(df.head())
    print("\nTop 5 product families by sales:")
    print(df.groupby("family")["sales"].sum().sort_values(ascending=False).head())