import numpy as np
import pandas as pd
from scipy import stats


def calculate_eoq(annual_demand, ordering_cost, holding_cost_per_unit):
    """
    Economic Order Quantity (EOQ).
    Minimizes total cost = ordering cost + holding cost.
    """
    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_per_unit <= 0:
        return 0
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
    return round(eoq, 0)


def calculate_safety_stock(daily_demand_std, lead_time_days, service_level=0.95):
    """
    Safety Stock = Z * sigma_d * sqrt(L)
    Z = z-score for desired service level
    sigma_d = standard deviation of daily demand
    L = lead time in days
    """
    z = stats.norm.ppf(service_level)
    ss = z * daily_demand_std * np.sqrt(lead_time_days)
    return round(max(0, ss), 0)


def calculate_reorder_point(avg_daily_demand, lead_time_days, safety_stock):
    """
    Reorder Point = (avg daily demand * lead time) + safety stock
    """
    rop = (avg_daily_demand * lead_time_days) + safety_stock
    return round(max(0, rop), 0)


def calculate_total_cost(annual_demand, order_qty, ordering_cost, holding_cost_per_unit):
    """
    Total Inventory Cost = Ordering Cost + Holding Cost
    Ordering Cost = (D/Q) * S
    Holding Cost = (Q/2) * H
    """
    if order_qty <= 0:
        return 0
    ordering = (annual_demand / order_qty) * ordering_cost
    holding = (order_qty / 2) * holding_cost_per_unit
    return round(ordering + holding, 2)


def full_inventory_analysis(
    daily_sales_series,
    ordering_cost=50,
    holding_cost_pct=0.25,
    unit_cost=10,
    lead_time_days=7,
    service_level=0.95,
):
    """
    Complete inventory optimization for a product.

    Parameters:
    - daily_sales_series: pandas Series of daily sales
    - ordering_cost: cost per order placed ($)
    - holding_cost_pct: annual holding cost as % of unit cost
    - unit_cost: cost per unit ($)
    - lead_time_days: supplier lead time in days
    - service_level: desired fill rate (0.90, 0.95, 0.99)

    Returns dict with all inventory parameters.
    """
    daily_sales = daily_sales_series.dropna()
    daily_sales = daily_sales[daily_sales >= 0]

    avg_daily = daily_sales.mean()
    std_daily = daily_sales.std()
    annual_demand = avg_daily * 365

    holding_cost_per_unit = unit_cost * holding_cost_pct

    eoq = calculate_eoq(annual_demand, ordering_cost, holding_cost_per_unit)
    safety_stock = calculate_safety_stock(std_daily, lead_time_days, service_level)
    rop = calculate_reorder_point(avg_daily, lead_time_days, safety_stock)
    total_cost_eoq = calculate_total_cost(annual_demand, eoq, ordering_cost, holding_cost_per_unit)

    # Compare with other order quantities
    orders_per_year = annual_demand / eoq if eoq > 0 else 0
    days_between_orders = 365 / orders_per_year if orders_per_year > 0 else 0
    avg_inventory = (eoq / 2) + safety_stock
    inventory_turnover = annual_demand / avg_inventory if avg_inventory > 0 else 0

    return {
        "avg_daily_demand": round(avg_daily, 2),
        "std_daily_demand": round(std_daily, 2),
        "annual_demand": round(annual_demand, 0),
        "eoq": eoq,
        "safety_stock": safety_stock,
        "reorder_point": rop,
        "total_annual_cost": total_cost_eoq,
        "orders_per_year": round(orders_per_year, 1),
        "days_between_orders": round(days_between_orders, 1),
        "avg_inventory_level": round(avg_inventory, 0),
        "inventory_turnover": round(inventory_turnover, 1),
        "service_level": service_level,
        "lead_time_days": lead_time_days,
        "unit_cost": unit_cost,
        "ordering_cost": ordering_cost,
        "holding_cost_pct": holding_cost_pct,
    }


def what_if_analysis(
    daily_sales_series,
    lead_times=[3, 5, 7, 10, 14],
    service_levels=[0.90, 0.95, 0.99],
    ordering_cost=50,
    holding_cost_pct=0.25,
    unit_cost=10,
):
    """
    Run inventory optimization across multiple scenarios.
    Returns a dataframe comparing different lead times and service levels.
    """
    rows = []
    for lt in lead_times:
        for sl in service_levels:
            result = full_inventory_analysis(
                daily_sales_series,
                ordering_cost=ordering_cost,
                holding_cost_pct=holding_cost_pct,
                unit_cost=unit_cost,
                lead_time_days=lt,
                service_level=sl,
            )
            rows.append({
                "lead_time": lt,
                "service_level": f"{sl:.0%}",
                "eoq": result["eoq"],
                "safety_stock": result["safety_stock"],
                "reorder_point": result["reorder_point"],
                "avg_inventory": result["avg_inventory_level"],
                "total_cost": result["total_annual_cost"],
                "turnover": result["inventory_turnover"],
            })

    return pd.DataFrame(rows)


def cost_curve_data(annual_demand, ordering_cost, holding_cost_per_unit, eoq, points=50):
    """
    Generate data for total cost curve visualization.
    Shows how cost changes with different order quantities.
    """
    if eoq <= 0:
        return pd.DataFrame()

    q_range = np.linspace(max(1, eoq * 0.2), eoq * 3, points)
    rows = []
    for q in q_range:
        ordering = (annual_demand / q) * ordering_cost
        holding = (q / 2) * holding_cost_per_unit
        total = ordering + holding
        rows.append({
            "order_qty": round(q, 0),
            "ordering_cost": round(ordering, 2),
            "holding_cost": round(holding, 2),
            "total_cost": round(total, 2),
        })

    df = pd.DataFrame(rows)
    return df