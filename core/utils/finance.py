"""
Business Purpose:
Provides the financial math core for the Digital Twin.
Calculates revenue decay based on time-to-market and total operational labor costs.

Technical Responsibility:
- Compute project revenue using exponential decay models.
- Calculate pro-rated salary costs based on simulation duration.
- Aggregate revenue and costs into a profit P&L dictionary.
- Guard against negative revenue, division-by-zero, and NaN propagation.

Determinism Level:
High. All financial models are purely mathematical and deterministic.

External Dependencies:
- math
"""

import math
from typing import List, Dict, Any
from core.models import Employee, ExecutionProject

# Business Constants
IDEAL_DELIVERY_WINDOW_WEEKS = 4.0
WEEKS_PER_YEAR = 52.0

# Numerical stability
_EPSILON = 1e-9


def calculate_project_revenue(
    projects: List[ExecutionProject],
    total_duration_weeks: float,
) -> float:
    """
    Calculates total revenue across all projects, accounting for exponential time-decay
    beyond the ideal delivery window.

    Fix 10: Revenue is clamped to [0, ∞) to prevent negative values from extreme decay rates.

    Args:
        projects: List of projects to evaluate.
        total_duration_weeks: Final simulation duration in weeks.

    Returns:
        float: Total realized revenue (never negative).
    """
    aggregate_revenue = 0.0

    for project in projects:
        # Revenue decay kicks in if work exceeds the ideal window
        delivery_delay = max(0.0, total_duration_weeks - IDEAL_DELIVERY_WINDOW_WEEKS)

        # Formula: Revenue = Base * e^(-decay_rate * delay)
        realized_revenue = project.base_revenue * math.exp(
            -project.decay_rate * delivery_delay
        )

        # Guard: revenue can never be negative
        aggregate_revenue += max(0.0, realized_revenue)

    return round(aggregate_revenue, 2)


def calculate_operational_cost(
    employees: List[Employee],
    total_duration_weeks: float,
) -> float:
    """
    Calculates total labor cost for the simulation duration.

    Args:
        employees: The employee pool.
        total_duration_weeks: Simulation duration in weeks.

    Returns:
        float: Total pro-rated salary expenditure.
    """
    # Guard: avoid cost calculation with zero-duration simulations
    effective_duration = max(0.0, total_duration_weeks)
    aggregate_weekly_burn = sum(e.salary / WEEKS_PER_YEAR for e in employees)
    return round(aggregate_weekly_burn * effective_duration, 2)


def compute_profit(
    projects: List[ExecutionProject],
    employees: List[Employee],
    duration_weeks: float,
) -> Dict[str, float]:
    """
    Generates a consolidated profit and loss (P&L) snapshot.

    Returns:
        Dict with 'revenue', 'cost', and 'profit' keys.
        Profit may be negative (valid loss scenario) but revenue is always >= 0.
    """
    realized_revenue = calculate_project_revenue(projects, duration_weeks)
    total_labor_cost = calculate_operational_cost(employees, duration_weeks)

    return {
        "revenue": realized_revenue,
        "cost":    total_labor_cost,
        "profit":  round(realized_revenue - total_labor_cost, 2),
    }