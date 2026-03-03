"""
Business Purpose:
Provides the financial math core for the Digital Twin.
Calculates revenue decay based on time-to-market and total operational labor costs.

Technical Responsibility:
- Compute project revenue using exponential decay models.
- Calculate pro-rated salary costs based on simulation duration.
- Aggregate revenue and costs into a profit P&L dictionary.

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

def calculate_project_revenue(
    projects: List[ExecutionProject], 
    total_duration_weeks: float
) -> float:
    """
    Calculates total revenue across all projects, accounting for exponential time-decay
    beyond the ideal delivery window.
    
    Args:
        projects: List of projects to evaluate.
        total_duration_weeks: Final simulation duration in weeks.
        
    Returns:
        float: Total realized revenue.
    """
    aggregate_revenue = 0.0

    for project in projects:
        # Revenue decay kicks in if work exceeds the ideal window
        delivery_delay = max(0.0, total_duration_weeks - IDEAL_DELIVERY_WINDOW_WEEKS)

        # Formula: Revenue = Base * e^(-decay_rate * delay)
        realized_revenue = project.base_revenue * math.exp(
            -project.decay_rate * delivery_delay
        )

        aggregate_revenue += realized_revenue

    return round(aggregate_revenue, 2)


def calculate_operational_cost(
    employees: List[Employee], 
    total_duration_weeks: float
) -> float:
    """
    Calculates total labor cost for the simulation duration.
    
    Args:
        employees: The employee pool.
        total_duration_weeks: Simulation duration in weeks.
        
    Returns:
        float: Total pro-rated salary expenditure.
    """
    aggregate_weekly_burn = sum(e.salary / WEEKS_PER_YEAR for e in employees)
    return round(aggregate_weekly_burn * total_duration_weeks, 2)


def compute_profit(
    projects: List[ExecutionProject], 
    employees: List[Employee], 
    duration_weeks: float
) -> Dict[str, float]:
    """
    Generates a consolidated profit and loss (P&L) snapshot.
    """
    realized_revenue = calculate_project_revenue(projects, duration_weeks)
    total_labor_cost = calculate_operational_cost(employees, duration_weeks)
    
    return {
        "revenue": realized_revenue,
        "cost": total_labor_cost,
        "profit": round(realized_revenue - total_labor_cost, 2)
    }