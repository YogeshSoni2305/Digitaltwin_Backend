"""
Business Purpose:
  Predictive Intelligence service — provides org-level risk forecasts,
  hiring impact simulations, portfolio conflict analysis, and M&A modeling.

Technical Responsibility:
  All functions were extracted verbatim from main.py (Phase 3 refactoring).
  No business logic has changed — only the location has moved from the
  routing layer to this dedicated service module.

Why extracted:
  - Business logic in the API layer (main.py) cannot be unit-tested without
    starting a full HTTP server.
  - Functions in this module can be imported and called directly in tests
    or by other services without going through FastAPI.

Determinism Level:
  High for all functions when called with the same seed.
  Trajectory and portfolio use seeded RNG for reproducibility.

External Dependencies:
  - math (stdlib)
  - random (stdlib)
  - core.state.AppContext
"""

import math
import random
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.state import AppContext


# =========================================================================
# /predict/attrition
# =========================================================================

def predict_attrition_risk(ctx: "AppContext", seed: int = 42) -> Dict[str, Any]:
    """
    Deterministic attrition probability per employee using the sigmoid model
    from behavioral.py:  z = 1.8*burnout + 1.2*centrality + 0.9*scarcity - 1.5

    Args:
        ctx: Application context holding the current employee roster.
        seed: RNG seed for reproducible per-employee noise.

    Returns:
        {"employees": [...sorted by probability descending...]}
    """
    rng = random.Random(seed)
    results = []

    for emp in ctx.employees:
        utilization = 0.7 + (rng.random() * 0.3)          # [0.70, 1.00]
        burnout      = min(1.0, utilization + (rng.random() * 0.2 - 0.1))
        centrality   = rng.random() * 0.4                  # [0.0, 0.4]
        scarcity     = 0.3 + rng.random() * 0.5            # [0.3, 0.8]

        z    = 1.8 * burnout + 1.2 * centrality + 0.9 * scarcity - 1.5
        prob = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
        prob = round(max(0.0, min(1.0, prob)), 2)

        band = "LOW"
        if prob > 0.70:   band = "CRITICAL"
        elif prob > 0.50: band = "HIGH"
        elif prob > 0.30: band = "MEDIUM"

        results.append({
            "employee_id": emp.id,
            "name":        emp.name,
            "probability": prob,
            "risk_band":   band,
        })

    return {"employees": sorted(results, key=lambda x: x["probability"], reverse=True)}


# =========================================================================
# /predict/hiring-impact
# =========================================================================

def predict_hiring_impact(request: Any) -> Dict[str, Any]:
    """
    24-week productivity and revenue recovery forecast using the logistic S-curve
    from scheduler.py.

    Args:
        request: HiringRequest Pydantic model.

    Returns:
        {roles, timeline, total_recovery_projection}
    """
    _k = 0.8
    ramp_up_weeks = max(request.hiring_delay_weeks, 1)
    midpoint      = ramp_up_weeks / 2.0

    timeline = []
    for week in range(1, 25):
        current_productivity = 0.0
        if week > request.hiring_delay_weeks:
            weeks_active         = week - request.hiring_delay_weeks
            ramp_factor          = 1.0 / (1.0 + math.exp(-_k * (weeks_active - midpoint)))
            current_productivity = ramp_factor * len(request.target_roles)

        timeline.append({
            "week":             week,
            "productivity":     round(current_productivity, 3),
            "revenue_recovery": round(current_productivity * 15000, 2),
        })

    return {
        "roles":                      request.target_roles,
        "timeline":                   timeline,
        "total_recovery_projection":  round(sum(t["revenue_recovery"] for t in timeline), 2),
    }


# =========================================================================
# /predict/trajectory
# =========================================================================

def predict_trajectory(request: Any) -> Dict[str, Any]:
    """
    Month-by-month risk trajectory with seeded stochastic attrition events.

    Args:
        request: TrajectoryRequest (months, attrition_multiplier, seed).

    Returns:
        {"months": [...]}
    """
    months          = []
    current_revenue    = 500_000.0
    current_stability  = 0.85
    current_fragility  = 0.35

    for m in range(1, request.months + 1):
        m_rng           = random.Random(request.seed + m)
        attrition_event = m_rng.random() < (0.05 * request.attrition_multiplier)

        if attrition_event:
            current_revenue   *= 0.92
            current_stability *= 0.95
            current_fragility *= 1.1
        else:
            current_revenue   *= 1.005
            current_stability  = min(0.95, current_stability * 1.002)
            current_fragility *= 0.99

        months.append({
            "month":     m,
            "revenue":   round(current_revenue,   2),
            "stability": round(current_stability, 3),
            "fragility": round(current_fragility, 3),
        })

    return {"months": months}


# =========================================================================
# /portfolio/simulate
# =========================================================================

def simulate_portfolio(request: Any) -> Dict[str, Any]:
    """
    Multi-project conflict density model with sinusoidal sprint-cycle variance.

    Args:
        request: PortfolioRequest (project_ids, seed).

    Returns:
        {portfolio_revenue, delay_exposure, avg_queue_delay_weeks,
         delay_probability, conflict_heatmap, projects_evaluated}
    """
    rng = random.Random(request.seed or 42)

    n_projects       = max(1, len(request.project_ids))
    conflict_density = n_projects * 0.15
    p_delay          = min(1.0, conflict_density * 0.8)
    avg_delay_weeks  = round(conflict_density * 2.5, 2)

    contention_penalty = 1.0 - (conflict_density * 0.1)
    portfolio_revenue  = round(500_000.0 * n_projects * max(0.5, contention_penalty), 2)

    heatmap = []
    for w in range(1, 13):
        sprint_cycle  = 1.0 + 0.25 * math.sin(math.pi * w / 2)
        weekly_noise  = 1.0 + (rng.random() - 0.5) * 0.1
        density_w     = min(1.0, conflict_density * sprint_cycle * weekly_noise)
        delay_prob_w  = min(1.0, p_delay * sprint_cycle)
        heatmap.append({
            "week":       w,
            "density":    round(density_w,    3),
            "delay_prob": round(delay_prob_w, 3),
        })

    return {
        "portfolio_revenue":     portfolio_revenue,
        "delay_exposure":        round(conflict_density * 10, 2),
        "avg_queue_delay_weeks": avg_delay_weeks,
        "delay_probability":     round(p_delay, 3),
        "conflict_heatmap":      heatmap,
        "projects_evaluated":    n_projects,
    }


# =========================================================================
# /budget/allocate
# =========================================================================

def allocate_budget(request: Any) -> Dict[str, Any]:
    """
    Log-based diminishing returns model for budget allocation impact.
    At 10% increase: log(1.1)*50 ≈ 4.76% revenue change (realistic).

    Args:
        request: BudgetRequest (adjustment_percent, training_investment, hiring_freeze).

    Returns:
        {revenue_delta_percent, stability_delta, attrition_delta, financial_summary}
    """
    investment_ratio = request.adjustment_percent / 100.0
    sign             = 1.0 if investment_ratio >= 0 else -1.0
    rev_change       = sign * math.log(1.0 + abs(investment_ratio)) * 50.0
    rev_change       = max(-30.0, min(30.0, rev_change))

    stab_change = min(0.08, request.training_investment * 0.08)

    if request.hiring_freeze:
        rev_change  = max(-30.0, rev_change - 3.0)
        stab_change = max(0.0, stab_change - 0.02)

    base_run_rate = 500_000.0
    new_run_rate  = base_run_rate * (1.0 + rev_change / 100.0)

    return {
        "revenue_delta_percent": round(rev_change, 2),
        "stability_delta":       round(stab_change, 3),
        "attrition_delta":       round(-stab_change * 0.5, 3),
        "financial_summary": {
            "new_run_rate":    round(max(0.0, new_run_rate), 2),
            "efficiency_gain": round(request.training_investment * 8.0, 2),
        },
    }


# =========================================================================
# /strategy/investment
# =========================================================================

def strategic_investment(request: Any) -> Dict[str, Any]:
    """
    Training / automation / redundancy investment impact model with log-scale
    diminishing returns and hard bounds on fragility and stability deltas.

    Args:
        request: InvestmentRequest (type, amount).

    Returns:
        {type, investment_amount, metrics: {fragility_delta, stability_delta}}
    """
    impact_rates = {
        "training":   {"fragility_reduction": 0.06, "stability_gain": 0.04},
        "automation": {"fragility_reduction": 0.04, "stability_gain": 0.02},
        "redundancy": {"fragility_reduction": 0.05, "stability_gain": 0.05},
    }
    selected = impact_rates.get(
        request.type,
        {"fragility_reduction": 0.03, "stability_gain": 0.02}
    )

    investment_units    = math.log(1.0 + max(0.0, request.amount) / 100_000.0)
    raw_fragility_delta = -selected["fragility_reduction"] * investment_units
    raw_stability_delta =  selected["stability_gain"]      * investment_units

    fragility_delta = max(-0.15, min(-0.02, raw_fragility_delta))
    stability_delta = max(0.01,  min(0.08,  raw_stability_delta))

    return {
        "type":              request.type,
        "investment_amount": request.amount,
        "metrics": {
            "fragility_delta": round(fragility_delta, 4),
            "stability_delta": round(stability_delta,  4),
        },
    }


# =========================================================================
# /strategy/merge
# =========================================================================

def strategy_merge(request: Any, ctx: "AppContext") -> Dict[str, Any]:
    """
    M&A scenario model: combines two org headcounts, applies friction and
    synergy, estimates redundancy overlap.

    Args:
        request: MergeRequest (other_org_data, merge_type).
        ctx: AppContext providing current employee count.

    Returns:
        {combined_fragility, revenue_projection, stability_score,
         overlap_count, other_employee_count, merge_type}
    """
    emp_field = request.other_org_data.get("employees", 0)

    if isinstance(emp_field, int):
        other_emp_count = max(0, emp_field)
    elif isinstance(emp_field, list):
        other_emp_count = len(emp_field)
    else:
        other_emp_count = 0

    base_emp_count  = ctx.employee_count()
    total_emp_count = base_emp_count + other_emp_count

    friction_map = {"aggressive": 0.25, "gradual": 0.10, "selective": 0.15}
    friction     = friction_map.get(request.merge_type, 0.15)

    synergy            = math.log(1.0 + total_emp_count) * 0.05
    combined_fragility = min(1.0, 0.35 + friction)
    stability_score    = max(0.0, 0.75 - friction)
    revenue_projection = 1_000_000.0 + (synergy * 200_000.0)
    overlap_count      = int(total_emp_count * 0.12)

    return {
        "combined_fragility":  round(combined_fragility, 2),
        "revenue_projection":  round(revenue_projection, 2),
        "stability_score":     round(stability_score,    2),
        "overlap_count":       overlap_count,
        "other_employee_count": other_emp_count,
        "merge_type":          request.merge_type,
    }
