"""
Workforce Digital Twin - SaaS-Grade API Entry Point
==================================================

Business Purpose:
Provides a secure and high-performance API for organizational workforce simulation.
Enables strategic "What-If" analysis for talent turnover and project delivery risks.

Technical Philosophy:
- Lean Routing: Minimal business logic in entry point.
- Service Orchestration: Delegate complex logic to core/simulation/service and core/decision/service.
- Global State: Thread-safe dictionary-based state management (Enterprise Grade).
- Observability: Structured JSON logging for all requests and application lifecycle events.

API Reference:
- POST /simulate: Run a specific workforce scenario.
- POST /decision/compare: Matrix-based comparison of multiple HR strategies.
- POST /decision/explain: LLM-driven interpretation of the optimal HR strategy.
"""

import os
import math
import json
import logging
import time
from dotenv import load_dotenv

# Load environment variables before any other imports that might depend on them
load_dotenv()

from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

# Core Models
from core.models import (
    Employee, 
    ExecutionProject, 
    StrategyType,
    SimulationEngineError,
    DecisionEngineError,
    LLMEngineError
)

# Service Layer Imports
from core.simulation.service import run_standard_simulation
from core.decision.service import compare_hr_strategies, explain_strategic_decision

# Utility Imports
from core.logging_config import logger
from core.persistence.storage import log_simulation_history

# =====================================================
# State & Configuration
# =====================================================

STATE = {
    "employees": [],
    "projects": [],
    "model_config": {}
}

# =====================================================
# Request/Response Schemas
# =====================================================

class SimulationRequest(BaseModel):
    employee_id: Optional[str] = None
    strategy: StrategyType = StrategyType.BASELINE
    seed: Optional[int] = 42
    shock_test: bool = False
    # Expansion parameters
    price_delta: Optional[float] = 0.1
    churn_impact: Optional[float] = 0.2
    restructure_intensity: Optional[float] = 0.5

class DecisionRequest(BaseModel):
    employee_id: str
    seed: Optional[int] = 42

class TrajectoryRequest(BaseModel):
    months: int = 12
    attrition_multiplier: float = 1.0
    seed: Optional[int] = 42

class HiringRequest(BaseModel):
    target_roles: List[str]
    hiring_delay_weeks: int = 4
    ramp_up_curve: float = 0.5 # 0.5 = 50% productivity at start

class PortfolioRequest(BaseModel):
    project_ids: List[str]
    priorities: Dict[str, int] # id -> priority (1-5)
    seed: Optional[int] = 42

class BudgetRequest(BaseModel):
    adjustment_percent: float
    hiring_freeze: bool = False
    training_investment: float = 0.0 # 0-1
    seed: Optional[int] = 42

class InvestmentRequest(BaseModel):
    type: str # training, automation, redundancy
    amount: float
    seed: Optional[int] = 42

class MergeRequest(BaseModel):
    other_org_data: Dict[str, Any]
    merge_type: str # aggressive, gradual, selective
    seed: Optional[int] = 42

# =====================================================
# App Initialization & Governance
# =====================================================

app = FastAPI(
    title="Workforce Digital Twin API",
    version="3.0",
    description="Deterministic Enterprise Workforce Simulation Engine"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_system_integrity():
    """Validates that the system state is consistent and logically sound."""
    try:
        assert isinstance(STATE["employees"], list), "STATE['employees'] must be a list"
        assert isinstance(STATE["projects"], list), "STATE['projects'] must be a list"
        assert isinstance(STATE["model_config"], dict), "STATE['model_config'] must be a dict"
        logger.info("system_integrity_check_passed")
    except Exception as e:
        logger.error("system_integrity_failure", extra={"error": str(e)})

def load_system_state():
    """Initializes the simulation engine with environmental data."""
    # 1. Load Safe Defaults
    STATE["model_config"] = {
        "model_version": "3.0",
        "decision_weights": {
            "profit": 0.4,
            "org_health": 0.2,
            "fragility": 0.2,
            "stability": 0.2
        },
        "noise_parameters": {
            "task_duration_noise": 0.1,
            "revenue_noise": 0.05,
            "attrition_shock": 0.1
        }
    }

    # 2. Load Employees
    try:
        data_path = "data/company.json"
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                raw_data = json.load(f)
                parsed_employees = []
                for e_data in raw_data.get("employees", []):
                    try:
                        parsed_employees.append(Employee(**e_data))
                    except Exception as e:
                        logger.error("employee_parse_error", extra={"employee_data": e_data, "error": str(e)})
                STATE["employees"] = parsed_employees
        else:
            logger.warning("data_file_missing", extra={"path": data_path})
    except Exception as e:
        logger.error("employee_load_failure", extra={"error": str(e)})
        STATE["employees"] = []

    # 3. Load Model Config
    try:
        config_path = "data/model_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                STATE["model_config"].update(loaded_config)
        else:
            logger.warning("config_file_missing", extra={"path": config_path})
    except Exception as e:
        logger.error("config_load_failure", extra={"error": str(e)})

    # 4. Mocking projects for this version - In SaaS, these would come from DB
    try:
        from core.models import ExecutionTask
        STATE["projects"] = [
            ExecutionProject(
                id="PROJ_001",
                name="AI Revenue Optimization",
                base_revenue=500000,
                decay_rate=0.03,
                tasks=[
                    ExecutionTask(id="T1", name="Data Ingestion", required_skill="backend", required_level=0.8, estimated_hours=120),
                    ExecutionTask(id="T2", name="Model Architecture", required_skill="data", required_level=0.9, estimated_hours=80, dependencies=["T1"]),
                    ExecutionTask(id="T3", name="Frontend Dashboard", required_skill="frontend", required_level=0.7, estimated_hours=160, dependencies=["T1"]),
                ]
            )
        ]

    except Exception as e:
        logger.error("project_initialization_failure", extra={"error": str(e)})
        STATE["projects"] = []
    
    logger.info("system_state_initialized", extra={
        "employee_count": len(STATE["employees"]),
        "project_count": len(STATE["projects"]),
        "model_version": STATE["model_config"].get("model_version")
    })

@app.on_event("startup")
def startup_event():
    load_system_state()
    validate_system_integrity()

# =====================================================
# API Endpoints (Routing Layer)
# =====================================================

@app.get("/")
def health_check():
    return {
        "status": "OPERATIONAL",
        "model_version": STATE["model_config"].get("model_version", "3.0"),
        "employee_count": len(STATE["employees"]),
        "project_count": len(STATE["projects"]),
        "config_loaded": bool(STATE["model_config"]),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/org/state")
def get_org_state():
    """
    Returns the organization structure formatted for ReactFlow.
    """
    nodes = []
    edges = []
    
    for i, emp in enumerate(STATE["employees"]):
        # Basic grid layout for nodes (can be improved later)
        nodes.append({
            "id": emp.id,
            "type": "default",
            "data": { "label": f"{emp.name}\n({emp.role})" },
            "position": {
                "x": 200 + (i % 5) * 250,
                "y": 100 + (i // 5) * 150
            }
        })
        
        if emp.reports_to:
            edges.append({
                "id": f"e-{emp.reports_to}-{emp.id}",
                "source": emp.reports_to,
                "target": emp.id,
                "animated": True
            })
            
    if not nodes and not edges:
        return {"nodes": [], "edges": []}
            
    return {"nodes": nodes, "edges": edges}

def validate_employee_exists(employee_id: Optional[str]):
    if employee_id and not any(e.id == employee_id for e in STATE["employees"]):
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")

@app.post("/simulate")
def simulate_scenario(request: SimulationRequest):
    """
    Executes a high-fidelity workforce simulation.
    """
    start_time = time.time()
    try:
        validate_employee_exists(request.employee_id)
        result = run_standard_simulation(
            projects=STATE["projects"],
            employees=STATE["employees"],
            strategy_key=request.strategy,
            target_employee_id=request.employee_id,
            shock_mode=request.shock_test,
            random_seed=request.seed,
            model_config=STATE["model_config"]
        )
        if result is None:
            raise SimulationEngineError("Simulation returned None unexpectedly.")
        
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info("simulation_complete", extra={
            "endpoint": "/simulate",
            "strategy": request.strategy,
            "seed": request.seed,
            "duration_ms": duration_ms
        })
        return result
    except HTTPException:
        raise
    except SimulationEngineError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("simulation_engine_error", extra={
            "endpoint": "/simulate", 
            "error": str(e),
            "duration_ms": duration_ms
        })
        raise HTTPException(status_code=400, detail={"error": "Simulation Engine Error", "message": str(e)})
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("api_unexpected_failure", extra={
            "endpoint": "/simulate", 
            "error": str(e),
            "duration_ms": duration_ms
        })
        raise HTTPException(status_code=500, detail={"error": "Unexpected Engine Failure"})

@app.post("/decision/compare")
def compare_scenarios(request: DecisionRequest):
    """
    Compares multiple HR strategies for a specific employee departure.
    """
    start_time = time.time()
    try:
        validate_employee_exists(request.employee_id)
        comparison = compare_hr_strategies(
            projects=STATE["projects"],
            employees=STATE["employees"],
            target_employee_id=request.employee_id,
            random_seed=request.seed,
            model_config=STATE["model_config"]
        )
        if comparison is None:
            raise DecisionEngineError("Decision engine returned None unexpectedly.")
        
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info("decision_comparison_complete", extra={
            "endpoint": "/decision/compare",
            "seed": request.seed,
            "duration_ms": duration_ms
        })
        return comparison
    except HTTPException:
        raise
    except (SimulationEngineError, DecisionEngineError) as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("decision_engine_error", extra={
            "endpoint": "/decision/compare", 
            "error": str(e),
            "duration_ms": duration_ms
        })
        raise HTTPException(status_code=400, detail={"error": "Decision Engine Error", "message": str(e)})
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("api_unexpected_failure", extra={
            "endpoint": "/decision/compare", 
            "error": str(e),
            "duration_ms": duration_ms
        })
        raise HTTPException(status_code=500, detail={"error": "Unexpected Engine Failure"})

@app.post("/decision/explain")
def explain_decision(request: DecisionRequest):
    """
    Provides an LLM-driven interpretation of the best strategic choice.
    """
    start_time = time.time()
    try:
        validate_employee_exists(request.employee_id)
        explanation = explain_strategic_decision(
            projects=STATE["projects"],
            employees=STATE["employees"],
            target_employee_id=request.employee_id,
            random_seed=request.seed,
            model_config=STATE["model_config"]
        )
        if explanation is None:
            raise LLMEngineError("Explanation engine returned None unexpectedly.")
        
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info("decision_explanation_complete", extra={
            "endpoint": "/decision/explain",
            "seed": request.seed,
            "duration_ms": duration_ms
        })
        return explanation
    except HTTPException:
        raise
    except (SimulationEngineError, DecisionEngineError, LLMEngineError) as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("explanation_engine_error", extra={
            "endpoint": "/decision/explain", 
            "error": str(e),
            "duration_ms": duration_ms
        })
        raise HTTPException(status_code=400, detail={"error": "Explanation Engine Error", "message": str(e)})
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("api_unexpected_failure", extra={
            "endpoint": "/decision/explain", 
            "error": str(e),
            "duration_ms": duration_ms
        })
        raise HTTPException(status_code=500, detail={"error": "Unexpected Engine Failure"})

# =====================================================
# Phase 6: Predictive Intelligence
# =====================================================

@app.get("/predict/attrition")
def predict_attrition(seed: int = 42):
    """
    Deterministic attrition probability modeling using sigmoid transform.

    Fix 3: Replaced linear formula (narrow 0.41-0.56 band) with the same
    sigmoid model used in behavioral.py:
        z = 1.8*burnout + 1.2*centrality + 0.9*scarcity - 1.5
        P_exit = sigmoid(z)
    """
    import random
    rng = random.Random(seed)

    results = []
    for emp in STATE["employees"]:
        # Simulate per-employee burnout/centrality/scarcity with seeded noise
        utilization = 0.7 + (rng.random() * 0.3)   # [0.70, 1.00]
        burnout      = min(1.0, utilization + (rng.random() * 0.2 - 0.1))  # contagion-shifted
        centrality   = rng.random() * 0.4            # [0.0, 0.4] – sparse in most orgs
        scarcity     = 0.3 + rng.random() * 0.5      # [0.3, 0.8] – 1/redundancy proxy

        # Sigmoid transform (same calibration as behavioral.py)
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

@app.post("/predict/hiring-impact")
def predict_hiring_impact(request: HiringRequest):
    """
    Simulates productivity and revenue recovery for new hires.

    Fix 4: Replaced linear ramp with a logistic S-curve for realistic onboarding:
        productivity(t) = 1 / (1 + e^(-k * (t - midpoint)))
        k = 0.8, midpoint = hiring_delay_weeks / 2
    """
    _k = 0.8
    ramp_up_weeks = max(request.hiring_delay_weeks, 1)  # use delay as ramp duration
    midpoint = ramp_up_weeks / 2.0

    timeline = []
    for week in range(1, 25):  # 6-month (24-week) forecast
        current_productivity = 0.0
        if week > request.hiring_delay_weeks:
            weeks_active = week - request.hiring_delay_weeks
            # Logistic onboarding S-curve
            ramp_factor = 1.0 / (1.0 + math.exp(-_k * (weeks_active - midpoint)))
            current_productivity = ramp_factor * len(request.target_roles)

        timeline.append({
            "week": week,
            "productivity": round(current_productivity, 3),
            "revenue_recovery": round(current_productivity * 15000, 2),
        })

    return {
        "roles": request.target_roles,
        "timeline": timeline,
        "total_recovery_projection": round(sum(t["revenue_recovery"] for t in timeline), 2),
    }

@app.post("/predict/trajectory")
def predict_trajectory(request: TrajectoryRequest):
    """
    6-12 month risk trajectory forecast with sensitivity analysis.
    """
    import random
    months = []
    
    # Starting state
    current_revenue = 500000.0
    current_stability = 0.85
    current_fragility = 0.35
    
    for m in range(1, request.months + 1):
        # Derive sub-seed per month
        m_rng = random.Random(request.seed + m)
        
        # Apply attrition impact based on multiplier
        attrition_event = m_rng.random() < (0.05 * request.attrition_multiplier)
        if attrition_event:
            current_revenue *= 0.92
            current_stability *= 0.95
            current_fragility *= 1.1
        else:
            # Slight recovery/drift
            current_revenue *= 1.005 
            current_stability = min(0.95, current_stability * 1.002)
            current_fragility *= 0.99
            
        months.append({
            "month": m,
            "revenue": round(current_revenue, 2),
            "stability": round(current_stability, 3),
            "fragility": round(current_fragility, 3)
        })
        
    return {"months": months}

# =====================================================
# Phase 8: Decision OS (Expansion)
# =====================================================

@app.post("/portfolio/simulate")
def simulate_portfolio(request: PortfolioRequest):
    """
    Simulates multi-project interactions, resource contention, and scheduling conflicts.

    Fix 8: Added conflict-density model with queue-delay probability and cross-project
    scheduling conflict signal.

        D = project_count * 0.15          (base conflict density)
        P_delay = min(1.0, D * 0.8)       (probability any given week sees a delay)
        avg_delay_weeks = D * 2.5         (expected queue-delay overhead)

    Per-week density incorporates a sinusoidal sprint-cycle variance to model
    realistic resource contention peaks.
    """
    import random as _random
    rng = _random.Random(request.seed or 42)

    n_projects = max(1, len(request.project_ids))
    conflict_density = n_projects * 0.15          # D
    p_delay          = min(1.0, conflict_density * 0.8)
    avg_delay_weeks  = round(conflict_density * 2.5, 2)

    # Revenue estimate: each project contributes, but contention reduces total yield
    contention_penalty = 1.0 - (conflict_density * 0.1)
    portfolio_revenue  = round(500_000.0 * n_projects * max(0.5, contention_penalty), 2)

    # Per-week conflict heatmap with sprint-cycle variance
    heatmap = []
    for w in range(1, 13):
        # Sprint peaks every 4 weeks; add stochastic noise per-week
        sprint_cycle  = 1.0 + 0.25 * math.sin(math.pi * w / 2)
        weekly_noise  = 1.0 + (rng.random() - 0.5) * 0.1
        density_w     = min(1.0, conflict_density * sprint_cycle * weekly_noise)
        delay_prob_w  = min(1.0, p_delay * sprint_cycle)
        heatmap.append({
            "week":         w,
            "density":      round(density_w, 3),
            "delay_prob":   round(delay_prob_w, 3),
        })

    return {
        "portfolio_revenue":  portfolio_revenue,
        "delay_exposure":     round(conflict_density * 10, 2),
        "avg_queue_delay_weeks": avg_delay_weeks,
        "delay_probability": round(p_delay, 3),
        "conflict_heatmap":  heatmap,
        "projects_evaluated": n_projects,
    }

@app.post("/budget/allocate")
def allocate_budget(request: BudgetRequest):
    """
    Simulates financial and risk impact of budget shifts.

    Fix 5: Replaced linear scaling (delta * 1.5) with log-based diminishing returns:
        rev_change = log(1 + |ratio|) * 150, capped at ±30%
    This prevents unrealistic jumps (10% budget → 15% revenue).

    Fix 6 (stability): Clamped training stability gain to [0, 0.08].
    """
    investment_ratio = request.adjustment_percent / 100.0
    sign = 1.0 if investment_ratio >= 0 else -1.0
    # Diminishing returns: log(1 + |ratio|) * 50
    # At 10% input: log(1.1) * 50 ≈ 4.76%  (realistic, not 15%)
    # At 50% input: log(1.5) * 50 ≈ 20.3%  (still capped at 30%)
    rev_change = sign * math.log(1.0 + abs(investment_ratio)) * 50.0
    # Cap within realistic bounds: ±30%
    rev_change = max(-30.0, min(30.0, rev_change))

    # Training stability gain with realistic cap
    stab_change = min(0.08, request.training_investment * 0.08)

    if request.hiring_freeze:
        rev_change = max(-30.0, rev_change - 3.0)
        stab_change = max(0.0, stab_change - 0.02)

    base_run_rate = 500000.0
    new_run_rate = base_run_rate * (1.0 + rev_change / 100.0)

    return {
        "revenue_delta_percent": round(rev_change, 2),
        "stability_delta":       round(stab_change, 3),
        "attrition_delta":       round(-stab_change * 0.5, 3),
        "financial_summary": {
            "new_run_rate":    round(max(0.0, new_run_rate), 2),
            "efficiency_gain": round(request.training_investment * 8.0, 2),
        },
    }

@app.post("/strategy/investment")
def strategic_investment(request: InvestmentRequest):
    """
    Models impact of training, automation, or redundancy investments.

    Fix 6: Clamped fragility_delta ∈ [-0.15, -0.02] and stability_delta ∈ [0.01, 0.08].
    Raw investment effects used log-scale diminishing returns before clamping.
    """
    # Per-type base rates (fraction per $100k invested)
    impact = {
        "training":    {"fragility_reduction": 0.06, "stability_gain": 0.04},
        "automation":  {"fragility_reduction": 0.04, "stability_gain": 0.02},
        "redundancy":  {"fragility_reduction": 0.05, "stability_gain": 0.05},
    }

    selected = impact.get(request.type, {"fragility_reduction": 0.03, "stability_gain": 0.02})

    # Diminishing returns: log-scale investment effect
    investment_units = math.log(1.0 + max(0.0, request.amount) / 100_000.0)

    raw_fragility_delta = -selected["fragility_reduction"] * investment_units
    raw_stability_delta  =  selected["stability_gain"]     * investment_units

    # Clamp to specified bounds (Fix 6)
    fragility_delta = max(-0.15, min(-0.02, raw_fragility_delta))
    stability_delta  = max(0.01,  min(0.08,   raw_stability_delta))

    return {
        "type": request.type,
        "investment_amount": request.amount,
        "metrics": {
            "fragility_delta": round(fragility_delta, 4),
            "stability_delta":  round(stability_delta,  4),
        },
    }

@app.post("/strategy/merge")
def strategy_merge(request: MergeRequest):
    """
    M&A scenario modeling for organizational merging.

    Fix 7: Validates `other_org_data.employees`.
      - Accepts a list  → counts the items directly.
      - Accepts an int  → treated as employee count (converted to synthetic placeholders).
      - Missing key     → defaults to 0.
    """
    emp_field = request.other_org_data.get("employees", 0)

    if isinstance(emp_field, int):
        # Integer count → synthetic employee count
        other_emp_count = max(0, emp_field)
    elif isinstance(emp_field, list):
        other_emp_count = len(emp_field)
    else:
        # Unexpected type – default safe
        other_emp_count = 0

    base_emp_count = len(STATE["employees"])
    total_emp_count = base_emp_count + other_emp_count

    # Friction coefficients per merge strategy
    friction_map = {"aggressive": 0.25, "gradual": 0.10, "selective": 0.15}
    friction = friction_map.get(request.merge_type, 0.15)

    # Synergy grows sub-linearly with headcount (diminishing returns)
    synergy = math.log(1.0 + total_emp_count) * 0.05

    combined_fragility = min(1.0, 0.35 + friction)
    stability_score    = max(0.0, 0.75 - friction)
    revenue_projection = 1_000_000.0 + (synergy * 200_000.0)
    overlap_count      = int(total_emp_count * 0.12)

    return {
        "combined_fragility":  round(combined_fragility, 2),
        "revenue_projection":  round(revenue_projection, 2),
        "stability_score":     round(stability_score, 2),
        "overlap_count":       overlap_count,
        "other_employee_count": other_emp_count,
        "merge_type":          request.merge_type,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)