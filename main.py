"""
Workforce Digital Twin - SaaS-Grade API Entry Point
==================================================

Business Purpose:
Provides a secure and high-performance API for organizational workforce simulation.
Enables strategic "What-If" analysis for talent turnover and project delivery risks.

Technical Philosophy:
- Lean Routing: Minimal business logic in entry point.
- Service Orchestration: Delegate complex logic to service layer modules.
- Typed Context: Typed AppContext singleton replaces global STATE dict.
- Observability: Structured JSON logging for all requests and lifecycle events.

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
import uuid
from contextvars import ContextVar
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables before any other imports that might depend on them
load_dotenv()

# Internal Imports
from core.settings import settings
from core.context import REQUEST_ID

from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from pydantic import BaseModel

# Core Models
from core.models import (
    Employee,
    ExecutionProject,
    StrategyType,
    SimulationEngineError,
    DecisionEngineError,
    LLMEngineError,
    ServiceResponse,
    DecisionResult,
    ExplanationResult
)

# Service Layer Imports
from core.simulation.service import run_standard_simulation
from core.decision.service import compare_hr_strategies, explain_strategic_decision

# Prediction Service (Phase 3: extracted from inline endpoint bodies)
from core.prediction.service import (
    predict_attrition_risk,
    predict_hiring_impact      as _predict_hiring_impact,
    predict_trajectory         as _predict_trajectory,
    simulate_portfolio         as _simulate_portfolio,
    allocate_budget            as _allocate_budget,
    strategic_investment       as _strategic_investment,
    strategy_merge             as _strategy_merge,
)

# Typed Application Context (replaces global STATE dict)
from core.state import app_ctx

# Utility Imports
from core.logging_config import logger
from core.persistence.storage import log_simulation_history

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
# Request Context & Middleware
# =====================================================

@app.middleware("http")
async def add_request_id(request, call_next):
    """Injects a unique request ID into the context for traceability."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = REQUEST_ID.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        REQUEST_ID.reset(token)

# =====================================================
# App Initialization & Governance
# =====================================================

app = FastAPI(
    title="Workforce Digital Twin API",
    version="3.0",
    description="Deterministic Enterprise Workforce Simulation Engine",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://workforce-twin.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_system_integrity():
    """Validates that the application context has loaded correctly."""
    try:
        assert isinstance(app_ctx.employees, list),    "app_ctx.employees must be a list"
        assert isinstance(app_ctx.projects, list),     "app_ctx.projects must be a list"
        assert isinstance(app_ctx.model_config, dict), "app_ctx.model_config must be a dict"
        logger.info("system_integrity_check_passed")
    except Exception as e:
        logger.error("system_integrity_failure", extra={"error": str(e)})

def load_system_state():
    """Initializes the simulation engine with environmental data."""
    # 1. Load Safe Defaults
    app_ctx.model_config = {
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
                app_ctx.employees = parsed_employees
                app_ctx.employee_map = {e.id: e for e in parsed_employees}
                
                # PRODUCTION BOUNDARY: Check max employees
                if len(app_ctx.employees) > settings.MAX_EMPLOYEES:
                    logger.error("max_employees_exceeded", extra={"limit": settings.MAX_EMPLOYEES})
                    raise ValueError(f"Too many employees loaded. Limit is {settings.MAX_EMPLOYEES}.")
                    
        else:
            logger.warning("data_file_missing", extra={"path": data_path})
    except Exception as e:
        logger.error("employee_load_failure", extra={"error": str(e)})
        app_ctx.employees = []

    # 3. Load Model Config (file overrides defaults if present)
    try:
        config_path = "data/model_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                app_ctx.model_config.update(loaded_config)
        else:
            logger.warning("config_file_missing", extra={"path": config_path})
    except Exception as e:
        logger.error("config_load_failure", extra={"error": str(e)})

    # 4. Mock projects (SaaS: these would come from DB)
    try:
        from core.models import ExecutionTask
        app_ctx.projects = [
            ExecutionProject(
                id="PROJ_001",
                name="AI Revenue Optimization",
                base_revenue=500000,
                decay_rate=0.03,
                tasks=[
                    ExecutionTask(id="T1", name="Data Ingestion",      required_skill="backend",  required_level=0.8, estimated_hours=120),
                    ExecutionTask(id="T2", name="Model Architecture",  required_skill="data",     required_level=0.9, estimated_hours=80,  dependencies=["T1"]),
                    ExecutionTask(id="T3", name="Frontend Dashboard",  required_skill="frontend", required_level=0.7, estimated_hours=160, dependencies=["T1"]),
                ]
            )
        ]
        
        # PRODUCTION BOUNDARY: Check max tasks
        total_tasks = sum(len(p.tasks) for p in app_ctx.projects)
        if total_tasks > settings.MAX_TASKS_PER_PROJECT:
            logger.error("max_tasks_exceeded", extra={"limit": settings.MAX_TASKS_PER_PROJECT})
            raise ValueError(f"Too many tasks loaded. Limit is {settings.MAX_TASKS_PER_PROJECT}.")
            
    except Exception as e:
        logger.error("project_initialization_failure", extra={"error": str(e)})
        app_ctx.projects = []

    logger.info("system_state_initialized", extra={
        "employee_count": app_ctx.employee_count(),
        "project_count":  len(app_ctx.projects),
        "model_version":  app_ctx.model_config.get("model_version")
    })


@asynccontextmanager
async def lifespan(app_instance):
    """FastAPI lifespan: replaces deprecated @app.on_event('startup')."""
    # 1. Load data
    load_system_state()
    validate_system_integrity()

    # 2. Production Ready Checks
    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("production_readiness_warning", extra={"reason": "GROQ_API_KEY_MISSING", "impact": "LLM explanations will fail"})

    if not app_ctx.is_initialised():
        logger.error("system_critical_failure", extra={"reason": "DATA_NOT_LOADED", "impact": "Simulation API will return 500s"})

    yield  # server is now running
    # (shutdown logic would go here if needed)

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Production protection against stalled requests."""
    try:
        return await asyncio.wait_for(call_next(request), timeout=settings.TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error("request_timeout", extra={"path": request.url.path})
        return JSONResponse(
            status_code=504,
            content={"error": "Gateway Timeout", "message": "The request took too long to process."}
        )

# In-memory rate limiting dictionary (Production should use Redis)
_rate_limits: Dict[str, Dict[str, Any]] = {}

@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """Simple Sliding Window Rate Limiting."""
    # Bypass health
    if request.url.path == "/":
        return await call_next(request)
        
    client_ip = request.client.host if request.client else "unknown"
    now_ts = time.time()
    
    # Initialize or reset bucket
    if client_ip not in _rate_limits or now_ts - _rate_limits[client_ip]["start_time"] > 60:
        _rate_limits[client_ip] = {"start_time": now_ts, "count": 0}
        
    if _rate_limits[client_ip]["count"] >= settings.RATE_LIMIT_PER_MINUTE:
        logger.warning("rate_limit_exceeded", extra={"client_ip": client_ip})
        return JSONResponse(
            status_code=429,
            content={"error": "Too Many Requests", "message": "Rate limit exceeded"}
        )
        
    _rate_limits[client_ip]["count"] += 1
    return await call_next(request)

@app.middleware("http")
async def auth_placeholder(request: Request, call_next):
    """
    Production Authentication Layer.
    """
    # bypass for health check
    if request.url.path == "/":
        return await call_next(request)
        
    # Standardised API key validation
    auth_header = request.headers.get("Authorization", "")
    expected_token = f"Bearer {settings.API_KEY}"
    
    if settings.ENV != "development" and auth_header != expected_token:
        logger.warning("unauthorized_access_attempt", extra={"path": request.url.path})
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized", "message": "Valid API Key required"}
        )
    
    return await call_next(request)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Production standard error structure for unexpected failures."""
    logger.error("unhandled_server_error", extra={"error": str(exc), "path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": "An unexpected system error occurred", "request_id": REQUEST_ID.get()}
    )


# =====================================================
# API Endpoints (Routing Layer)
# =====================================================

@app.get("/")
def health_check():
    """
    Detailed system health and vitals for monitoring.
    """
    return {
        "status":         "OPERATIONAL",
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "request_id":     REQUEST_ID.get(),
        "vitals": {
            "model_version":  app_ctx.model_config.get("model_version", "3.0"),
            "employee_count": app_ctx.employee_count(),
            "project_count":  len(app_ctx.projects),
            "persistence_safe": True,
            "llm_active":     bool(os.environ.get("GROQ_API_KEY"))
        }
    }

@app.get("/org/state")
def get_org_state():
    """Returns the organization structure formatted for ReactFlow."""
    nodes = []
    edges = []

    for i, emp in enumerate(app_ctx.employees):
        nodes.append({
            "id":   emp.id,
            "type": "default",
            "data": {"label": f"{emp.name}\n({emp.role})"},
            "position": {
                "x": 200 + (i % 5) * 250,
                "y": 100 + (i // 5) * 150,
            },
        })
        if emp.reports_to:
            edges.append({
                "id":       f"e-{emp.reports_to}-{emp.id}",
                "source":   emp.reports_to,
                "target":   emp.id,
                "animated": True,
            })

    if not nodes and not edges:
        return {"nodes": [], "edges": []}
    return {"nodes": nodes, "edges": edges}

def validate_employee_exists(employee_id: Optional[str]):
    if employee_id and not app_ctx.has_employee(employee_id):
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")

@app.post("/simulate", response_model=ServiceResponse)
def simulate_scenario(request: SimulationRequest, background_tasks: BackgroundTasks) -> ServiceResponse:
    """Executes a high-fidelity workforce simulation."""
    start_time = time.time()
    try:
        validate_employee_exists(request.employee_id)
        result = run_standard_simulation(
            projects=app_ctx.projects,
            employees=app_ctx.employees,
            strategy_key=request.strategy,
            target_employee_id=request.employee_id,
            shock_mode=request.shock_test,
            random_seed=request.seed,
            model_config=app_ctx.model_config,
        )
        if result is None:
            raise SimulationEngineError("Simulation returned None unexpectedly.")

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info("simulation_complete", extra={
            "endpoint": "/simulate", "strategy": request.strategy,
            "seed": request.seed, "duration_ms": duration_ms,
        })
        # BackgroundTasks: audit logging runs AFTER the response is sent
        background_tasks.add_task(log_simulation_history, {
            "strategy": request.strategy, "employee_id": request.employee_id,
            "seed": request.seed, "duration_ms": duration_ms,
        })
        return result
    except HTTPException:
        raise
    except SimulationEngineError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("simulation_engine_error", extra={"endpoint": "/simulate", "error": str(e), "duration_ms": duration_ms})
        raise HTTPException(status_code=400, detail={"error": "Simulation Engine Error", "message": str(e)})
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("api_unexpected_failure", extra={"endpoint": "/simulate", "error": str(e), "duration_ms": duration_ms})
        raise HTTPException(status_code=500, detail={"error": "Unexpected Engine Failure"})

@app.post("/decision/compare", response_model=DecisionResult)
def compare_scenarios(request: DecisionRequest) -> DecisionResult:
    """Compares multiple HR strategies for a specific employee departure."""
    start_time = time.time()
    try:
        validate_employee_exists(request.employee_id)
        comparison = compare_hr_strategies(
            projects=app_ctx.projects,
            employees=app_ctx.employees,
            target_employee_id=request.employee_id,
            random_seed=request.seed,
            model_config=app_ctx.model_config,
        )
        if comparison is None:
            raise DecisionEngineError("Decision engine returned None unexpectedly.")
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info("decision_comparison_complete", extra={"endpoint": "/decision/compare", "seed": request.seed, "duration_ms": duration_ms})
        return comparison
    except HTTPException:
        raise
    except (SimulationEngineError, DecisionEngineError) as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("decision_engine_error", extra={"endpoint": "/decision/compare", "error": str(e), "duration_ms": duration_ms})
        raise HTTPException(status_code=400, detail={"error": "Decision Engine Error", "message": str(e)})
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("api_unexpected_failure", extra={"endpoint": "/decision/compare", "error": str(e), "duration_ms": duration_ms})
        raise HTTPException(status_code=500, detail={"error": "Unexpected Engine Failure"})

@app.post("/decision/explain", response_model=ExplanationResult)
def explain_decision(request: DecisionRequest) -> ExplanationResult:
    """Provides an LLM-driven interpretation of the best strategic choice."""
    start_time = time.time()
    try:
        validate_employee_exists(request.employee_id)
        explanation = explain_strategic_decision(
            projects=app_ctx.projects,
            employees=app_ctx.employees,
            target_employee_id=request.employee_id,
            random_seed=request.seed,
            model_config=app_ctx.model_config,
        )
        if explanation is None:
            raise LLMEngineError("Explanation engine returned None unexpectedly.")
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info("decision_explanation_complete", extra={"endpoint": "/decision/explain", "seed": request.seed, "duration_ms": duration_ms})
        return explanation
    except HTTPException:
        raise
    except (SimulationEngineError, DecisionEngineError, LLMEngineError) as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("explanation_engine_error", extra={"endpoint": "/decision/explain", "error": str(e), "duration_ms": duration_ms})
        raise HTTPException(status_code=400, detail={"error": "Explanation Engine Error", "message": str(e)})
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("api_unexpected_failure", extra={"endpoint": "/decision/explain", "error": str(e), "duration_ms": duration_ms})

# =====================================================
# Phase 6: Predictive Intelligence
# =====================================================

@app.get("/predict/attrition")
def predict_attrition(seed: int = 42):
    """Deterministic attrition probability using sigmoid model (delegated to prediction service)."""
    return predict_attrition_risk(app_ctx, seed)

@app.post("/predict/hiring-impact")
def predict_hiring_impact(request: HiringRequest):
    """Hiring productivity and revenue recovery forecast (delegated to prediction service)."""
    return _predict_hiring_impact(request)

@app.post("/predict/trajectory")
def predict_trajectory(request: TrajectoryRequest):
    """Month-by-month risk trajectory forecast (delegated to prediction service)."""
    return _predict_trajectory(request)

# =====================================================
# Phase 8: Decision OS (Expansion)
# =====================================================

@app.post("/portfolio/simulate")
def simulate_portfolio(request: PortfolioRequest):
    """Multi-project conflict density simulation (delegated to prediction service)."""
    return _simulate_portfolio(request)

@app.post("/budget/allocate")
def allocate_budget(request: BudgetRequest):
    """Log-scale budget impact simulation (delegated to prediction service)."""
    return _allocate_budget(request)

@app.post("/strategy/investment")
def strategic_investment(request: InvestmentRequest):
    """Training/automation/redundancy investment model (delegated to prediction service)."""
    return _strategic_investment(request)

@app.post("/strategy/merge")
def strategy_merge(request: MergeRequest):
    """M&A scenario model (delegated to prediction service)."""
    return _strategy_merge(request, app_ctx)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)