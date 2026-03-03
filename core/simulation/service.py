"""
Business Purpose:
Acts as the central orchestration layer (Service Layer) for the Digital Twin.
Coordinates simulation runs, risk analysis, and LLM interpretations.

Technical Responsibility:
- Orchestrate high-level simulation requests.
- Integrate structural and behavioral risk analysis.
- Prepare snapshots for LLM interpretation.
- Manage persistence calls for auditability.

Determinism Level:
High. Service orchestration is deterministic.

External Dependencies:
- networkx
"""

import networkx as nx
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


# Internal Imports
from core.models import Employee, ExecutionProject, StrategyType, SimulationEngineError
from core.simulation.executor import simulate_project_execution
from core.simulation.scheduler import (
    simulate_no_replacement, 
    simulate_immediate_replacement, 
    simulate_delayed_replacement,
    simulate_baseline
)

from core.monte_carlo.engine import run_monte_carlo_simulation
from core.risk.structural import compute_structural_fragility
from core.risk.behavioral import compute_behavioral_fragility
from core.decision.ranking import rank_strategies
from core.utils.finance import compute_profit
from core.utils.validation import compute_aggregate_org_health
from core.utils.structure import validate_span_of_control, build_reporting_map
from core.utils.skills import compute_skill_redundancy
from core.persistence.storage import log_simulation_history, log_decision_history

def run_standard_simulation(
    projects: List[ExecutionProject],
    employees: List[Employee],
    strategy_key: StrategyType,
    target_employee_id: Optional[str],
    shock_mode: bool = False,
    random_seed: Optional[int] = None,
    model_config: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """
    Orchestrates a full simulation run including Monte Carlo and multi-layer risk analysis.
    """
    
    # 1. Map Strategy to Logic
    STRATEGY_FUNCTION_MAP = {
        StrategyType.NO_REPLACE: simulate_no_replacement,
        StrategyType.IMMEDIATE: simulate_immediate_replacement,
        StrategyType.DELAYED: simulate_delayed_replacement,
        StrategyType.BASELINE: simulate_baseline
    }

    
    selected_strategy_fn = STRATEGY_FUNCTION_MAP.get(strategy_key)
    if not selected_strategy_fn:
        raise SimulationEngineError(f"Unsupported strategy: {strategy_key}")

    # 2. Execution Logic Run
    try:
        execution_result = selected_strategy_fn(projects, employees, target_employee_id)
        if execution_result is None:
            raise SimulationEngineError(f"Strategy {strategy_key} returned None")
    except Exception as e:
        if isinstance(e, SimulationEngineError):
            raise
        raise SimulationEngineError(f"Execution failure in strategy {strategy_key}: {str(e)}")
    
    # 3. Stochastic Risk Analysis (Monte Carlo)
    noise_config = model_config.get("noise_parameters")
    mc_stats = run_monte_carlo_simulation(
        projects, 
        employees, 
        selected_strategy_fn, 
        strategy_kwargs={"removed_employee_id": target_employee_id} if target_employee_id else {},
        iteration_count=30,
        master_seed=random_seed,
        noise_params=noise_config
    )

    # 4. Financial Post-Processing
    finance_metrics = compute_profit(projects, employees, execution_result["duration_weeks"])

    # 5. Organizational Health & Structural Risk
    redundancy_map = compute_skill_redundancy(employees)
    health_score = compute_aggregate_org_health(
        validate_span_of_control(employees),
        redundancy_map
    )
    structural_risk = compute_structural_fragility(employees)

    # 6. Behavioral Risk & Network Contagion
    reporting_net = nx.Graph()
    for e in employees:
        reporting_net.add_node(e.id)
        if e.reports_to:
            reporting_net.add_edge(e.reports_to, e.id)
    
    centrality_map = nx.betweenness_centrality(reporting_net)
    
    # Apply Stress Test if Shock Mode is active
    effective_utilization = execution_result["utilization"].copy()
    if shock_mode:
        # Shock Top 20% central hubs
        sorted_hubs = sorted(centrality_map.items(), key=lambda x: x[1], reverse=True)
        top_hubs_count = max(1, int(len(employees) * 0.2))
        for i in range(top_hubs_count):
            node_id = sorted_hubs[i][0]
            if node_id in effective_utilization:
                effective_utilization[node_id] = min(1.0, effective_utilization[node_id] + 0.2)

    behavioral_risk = compute_behavioral_fragility(
        employees, reporting_net, effective_utilization, centrality_map, redundancy_map
    )

    # 7. Final Response Assembly
    service_response = {
        "execution": execution_result,
        "financial": finance_metrics,
        "risk": {
            "average_profit": mc_stats["mean_profit"],
            "p5_profit": mc_stats["p5_profit"],
            "p95_profit": mc_stats["p95_profit"],
            "profit_variance": mc_stats["profit_variance"],
            "stability_score": mc_stats["stability_score"],
            "mean_structural_fragility": mc_stats.get("mean_structural_fragility", structural_risk["fragility_score"]),
            "mean_behavioral_fragility": mc_stats.get("mean_behavioral_fragility", behavioral_risk["behavioral_fragility_index"]),
            "seed_used": mc_stats["seed_used"]
        },
        "organization": {
            "health_score": health_score,
            "structural_fragility_score": structural_risk["fragility_score"],
            "behavioral_fragility_index": behavioral_risk["behavioral_fragility_index"],
            "risk_level": structural_risk["risk_level"]
        },
        "governance": {
            "model_version": model_config.get("model_version"),
            "timestamp": datetime.now(timezone.utc).isoformat()

        }
    }

    # 8. Persistence Audit
    log_simulation_history({
        "strategy": strategy_key,
        "employee_id": target_employee_id,
        "profit": mc_stats["mean_profit"],
        "stability": mc_stats["stability_score"],
        "model_version": model_config.get("model_version")
    })

    return service_response
