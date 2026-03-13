"""
Business Purpose:
Orchestrates the strategic comparison of different HR scenarios.
Integrates mathematical ranking with LLM-based narrative interpretation.

Technical Responsibility:
- Coordinate the evaluation of multiple strategies (Comparison Matrix).
- Invoke ranking logic to determine the 'Best Strategy'.
- Orchestrate LLM explanations for the winning scenario.
- Manage persistence of comparative decision records.

Determinism Level:
High for ranking. Non-deterministic for LLM narrative segments as noted in the LLM module.
"""

from typing import List, Dict, Any, Optional

# Internal Imports
from core.models import (
    Employee, 
    ExecutionProject, 
    StrategyType, 
    SimulationEngineError, 
    DecisionEngineError,
    DecisionResult,
    ExplanationResult
)
from core.simulation.service import run_standard_simulation
from core.decision.ranking import rank_strategies
from core.llm.explainer import interpret_simulation_outcome
from core.persistence.storage import log_decision_history

def compare_hr_strategies(
    projects: List[ExecutionProject],
    employees: List[Employee],
    target_employee_id: str,
    random_seed: Optional[int] = 42,
    model_config: Dict[str, Any] = {}
) -> DecisionResult:
    """
    Simulates a matrix of HR strategies and ranks them to find the optimal outcome.
    """
    # FIX-3: Added BASELINE as the control reference so all departure strategies
    # are compared against the status-quo outcome.
    strategy_options = [
        StrategyType.BASELINE,
        StrategyType.NO_REPLACE,
        StrategyType.IMMEDIATE,
        StrategyType.DELAYED,
    ]
    scenario_outcomes = []

    for strategy in strategy_options:
        simulation_data = run_standard_simulation(
            projects, employees, strategy, target_employee_id, random_seed=random_seed, model_config=model_config
        )

        org = simulation_data.organization
        risk = simulation_data.risk

        # FIX-1: Build canonical metric dict that matches ranking.py metric_keys exactly.
        # Now using typed Pydantic accessors
        metrics = {
            "profit":                    risk.average_profit,
            "volatility":                risk.profit_variance,
            "stability_score":           risk.stability_score,
            "org_health":                org.health_score,               
            "fragility_score":           org.structural_fragility_score,  
            "behavioral_fragility_index": org.behavioral_fragility_index,
        }

        # Burnout concentration: fraction of team at HIGH or MEDIUM burnout
        burnout_map = simulation_data.execution.burnout_risk
        burnout_concentration = (
            len([v for v in burnout_map.values() if v in ["HIGH", "MEDIUM"]]) / len(burnout_map)
            if burnout_map else 0.0
        )
        metrics["burnout_index"] = burnout_concentration

        # Attach strategy identity
        metrics["strategy"] = strategy
        scenario_outcomes.append(metrics)


    # Invoke Ranking Engine
    try:
        decision_result = rank_strategies(scenario_outcomes, decision_weights=model_config.get("decision_weights", {}))
        if decision_result is None:
            raise DecisionEngineError("Ranking engine returned None")
    except Exception as e:
        if isinstance(e, DecisionEngineError):
            raise
        raise DecisionEngineError(f"Ranking failure: {str(e)}")
    
    # Enrich result
    best_strategy = decision_result.ranked_strategies[0].strategy
    
    # Audit Persistence
    log_decision_history({
        "employee_id": target_employee_id,
        "best_strategy": best_strategy,
        "strategies_evaluated": strategy_options,
        "model_version": model_config.get("model_version")
    })
    
    return DecisionResult(
        best_strategy=best_strategy,
        comparison_matrix=decision_result.ranked_strategies,
        ranked_strategies=decision_result.ranked_strategies,
        governance=decision_result.governance
    )


def explain_strategic_decision(
    projects: List[ExecutionProject],
    employees: List[Employee],
    target_employee_id: str,
    random_seed: Optional[int] = 42,
    model_config: Dict[str, Any] = {}
) -> ExplanationResult:
    """
    Runs a full comparison and then generates a grounded LLM executive summary for the winner.
    """
    # 1. Run full comparison
    comparison_results = compare_hr_strategies(
        projects, employees, target_employee_id, random_seed, model_config
    )
    
    best_strategy_name = comparison_results.best_strategy
    best_scenario_snapshot = next(
        (s for s in comparison_results.comparison_matrix 
        if s.strategy == best_strategy_name), None
    )
    if not best_scenario_snapshot:
        raise DecisionEngineError(f"Best strategy {best_strategy_name} not found in comparison matrix")

    
    # 2. Invoke LLM Interpretation
    executive_narrative = interpret_simulation_outcome(
        target_scenario_data=best_scenario_snapshot.raw_metrics,
        strategy_name=best_strategy_name,
        simulation_seed=random_seed or 0,
        engine_version=model_config.get("model_version", "v3.0")
    )
    
    return ExplanationResult(
        best_strategy=best_strategy_name,
        rankings=comparison_results.ranked_strategies,
        executive_summary=executive_narrative,
        winning_data_snapshot=best_scenario_snapshot.raw_metrics
    )
