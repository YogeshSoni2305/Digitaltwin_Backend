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
    DecisionEngineError
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
) -> Dict[str, Any]:
    """
    Simulates a matrix of HR strategies and ranks them to find the optimal outcome.
    """
    strategy_options = [
        StrategyType.NO_REPLACE, 
        StrategyType.IMMEDIATE, 
        StrategyType.DELAYED
    ]
    scenario_outcomes = []
    
    for strategy in strategy_options:
        simulation_data = run_standard_simulation(
            projects, employees, strategy, target_employee_id, random_seed=random_seed, model_config=model_config
        )
        
        # Extract metrics for the ranking engine
        metrics = simulation_data["organization"].copy()
        metrics["profit"] = simulation_data["risk"]["average_profit"]
        metrics["volatility"] = simulation_data["risk"]["profit_variance"]
        metrics["stability_score"] = simulation_data["risk"]["stability_score"]
        
        # Calculate categorical burnout concentration for ranking logic
        burnout_map = simulation_data["execution"]["burnout_risk"]
        burnout_concentration = (
            len([v for v in burnout_map.values() if v in ["HIGH", "MEDIUM"]]) / len(burnout_map)
            if burnout_map else 0.0
        )
        metrics["burnout_index"] = burnout_concentration
        
        # Attach identity
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
    best_strategy = decision_result["ranked_strategies"][0]["strategy"]
    decision_result["best_strategy"] = best_strategy
    decision_result["comparison_matrix"] = decision_result["ranked_strategies"]
    
    # Audit Persistence
    log_decision_history({
        "employee_id": target_employee_id,
        "best_strategy": best_strategy,
        "strategies_evaluated": strategy_options,
        "model_version": model_config.get("model_version")
    })
    
    return decision_result


def explain_strategic_decision(
    projects: List[ExecutionProject],
    employees: List[Employee],
    target_employee_id: str,
    random_seed: Optional[int] = 42,
    model_config: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """
    Runs a full comparison and then generates a grounded LLM executive summary for the winner.
    """
    # 1. Run full comparison
    comparison_results = compare_hr_strategies(
        projects, employees, target_employee_id, random_seed, model_config
    )
    
    best_strategy_name = comparison_results["best_strategy"]
    best_scenario_snapshot = next(
        (s for s in comparison_results["comparison_matrix"] 
        if s["strategy"] == best_strategy_name), None
    )
    if not best_scenario_snapshot:
        raise DecisionEngineError(f"Best strategy {best_strategy_name} not found in comparison matrix")

    
    # 2. Invoke LLM Interpretation
    executive_narrative = interpret_simulation_outcome(
        target_scenario_data=best_scenario_snapshot["raw_metrics"],
        strategy_name=best_strategy_name,
        simulation_seed=random_seed or 0,
        engine_version=model_config.get("model_version", "v3.0")
    )
    
    return {
        "best_strategy": best_strategy_name,
        "rankings": comparison_results["ranked_strategies"],
        "executive_summary": executive_narrative,
        "winning_data_snapshot": best_scenario_snapshot["raw_metrics"]
    }
