"""
Business Purpose:
Executes large-scale stochastic simulations (Monte Carlo) to quantify uncertainty and risk.
Evaluates the stability and volatility of HR strategies under environmental noise.

Technical Responsibility:
- Orchestrate parallel execution using ProcessPoolExecutor.
- Apply randomized noise (Duration, Revenue, Attrition) to simulation parameters.
- Aggregate iteration results into statistical metrics (Mean, Variance, P5, P95).
- Calculate the Stability Score (volatility-adjusted performance).

Determinism Level:
High. Each run uses a deterministic sub-seed derived from a master seed.

External Dependencies:
- concurrent.futures
- statistics
"""

import random
import statistics
import copy
import networkx as nx
import concurrent.futures
import os
from typing import List, Dict, Any, Optional, Callable
from core.models import SimulationEngineError

# Internal Imports (Refactored Structure)
from core.utils.finance import compute_profit
from core.risk.behavioral import convert_burnout_to_attrition_probability, compute_behavioral_fragility
from core.risk.structural import compute_structural_fragility
from core.utils.skills import compute_skill_redundancy

def _execute_single_iteration(
    iteration_seed: int, 
    projects: List[Any], 
    employees: List[Any], 
    strategy_function: Callable, 
    strategy_kwargs: Dict[str, Any], 
    noise_config: Dict[str, float]
) -> Dict[str, float]:
    """
    Performs one stochastic simulation run.
    
    Technical Note:
    Uses deepcopy to ensure isolation between parallel processes and prevent state corruption.
    """
    try:
        import numpy as np
        np.random.seed(iteration_seed)
    except ImportError:
        pass
    random.seed(iteration_seed)

    # 1. Isolate Organizational State
    isolated_projects = copy.deepcopy(projects)
    isolated_employees = copy.deepcopy(employees)


    # 2. Inject Operational Noise (Duration)
    for project in isolated_projects:
        for task in project.tasks:
            duration_noise = random.uniform(
                1.0 - noise_config["task_duration_noise"], 
                1.0 + noise_config["task_duration_noise"]
            )
            task.estimated_hours *= duration_noise

    # 3. Inject Market Noise (Revenue)
    for project in isolated_projects:
        revenue_noise = random.uniform(
            1.0 - noise_config["revenue_noise"], 
            1.0 + noise_config["revenue_noise"]
        )
        project.base_revenue *= revenue_noise

    # 4. Re-evaluate Strategy Behavior
    scenario_result = strategy_function(
        isolated_projects, 
        isolated_employees, 
        **(strategy_kwargs or {})
    )
    final_duration = scenario_result["duration_weeks"]

    # 5. Composite Metric Re-calculation
    finance_metrics = compute_profit(isolated_projects, isolated_employees, final_duration)
    realized_profit = finance_metrics["profit"]

    # 6. Attrition Shock Simulation
    # Converts predicted burnout into discrete exit events
    attrition_probabilities = convert_burnout_to_attrition_probability(scenario_result["burnout_risk"])
    for prob in attrition_probabilities.values():
        if random.random() < prob:
            # Departure impact: additional project slippage / cost overhead
            realized_profit *= (1.0 - noise_config["attrition_shock"])

    # 7. Structural & Behavioral Snapshot
    structural_analysis = compute_structural_fragility(isolated_employees)
    
    # Rebuild network for behavioral contagion analysis
    net = nx.Graph()
    for e in isolated_employees:
        net.add_node(e.id)
        if e.reports_to:
            net.add_edge(e.reports_to, e.id)
    
    centrality_scores = nx.betweenness_centrality(net)
    redundancy_map = compute_skill_redundancy(isolated_employees)
    behavioral_analysis = compute_behavioral_fragility(
        isolated_employees, net, scenario_res_util := scenario_result["utilization"], centrality_scores, redundancy_map
    )

    return {
        "profit": realized_profit,
        "duration_weeks": final_duration,
        "structural_fragility": structural_analysis["fragility_score"],
        "behavioral_fragility": behavioral_analysis["behavioral_fragility_index"],
        "contagion_level": behavioral_analysis["average_attrition_probability"]
    }


def run_monte_carlo_simulation(
    projects: List[Any], 
    employees: List[Any], 
    strategy_function: Callable, 
    strategy_kwargs: Optional[Dict[str, Any]] = None, 
    iteration_count: int = 50, 
    master_seed: Optional[int] = None,
    noise_params: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Orchestrates a parallel Monte Carlo simulation to evaluate strategy robustness.
    
    Args:
        projects: Target projects.
        employees: Target employee pool.
        strategy_function: The HR strategy function to evaluate.
        iteration_count: Number of stochastic runs.
        master_seed: Optional anchor for deterministic results.
        noise_params: Configuration for stochastic variance.
        
    Returns:
        Dict[str, Any]: Statistical summary of the strategy's performance profile.
    """
    if noise_params is None:
        noise_params = {
            "task_duration_noise": 0.1,
            "revenue_noise": 0.05,
            "attrition_shock": 0.1
        }
    
    if iteration_count < 1:
        iteration_count = 30


    # Establish determinism using a repeatable sequence
    if master_seed is None:
        master_seed = random.randint(0, 1000000)
    
    # Use a dedicated generator for sub-seeds to ensure they don't depend on global state
    seed_rng = random.Random(master_seed)
    sub_seeds = [seed_rng.randint(0, 1000000) for _ in range(iteration_count)]

    aggregated_metrics = {
        "profit": [],
        "duration_weeks": [],
        "structural_fragility": [],
        "behavioral_fragility": [],
        "contagion_level": []
    }

    # Parallel Execution Layer (SaaS Grade)
    # CPU-bound simulations executed in a worker pool
    max_parallel_workers = min(os.cpu_count() or 1, 8)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel_workers) as executor:
        execution_futures = [
            executor.submit(_execute_single_iteration, s, projects, employees, strategy_function, strategy_kwargs, noise_params)
            for s in sub_seeds
        ]
        
        for future in concurrent.futures.as_completed(execution_futures):
            try:
                iteration_data = future.result()
                if iteration_data is None:
                    raise SimulationEngineError("Monte Carlo iteration returned None")
                for key in aggregated_metrics:
                    aggregated_metrics[key].append(iteration_data[key])
            except Exception as e:
                raise SimulationEngineError(f"Monte Carlo parallel iteration failed: {str(e)}")

    # Statistical Synthesis
    summary_statistics = {}
    for key, values in aggregated_metrics.items():
        mean_magnitude = statistics.mean(values)
        variance_magnitude = statistics.variance(values) if len(values) > 1 else 0.0
        summary_statistics[f"mean_{key}"] = round(mean_magnitude, 2)
        summary_statistics[f"variance_{key}"] = round(variance_magnitude, 4)
        
        if key == "profit":
            sorted_profits = sorted(values)
            summary_statistics["p5_profit"] = round(sorted_profits[int(0.05 * iteration_count)], 2)
            summary_statistics["p95_profit"] = round(sorted_profits[int(0.95 * iteration_count) - 1], 2)
            summary_statistics["probability_of_loss"] = round(len([v for v in values if v < 0]) / iteration_count, 2)
            # Standardizing naming for downstream decision engine
            summary_statistics["profit_variance"] = summary_statistics[f"variance_{key}"]
        
        if key == "duration_weeks":
            summary_statistics["duration_variance"] = summary_statistics[f"variance_{key}"]

    # Stability Score: Inverse of relative variance (clamped 0-1)
    # Higher score = more predictable financial outcome
    profit_variability = summary_statistics["profit_variance"] / (abs(summary_statistics["mean_profit"]) + 1e-6)
    summary_statistics["stability_score"] = round(max(0.0, min(1.0, 1.0 - profit_variability)), 4)
    
    # Audit Governance Metadata
    summary_statistics["governance"] = {
        "master_seed": master_seed,
        "iterations": iteration_count,
        "noise_profile": noise_params
    }

    # Legacy mapping for compatibility
    summary_statistics["seed_used"] = master_seed
    summary_statistics["runs"] = iteration_count

    return summary_statistics