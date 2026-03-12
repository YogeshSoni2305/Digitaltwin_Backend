"""
Business Purpose:
Executes large-scale stochastic simulations (Monte Carlo) to quantify uncertainty and risk.
Evaluates the stability and volatility of HR strategies under environmental noise.

Technical Responsibility:
- Orchestrate parallel execution using ThreadPoolExecutor (avoids pickle failures).
- Apply randomized noise (Duration, Revenue, Attrition) to simulation parameters.
- Aggregate iteration results into statistical metrics (Mean, Variance, P5, P95).
- Calculate the Stability Score (volatility-adjusted performance).

Determinism Level:
High. Each run receives a unique sub-seed derived from a master Random instance,
ensuring full reproducibility when the same master_seed is supplied.

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

# Internal Imports
from core.utils.finance import compute_profit
from core.risk.behavioral import convert_burnout_to_attrition_probability, compute_behavioral_fragility
from core.risk.structural import compute_structural_fragility
from core.utils.skills import compute_skill_redundancy

# Numerical stability
_EPSILON = 1e-9
# Sub-seed range (10^9 to minimise collision probability)
_SEED_RANGE = 10 ** 9


def _execute_single_iteration(
    iteration_seed: int,
    projects: List[Any],
    employees: List[Any],
    strategy_function: Callable,
    strategy_kwargs: Dict[str, Any],
    noise_config: Dict[str, float],
) -> Dict[str, float]:
    """
    Performs one stochastic simulation run.

    Each iteration receives a unique sub-seed so results are fully reproducible
    when the master seed is known.
    """
    # Seed both the standard library RNG and numpy (if present)
    try:
        import numpy as np
        np.random.seed(iteration_seed % (2 ** 32))
    except ImportError:
        pass
    random.seed(iteration_seed)

    # 1. Isolate Organisational State
    isolated_projects = copy.deepcopy(projects)
    isolated_employees = copy.deepcopy(employees)

    # 2. Inject Operational Noise (Duration ±noise%)
    for project in isolated_projects:
        for task in project.tasks:
            noise = random.uniform(
                1.0 - noise_config["task_duration_noise"],
                1.0 + noise_config["task_duration_noise"],
            )
            task.estimated_hours = max(_EPSILON, task.estimated_hours * noise)

    # 3. Inject Market Noise (Revenue ±noise%)
    for project in isolated_projects:
        noise = random.uniform(
            1.0 - noise_config["revenue_noise"],
            1.0 + noise_config["revenue_noise"],
        )
        project.base_revenue = max(0.0, project.base_revenue * noise)

    # 4. Execute Strategy
    scenario_result = strategy_function(isolated_projects, isolated_employees, **(strategy_kwargs or {}))
    final_duration = max(0.0, scenario_result["duration_weeks"])

    # 5. Financial Composite
    finance_metrics = compute_profit(isolated_projects, isolated_employees, final_duration)
    realized_profit = finance_metrics["profit"]

    # Guard against NaN/negative cascades
    if not isinstance(realized_profit, (int, float)) or realized_profit != realized_profit:
        realized_profit = 0.0

    # 6. Attrition Shock Simulation
    attrition_probabilities = convert_burnout_to_attrition_probability(scenario_result["burnout_risk"])
    for prob in attrition_probabilities.values():
        if random.random() < prob:
            realized_profit = realized_profit * (1.0 - noise_config["attrition_shock"])

    # 7. Structural & Behavioral Snapshot
    structural_analysis = compute_structural_fragility(isolated_employees)

    net = nx.Graph()
    for e in isolated_employees:
        net.add_node(e.id)
        if e.reports_to:
            net.add_edge(e.reports_to, e.id)

    centrality_scores = nx.betweenness_centrality(net)
    redundancy_map = compute_skill_redundancy(isolated_employees)

    utilization_snapshot = scenario_result.get("utilization", {})
    behavioral_analysis = compute_behavioral_fragility(
        isolated_employees, net, utilization_snapshot, centrality_scores, redundancy_map
    )

    return {
        "profit": realized_profit,
        "duration_weeks": final_duration,
        "structural_fragility": structural_analysis["fragility_score"],
        "behavioral_fragility": behavioral_analysis["behavioral_fragility_index"],
        "contagion_level": behavioral_analysis["average_attrition_probability"],
    }


def run_monte_carlo_simulation(
    projects: List[Any],
    employees: List[Any],
    strategy_function: Callable,
    strategy_kwargs: Optional[Dict[str, Any]] = None,
    iteration_count: int = 50,
    master_seed: Optional[int] = None,
    noise_params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Orchestrates a parallel Monte Carlo simulation to evaluate strategy robustness.

    Seed propagation:
        master_rng = random.Random(master_seed)
        iteration_seeds = [master_rng.randint(0, 10^9) for _ in range(N)]

    This guarantees identical iteration seeds for the same master_seed
    regardless of global RNG state.

    Uses ThreadPoolExecutor to avoid multiprocessing pickle failures with
    locally-defined callables while still enabling I/O-level concurrency.

    Args:
        projects: Target projects.
        employees: Target employee pool.
        strategy_function: HR strategy function to evaluate.
        strategy_kwargs: Extra keyword arguments forwarded to the strategy.
        iteration_count: Number of stochastic runs (minimum 1).
        master_seed: Anchor for deterministic reproducibility.
        noise_params: Stochastic variance configuration.

    Returns:
        Dict[str, Any]: Statistical summary of the strategy's performance profile.
    """
    if noise_params is None:
        noise_params = {
            "task_duration_noise": 0.10,
            "revenue_noise": 0.05,
            "attrition_shock": 0.10,
        }

    iteration_count = max(1, iteration_count)

    # Deterministic seed chain
    if master_seed is None:
        master_seed = random.randint(0, _SEED_RANGE)

    master_rng = random.Random(master_seed)
    sub_seeds = [master_rng.randint(0, _SEED_RANGE) for _ in range(iteration_count)]

    aggregated_metrics: Dict[str, List[float]] = {
        "profit": [],
        "duration_weeks": [],
        "structural_fragility": [],
        "behavioral_fragility": [],
        "contagion_level": [],
    }

    # ThreadPoolExecutor avoids multiprocessing pickle failures for locally
    # defined or lambda callables while still parallelising I/O.
    max_workers = min(os.cpu_count() or 1, 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _execute_single_iteration,
                seed,
                projects,
                employees,
                strategy_function,
                strategy_kwargs or {},
                noise_params,
            )
            for seed in sub_seeds
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                iteration_data = future.result()
                if iteration_data is None:
                    raise SimulationEngineError("Monte Carlo iteration returned None")
                for key in aggregated_metrics:
                    value = iteration_data[key]
                    # Guard: skip NaN values
                    if isinstance(value, (int, float)) and value == value:
                        aggregated_metrics[key].append(value)
                    else:
                        aggregated_metrics[key].append(0.0)
            except Exception as e:
                raise SimulationEngineError(f"Monte Carlo iteration failed: {str(e)}")

    # Statistical Synthesis
    summary_statistics: Dict[str, Any] = {}

    for key, values in aggregated_metrics.items():
        if not values:
            values = [0.0]
        mean_val = statistics.mean(values)
        var_val  = statistics.variance(values) if len(values) > 1 else 0.0

        summary_statistics[f"mean_{key}"] = round(mean_val, 2)
        summary_statistics[f"variance_{key}"] = round(var_val, 4)

        if key == "profit":
            sorted_profits = sorted(values)
            n = len(sorted_profits)
            summary_statistics["p5_profit"]  = round(sorted_profits[max(0, int(0.05 * n))], 2)
            summary_statistics["p95_profit"] = round(sorted_profits[min(n - 1, int(0.95 * n) - 1)], 2)
            summary_statistics["probability_of_loss"] = round(
                len([v for v in values if v < 0]) / max(n, 1), 2
            )
            summary_statistics["profit_variance"] = summary_statistics[f"variance_{key}"]

        if key == "duration_weeks":
            summary_statistics["duration_variance"] = summary_statistics[f"variance_{key}"]

    # Stability Score: inverse of relative profit variance, clamped [0, 1]
    mean_profit = summary_statistics.get("mean_profit", 0.0)
    profit_var  = summary_statistics.get("profit_variance", 0.0)
    profit_variability = profit_var / (abs(mean_profit) + _EPSILON)
    summary_statistics["stability_score"] = round(max(0.0, min(1.0, 1.0 - profit_variability)), 4)

    # Audit Governance Metadata
    summary_statistics["governance"] = {
        "master_seed": master_seed,
        "iterations": iteration_count,
        "noise_profile": noise_params,
    }

    # Legacy compatibility aliases
    summary_statistics["seed_used"] = master_seed
    summary_statistics["runs"] = iteration_count

    return summary_statistics