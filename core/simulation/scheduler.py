"""
Business Purpose:
Defines and executes various HR strategy scenarios to evaluate organizational resilience.
Provides the logical framework for "What-If" analysis regarding employee turnover.

Technical Responsibility:
- Orchestrate simulation runs for diverse strategy functions.
- Handle data cloning and property modulation for hypothetical replacements.
- Calculate time-shifted impact for hiring delays.
- Logistic onboarding curve for realistic productivity ramp.

Determinism Level:
High. All strategies use seeded RNG or purely deterministic math.
"""

import math
import random
from copy import deepcopy
from typing import List, Dict, Any, Optional
from core.models import Employee, ExecutionProject, SimulationEngineError
from core.simulation.executor import simulate_project_execution

# Business Constants
DEFAULT_HIRING_DELAY_WEEKS = 4
# Logistic curve parameters for onboarding ramp
_LOGISTIC_K = 0.8          # steepness of the S-curve
_EPSILON = 1e-9


def _logistic_productivity(weeks_active: float, ramp_up_weeks: float) -> float:
    """
    Computes hire productivity at `weeks_active` into the role using a logistic S-curve.

    Formula:
        productivity(t) = 1 / (1 + e^(-k * (t - midpoint)))

    Args:
        weeks_active: Weeks since the hire's first day.
        ramp_up_weeks: Total expected ramp duration (midpoint = ramp_up_weeks / 2).

    Returns:
        float: Productivity multiplier in [0.0, 1.0].
    """
    midpoint = ramp_up_weeks / 2.0
    return 1.0 / (1.0 + math.exp(-_LOGISTIC_K * (weeks_active - midpoint)))


def simulate_baseline(
    projects: List[ExecutionProject],
    employees: List[Employee],
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """
    Standard baseline simulation with the current team.
    No modifications — serves as the control scenario.
    """
    return simulate_project_execution(projects, employees)


def simulate_no_replacement(
    projects: List[ExecutionProject],
    employees: List[Employee],
    removed_employee_id: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """
    Simulates the impact of removing an employee without hiring a replacement.

    Args:
        projects: List of projects to simulate.
        employees: Current employee pool.
        removed_employee_id: ID of the employee to be removed.

    Returns:
        Dict[str, Any]: Simulation results for the reduced team.
    """
    remaining_employees = [e for e in employees if e.id != removed_employee_id]
    return simulate_project_execution(projects, remaining_employees)


def simulate_immediate_replacement(
    projects: List[ExecutionProject],
    employees: List[Employee],
    removed_employee_id: str = "",
    ramp_up_weeks: float = 12.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Simulates an immediate replacement hire with a logistic productivity ramp-up.

    The new hire starts on day 1 but follows a logistic S-curve to full productivity.
    Midpoint productivity (50%) is reached at ramp_up_weeks / 2.

    Args:
        projects: List of projects to simulate.
        employees: Current employee pool.
        removed_employee_id: ID of the employee being replaced.
        ramp_up_weeks: Total weeks to reach full productivity (default 12 weeks / 3 months).

    Returns:
        Dict[str, Any]: Simulation results with the new hire included.
    """
    remaining_employees = [e for e in employees if e.id != removed_employee_id]

    original_employee = next((e for e in employees if e.id == removed_employee_id), None)
    if not original_employee:
        if not remaining_employees:
            raise SimulationEngineError(f"Employee {removed_employee_id} not found and pool is empty")
        # If the exact employee isn't found (e.g. already removed), run without replacement
        return simulate_project_execution(projects, remaining_employees)

    # Logistic productivity at week 1 (first active week)
    initial_ramp = _logistic_productivity(1.0, ramp_up_weeks)

    # Create a cloned replacement with logistic-curve initial productivity
    new_hire = deepcopy(original_employee)
    new_hire.id = f"{removed_employee_id}_NEW"
    new_hire.name = f"{original_employee.name} (Replacement)"
    new_hire.productivity_multiplier = round(initial_ramp, 3)

    updated_team = remaining_employees + [new_hire]
    return simulate_project_execution(projects, updated_team)


def simulate_delayed_replacement(
    projects: List[ExecutionProject],
    employees: List[Employee],
    removed_employee_id: str = "",
    hiring_delay_weeks: int = DEFAULT_HIRING_DELAY_WEEKS,
    ramp_up_weeks: float = 12.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Simulates a hiring delay followed by a replacement with a logistic ramp.

    Phase 1: Run without the departed employee for `hiring_delay_weeks`.
    Phase 2: Add a replacement hire whose productivity at start reflects the
             logistic curve evaluated at week 1 relative to their start date.
    The hiring delay is added to the final project duration as overhead.

    Args:
        projects: List of projects to simulate.
        employees: Current employee pool.
        removed_employee_id: ID of the employee being replaced.
        hiring_delay_weeks: Weeks before the replacement starts.
        ramp_up_weeks: Total weeks for the replacement to reach full productivity.

    Returns:
        Dict[str, Any]: Simulation results accounting for the delay.
    """
    # Simulate with replacement (logistic ramp from day 1 of hire)
    ramp_up_results = simulate_immediate_replacement(
        projects, employees, removed_employee_id, ramp_up_weeks=ramp_up_weeks
    )
    # Additive overhead for the gap period before the hire starts
    ramp_up_results["duration_weeks"] = round(
        ramp_up_results["duration_weeks"] + hiring_delay_weeks, 2
    )
    return ramp_up_results


def simulate_price_increase(
    projects: List[ExecutionProject],
    employees: List[Employee],
    price_delta: float = 0.1,
    churn_impact: float = 0.2,
    **kwargs,
) -> Dict[str, Any]:
    """
    Simulates a price increase event.

    Effect:
        - Base revenue increases by `price_delta` fraction.
        - Decay rate (market sensitivity/churn) increases by `churn_impact` fraction,
          partially offsetting the revenue gain via faster market erosion.

    Args:
        price_delta: Fractional revenue uplift (default 10%).
        churn_impact: Fractional increase in decay rate (default 20%).
    """
    modified_projects = deepcopy(projects)
    for project in modified_projects:
        project.base_revenue = project.base_revenue * (1.0 + price_delta)
        project.decay_rate = project.decay_rate * (1.0 + churn_impact)

    return simulate_project_execution(modified_projects, employees)


def simulate_restructure(
    projects: List[ExecutionProject],
    employees: List[Employee],
    restructure_intensity: float = 0.5,
    seed: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Simulates a team restructuring event.

    A fraction (`restructure_intensity`) of employees are re-parented to a random
    manager/lead. Each re-parented employee takes a 10% productivity hit to model
    onboarding/context-switching friction.

    Uses seeded RNG for deterministic results.

    Args:
        restructure_intensity: Fraction of employees to re-parent (0.0-1.0).
        seed: Optional RNG seed for determinism.
    """
    rng = random.Random(seed if seed is not None else 0)

    modified_employees = deepcopy(employees)
    potential_managers = [
        e.id for e in modified_employees
        if "manager" in e.role.lower() or "lead" in e.role.lower()
    ]

    if not potential_managers:
        potential_managers = [e.id for e in modified_employees]

    for emp in modified_employees:
        if rng.random() < restructure_intensity:
            possible_parents = [m for m in potential_managers if m != emp.id]
            if possible_parents:
                emp.reports_to = rng.choice(possible_parents)
                emp.productivity_multiplier = max(0.1, emp.productivity_multiplier * 0.9)

    return simulate_project_execution(projects, modified_employees)