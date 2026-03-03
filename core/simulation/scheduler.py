"""
Business Purpose:
Defines and executes various HR strategy scenarios to evaluate organizational resilience.
Provides the logical framework for "What-If" analysis regarding employee turnover.

Technical Responsibility:
- Orchestrate simulation runs for diverse strategy functions.
- Handle data cloning and property modulation for hypothetical replacements.
- Calculate time-shifted impact for hiring delays.

Determinism Level:
High. Results depend entirely on the underlying deterministic simulation executor.
"""

from copy import deepcopy
from typing import List, Dict, Any, Optional
from core.models import Employee, ExecutionProject, SimulationEngineError
from core.simulation.executor import simulate_project_execution

# Business Constants
DEFAULT_RAMP_PRODUCTIVITY = 0.5
DEFAULT_HIRING_DELAY_WEEKS = 4

def simulate_no_replacement(
    projects: List[ExecutionProject], 
    employees: List[Employee], 
    removed_employee_id: str
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
    removed_employee_id: str, 
    ramp_factor: float = DEFAULT_RAMP_PRODUCTIVITY
) -> Dict[str, Any]:
    """
    Simulates the impact of an immediate replacement hire with a productivity ramp-up period.
    
    Args:
        projects: List of projects to simulate.
        employees: Current employee pool.
        removed_employee_id: ID of the employee being replaced.
        ramp_factor: Productivity multiplier for the new hire (0.0 to 1.0).
        
    Returns:
        Dict[str, Any]: Simulation results with the new hire included.
    """
    remaining_employees = [e for e in employees if e.id != removed_employee_id]

    original_employee = next((e for e in employees if e.id == removed_employee_id), None)
    if not original_employee:
        raise SimulationEngineError(f"Employee {removed_employee_id} not found")

    
    # Create a cloned replacement with lower productivity
    new_hire = deepcopy(original_employee)
    new_hire.id = f"{removed_employee_id}_NEW"
    new_hire.name = f"{original_employee.name} (Replacement)"
    new_hire.productivity_multiplier = ramp_factor

    updated_team = remaining_employees + [new_hire]

    return simulate_project_execution(projects, updated_team)


def simulate_delayed_replacement(
    projects: List[ExecutionProject], 
    employees: List[Employee], 
    removed_employee_id: str, 
    hiring_delay_weeks: int = DEFAULT_HIRING_DELAY_WEEKS, 
    ramp_factor: float = DEFAULT_RAMP_PRODUCTIVITY
) -> Dict[str, Any]:
    """
    Simulates the impact of a hiring delay followed by a replacement hire.
    
    Technical Note:
    This currently approximates the delay by simulating without the employee and 
    adding the delay weeks to the final project duration.
    
    Args:
        projects: List of projects to simulate.
        employees: Current employee pool.
        removed_employee_id: ID of the employee being replaced.
        hiring_delay_weeks: Number of weeks before the replacement starts.
        ramp_factor: Productivity multiplier for the new hire.
        
    Returns:
        Dict[str, Any]: Simulation results accounting for the delay.
    """
    # 1. Simulate the ramp-up scenario (assuming they were hired eventually)
    ramp_up_results = simulate_immediate_replacement(
        projects, employees, removed_employee_id, ramp_factor
    )

    # 2. Add the hiring delay to the final duration
    # In a real SaaS engine, we might run two simulations sequentially,
    # but for this model, the delay is an additive overhead.
    ramp_up_results["duration_weeks"] += hiring_delay_weeks
    return ramp_up_results

def simulate_baseline(
    projects: List[ExecutionProject], 
    employees: List[Employee], 
    *args, **kwargs
) -> Dict[str, Any]:
    """
    Standard baseline simulation with the current team.
    """
    return simulate_project_execution(projects, employees)