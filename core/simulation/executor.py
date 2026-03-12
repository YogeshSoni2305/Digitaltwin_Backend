"""
Business Purpose:
Orchestrates the deterministic simulation of project tasks and their assignment to employees.
Calculates project duration, resource utilization, and burnout risk levels.

Technical Responsibility:
- Build task conversion to Directed Acyclic Graphs (DAG).
- Implement topological sort for execution ordering.
- Assign employees based on skill matching and load balancing.
- Recalculate full-stack metrics post-simulation.

Determinism Level:
High. Assignment is deterministic based on skill scores and current utilization.

External Dependencies:
- networkx
"""

import math
import networkx as nx
from typing import List, Dict, Optional, Any
from core.models import Employee, ExecutionProject, ExecutionTask

# Business Constants
THRESHOLD_BURNOUT_HIGH = 0.90
THRESHOLD_BURNOUT_MEDIUM = 0.75

# Numerical stability epsilon
_EPSILON = 1e-9


def build_task_graph(project: ExecutionProject) -> nx.DiGraph:
    """
    Constructs a Directed Acyclic Graph (DAG) for project tasks based on dependencies.

    Args:
        project: The ExecutionProject containing tasks.

    Returns:
        nx.DiGraph: The task dependency graph.
    """
    dependency_graph = nx.DiGraph()

    for task in project.tasks:
        dependency_graph.add_node(task.id, data=task)

    for task in project.tasks:
        for dependency_id in task.dependencies:
            dependency_graph.add_edge(dependency_id, task.id)

    return dependency_graph


def _score_candidate(
    employee: Employee,
    task: ExecutionTask,
    employee_workload: Dict[str, float],
    current_time_week: float,
) -> float:
    """
    Compute the skill-gap-penalty suitability score for an employee/task pair.

    Formula:
        SS = (skill_match * 0.6) + (utilization_score * 0.3) - (skill_gap_penalty * 0.2)

    Where:
        skill_match      = employee proficiency for the required skill (0.0 if absent)
        utilization_score = 1 - (accumulated_workload / current_time) — lower load = higher score
        skill_gap_penalty = max(0, required_level - actual_level)

    This never produces -inf so the engine can always pick the least-bad candidate.
    """
    actual_level = employee.skills.get(task.required_skill, 0.0)
    skill_gap_penalty = max(0.0, task.required_level - actual_level)

    # Guard division-by-zero: if simulation is at t=0 utilization is 0 by definition
    if current_time_week > _EPSILON:
        utilization_rate = employee_workload.get(employee.id, 0.0) / current_time_week
    else:
        utilization_rate = 0.0

    utilization_score = max(0.0, 1.0 - utilization_rate)

    return (actual_level * 0.6) + (utilization_score * 0.3) - (skill_gap_penalty * 0.2)


def assign_best_employee(
    task: ExecutionTask,
    employees: List[Employee],
    employee_workload: Dict[str, float],
    current_time_week: float,
) -> Optional[Employee]:
    """
    Selects the most suitable employee for a task using skill-gap-penalty scoring.

    Strategy (two-tier):
      Tier 1 – Perfect-match candidates (have the skill at or above required level).
               These are scored and the best is returned.
      Tier 2 – If no perfect match exists, ALL employees are scored with the
               skill-gap-penalty model and the least-bad candidate is chosen.
               This prevents operational stalls.

    Returns None only when the employee pool is completely empty.

    Args:
        task: The ExecutionTask to be assigned.
        employees: Pool of available employees.
        employee_workload: Accumulated workload (weeks) per employee ID.
        current_time_week: Current simulation progress in weeks.

    Returns:
        Optional[Employee]: Best candidate or None if no employees exist.
    """
    if not employees:
        return None

    # Tier 1: Exact-skill candidates (required level met)
    eligible_employees = [
        e for e in employees
        if e.skills.get(task.required_skill, 0.0) >= task.required_level
    ]

    # Tier 2: Skill-gap-penalty fallback — score everyone if no perfect match
    candidate_pool = eligible_employees if eligible_employees else employees

    best_candidate = max(
        candidate_pool,
        key=lambda e: _score_candidate(e, task, employee_workload, current_time_week)
    )
    return best_candidate


def simulate_project_execution(
    projects: List[ExecutionProject],
    employees: List[Employee],
) -> Dict[str, Any]:
    """
    Executes a deterministic simulation of multiple projects.

    Args:
        projects: List of projects to simulate.
        employees: List of employees available for work.

    Returns:
        Dict[str, Any]: Results containing duration_weeks, utilization map, and burnout_risk map.
    """
    if not employees:
        return {
            "duration_weeks": 0.0,
            "utilization": {},
            "burnout_risk": {},
        }

    # Track finish time and total hours for each employee
    employee_finish_time: Dict[str, float] = {e.id: 0.0 for e in employees}
    employee_total_hours: Dict[str, float] = {e.id: 0.0 for e in employees}

    task_completion_weeks: Dict[str, float] = {}
    simulation_duration = 0.0

    for project in projects:
        task_graph = build_task_graph(project)
        execution_order = list(nx.topological_sort(task_graph))

        for task_id in execution_order:
            task_data: ExecutionTask = task_graph.nodes[task_id]["data"]

            # Dependency Check
            dependency_finish_times = [
                task_completion_weeks.get(dep_id, 0.0)
                for dep_id in task_data.dependencies
            ]
            earliest_start_week = max(dependency_finish_times) if dependency_finish_times else 0.0

            # Assignment Logic (never raises – falls back to penalty model)
            assigned_employee = assign_best_employee(
                task_data, employees, employee_finish_time, simulation_duration
            )

            if not assigned_employee:
                # This can only occur if employees list is empty (guarded above)
                continue

            # Capacity guard: avoid division-by-zero for zero-capacity employees
            capacity = max(assigned_employee.capacity_hours_per_week, _EPSILON)
            weeks_required = task_data.estimated_hours / capacity

            task_start_week = max(earliest_start_week, employee_finish_time[assigned_employee.id])
            task_finish_week = task_start_week + weeks_required

            # State Update
            employee_finish_time[assigned_employee.id] = task_finish_week
            employee_total_hours[assigned_employee.id] += task_data.estimated_hours
            task_completion_weeks[task_data.id] = task_finish_week

            simulation_duration = max(simulation_duration, task_finish_week)

    # Post-Calculation: Utilization and Burnout
    utilization_report: Dict[str, float] = {}
    burnout_risk_report: Dict[str, str] = {}

    for employee in employees:
        total_potential_capacity = employee.capacity_hours_per_week * simulation_duration
        actual_hours_used = employee_total_hours[employee.id]

        usage_rate = (
            actual_hours_used / total_potential_capacity
            if total_potential_capacity > _EPSILON
            else 0.0
        )
        utilization_report[employee.name] = round(usage_rate, 2)

        if usage_rate > THRESHOLD_BURNOUT_HIGH:
            burnout_risk_report[employee.name] = "HIGH"
        elif usage_rate > THRESHOLD_BURNOUT_MEDIUM:
            burnout_risk_report[employee.name] = "MEDIUM"
        else:
            burnout_risk_report[employee.name] = "LOW"

    return {
        "duration_weeks": round(simulation_duration, 2),
        "utilization": utilization_report,
        "burnout_risk": burnout_risk_report,
    }
