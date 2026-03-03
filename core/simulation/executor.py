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

import networkx as nx
from typing import List, Dict, Optional, Any
from core.models import Employee, ExecutionProject, ExecutionTask

# Business Constants
THRESHOLD_BURNOUT_HIGH = 0.90
THRESHOLD_BURNOUT_MEDIUM = 0.75

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


def assign_best_employee(
    task: ExecutionTask, 
    employees: List[Employee], 
    employee_workload: Dict[str, float], 
    current_time_week: float
) -> Optional[Employee]:
    """
    Selects the most suitable employee for a task based on skill proficiency and current workload.
    
    Args:
        task: The ExecutionTask to be assigned.
        employees: List of available employees.
        employee_workload: Current cumulative workload (in weeks) for each employee ID.
        current_time_week: Current progress of the simulation.
        
    Returns:
        Optional[Employee]: The best candidate or None if no eligible staff found.
    """
    eligible_employees = [
        employee for employee in employees
        if task.required_skill in employee.skills
        and employee.skills[task.required_skill] >= task.required_level
    ]

    if not eligible_employees:
        return None

    best_candidate = None
    max_suitability_score = -float("inf")

    for employee in eligible_employees:
        skill_proficiency = employee.skills[task.required_skill]

        # Calculate approximate utilization based on total time elapsed
        utilization_rate = (employee_workload[employee.id] / current_time_week) if current_time_week > 0 else 0

        # Balanced Suitability Score (60% Skill, 40% Load Balance)
        suitability_score = (skill_proficiency * 0.6) - (utilization_rate * 0.4)

        if suitability_score > max_suitability_score:
            max_suitability_score = suitability_score
            best_candidate = employee

    return best_candidate


def simulate_project_execution(
    projects: List[ExecutionProject], 
    employees: List[Employee]
) -> Dict[str, Any]:
    """
    Executes a deterministic simulation of multiple projects.
    
    Args:
        projects: List of projects to simulate.
        employees: List of employees available for work.
        
    Returns:
        Dict[str, Any]: Results containing duration_weeks, utilization map, and burnout_risk map.
    """
    # Track finish time and total hours for each employee
    employee_finish_time = {e.id: 0.0 for e in employees}
    employee_total_hours = {e.id: 0.0 for e in employees}

    task_completion_weeks = {}
    simulation_duration = 0.0

    for project in projects:
        task_graph = build_task_graph(project)
        execution_order = list(nx.topological_sort(task_graph))

        for task_id in execution_order:
            task_data = task_graph.nodes[task_id]["data"]

            # Dependency Check
            dependency_finish_times = [
                task_completion_weeks.get(dep_id, 0.0)
                for dep_id in task_data.dependencies
            ]
            earliest_start_week = max(dependency_finish_times) if dependency_finish_times else 0.0

            # Assignment Logic
            assigned_employee = assign_best_employee(
                task_data, employees, employee_finish_time, simulation_duration
            )

            if not assigned_employee:
                raise ValueError(f"Operational Stall: No eligible employee for task '{task_data.name}' (Required Skill: {task_data.required_skill})")

            # Timing Calculation
            weeks_required = task_data.estimated_hours / assigned_employee.capacity_hours_per_week
            task_start_week = max(earliest_start_week, employee_finish_time[assigned_employee.id])
            task_finish_week = task_start_week + weeks_required

            # State Update
            employee_finish_time[assigned_employee.id] = task_finish_week
            employee_total_hours[assigned_employee.id] += task_data.estimated_hours
            task_completion_weeks[task_data.id] = task_finish_week

            simulation_duration = max(simulation_duration, task_finish_week)

    # Post-Calculation: Utilization and Burnout
    utilization_report = {}
    burnout_risk_report = {}

    for employee in employees:
        total_potential_capacity = employee.capacity_hours_per_week * simulation_duration
        actual_hours_used = employee_total_hours[employee.id]

        usage_rate = actual_hours_used / total_potential_capacity if total_potential_capacity > 0 else 0.0
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
        "burnout_risk": burnout_risk_report
    }
