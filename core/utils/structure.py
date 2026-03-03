"""
Business Purpose:
Maps the organizational reporting lines to facilitate network and hierarchical analysis.
Identifies management bottlenecks via span-of-control validation.

Technical Responsibility:
- Transform flat employee lists into reporting maps (Parent -> Children).
- Detect managers whose direct reporting line exceeds operational efficiency thresholds.

Determinism Level:
High. The mapping is a direct mathematical derivative of the input employee state.

External Dependencies:
- collections.defaultdict
"""

from typing import List, Dict
from collections import defaultdict
from core.models import Employee

# Business Constants
MAX_EFFICIENT_SPAN_OF_CONTROL = 7

def build_reporting_map(employees: List[Employee]) -> Dict[str, List[str]]:
    """
    Constructs a dictionary mapping manager IDs to lists of their direct reports' IDs.
    
    Args:
        employees: The employee pool.
        
    Returns:
        Dict[str, List[str]]: Reporting index (Manager ID -> [Report IDs]).
    """
    reporting_index = defaultdict(list)

    for employee in employees:
        if employee.reports_to:
            reporting_index[employee.reports_to].append(employee.id)

    return dict(reporting_index)


def validate_span_of_control(
    employees: List[Employee], 
    threshold: int = MAX_EFFICIENT_SPAN_OF_CONTROL
) -> Dict[str, int]:
    """
    Identifies managers who have too many direct reports, potentially causing operational friction.
    
    Args:
        employees: The employee pool.
        threshold: The maximum number of direct reports before a violation is flagged.
        
    Returns:
        Dict[str, int]: Mapping of Manager IDs to their actual report count (violations only).
    """
    reporting_map = build_reporting_map(employees)
    span_violations_report = {}

    for manager_id, report_list in reporting_map.items():
        actual_count = len(report_list)
        if actual_count > threshold:
            span_violations_report[manager_id] = actual_count

    return span_violations_report