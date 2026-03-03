"""
Business Purpose:
Ensures the logical integrity and health of the organizational structure.
Validates governance rules (e.g., single director) and calculates a holistic 'Health Score'.

Technical Responsibility:
- Validate that the organization has a single top-level 'Director'.
- Check for orphan nodes or invalid reporting identifiers.
- Calculate a quantitative Health Score with specific penalty weights for risk factors.

Determinism Level:
High. All validation logic is rule-based and deterministic.

External Dependencies:
- None
"""

from typing import List, Dict
from core.models import Employee

# Business Constants: Health Penalties
PENALTY_SPAN_VIOLATION = 10
PENALTY_SKILL_RISK_HIGH = 15
PENALTY_SKILL_RISK_MEDIUM = 5
INITIAL_HEALTH_SCORE = 100

def validate_director_count(employees: List[Employee]) -> bool:
    """Verifies that exactly one 'Director' exists in the hierarchy."""
    directors = [emp for emp in employees if emp.role == "Director"]
    return len(directors) == 1


def validate_reporting_chain_integrity(employees: List[Employee]) -> bool:
    """Ensures every employee reports to an existing organizational ID."""
    organizational_ids = {emp.id for emp in employees}

    for employee in employees:
        if employee.reports_to and employee.reports_to not in organizational_ids:
            return False

    return True


def compute_aggregate_org_health(
    span_violations: List[str], 
    skill_risk_map: Dict[str, str]
) -> float:
    """
    Calculates a holistic health score (0-100) based on structural and skill risks.
    
    Args:
        span_violations: List of manager IDs with span-of-control issues.
        skill_risk_map: Mapping of skill names to risk levels (LOW/MEDIUM/HIGH).
        
    Returns:
        float: Normalized health score.
    """
    aggregate_score = INITIAL_HEALTH_SCORE

    # Structural penalties
    aggregate_score -= len(span_violations) * PENALTY_SPAN_VIOLATION

    # Talent risk penalties
    for risk_level in skill_risk_map.values():
        if risk_level == "HIGH":
            aggregate_score -= PENALTY_SKILL_RISK_HIGH
        elif risk_level == "MEDIUM":
            aggregate_score -= PENALTY_SKILL_RISK_MEDIUM

    return max(0.0, float(aggregate_score))