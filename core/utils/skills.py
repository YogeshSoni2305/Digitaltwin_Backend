"""
Business Purpose:
Analyzes the distribution and scarcity of technical skills within the organization.
Identifies "Single Points of Failure" and critical employees whose departure would impact operations.

Technical Responsibility:
- Aggregate skill proficiency across the entire employee pool.
- Calculate redundancy levels for specific technical domains.
- Map skills to qualitative risk levels (HIGH, MEDIUM, LOW).
- Identify "Criticality Scores" for individuals based on skill scarcity.

Determinism Level:
High. Calculations are based on fixed employee skill matrices.

External Dependencies:
- collections.defaultdict
"""

from typing import List, Dict, Set
from collections import defaultdict
from core.models import Employee

# Business Constants
PROFICIENCY_THRESHOLD_DEFAULT = 0.7
MINIMUM_REDUNDANCY_SAFE = 2

def compute_global_skill_coverage(employees: List[Employee]) -> Dict[str, List[float]]:
    """
    Aggregates all proficiency levels for every skill found in the organization.
    
    Returns:
        Dict[str, List[float]]: Mapping of skill names to a list of proficiency levels.
    """
    skill_coverage_map = defaultdict(list)

    for employee in employees:
        for skill_name, proficiency_level in employee.skills.items():
            skill_coverage_map[skill_name].append(proficiency_level)

    return dict(skill_coverage_map)


def compute_skill_redundancy(
    employees: List[Employee], 
    proficiency_threshold: float = PROFICIENCY_THRESHOLD_DEFAULT
) -> Dict[str, int]:
    """
    Counts how many employees exceed a proficiency threshold for each skill.
    
    Args:
        employees: The employee pool.
        proficiency_threshold: The level required to be considered 'redundant' support.
        
    Returns:
        Dict[str, int]: Count of proficient staff for each skill.
    """
    redundancy_counts = defaultdict(int)

    for employee in employees:
        for skill_name, proficiency_level in employee.skills.items():
            if proficiency_level >= proficiency_threshold:
                redundancy_counts[skill_name] += 1

    return dict(redundancy_counts)


def map_skills_to_risk_levels(
    redundancy_map: Dict[str, int], 
    safe_threshold: int = MINIMUM_REDUNDANCY_SAFE
) -> Dict[str, str]:
    """
    Categorizes skills into risk levels based on their redundancy.
    """
    skill_risk_report = {}
    for skill_name, count in redundancy_map.items():
        if count < safe_threshold:
            skill_risk_report[skill_name] = "HIGH"
        elif count == safe_threshold:
            skill_risk_report[skill_name] = "MEDIUM"
        else:
            skill_risk_report[skill_name] = "LOW"
    return skill_risk_report


def identify_critical_talent(
    employees: List[Employee], 
    redundancy_map: Dict[str, int], 
    proficiency_threshold: float = PROFICIENCY_THRESHOLD_DEFAULT
) -> Dict[str, int]:
    """
    Identifies employees who possess rare skills (redundancy <= 1) at a high proficiency.
    
    Returns:
        Dict[str, int]: Mapping of employee names to their 'Criticality Score' (count of rare skills).
    """
    critical_talent_map = {}

    for employee in employees:
        rarity_score = 0
        for skill_name, proficiency_level in employee.skills.items():
            # If the employee is proficient and they are the only (or one of very few) who are
            if proficiency_level >= proficiency_threshold and redundancy_map.get(skill_name, 0) <= 1:
                rarity_score += 1
        
        if rarity_score > 0:
            critical_talent_map[employee.name] = rarity_score

    return critical_talent_map