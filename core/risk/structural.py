"""
Business Purpose:
Analyzes the structural robustness of the organization's reporting hierarchy.
Identifies "Key Person" dependencies, high centralization risks, and span-of-control violations.

Technical Responsibility:
- Convert reporting lines into graph structures for network analysis.
- Calculate betweenness centrality to identify critical communication hubs.
- Implement Structural Caching using md5 organization hashing for sub-millisecond lookups.
- Aggregate qualitative risk categories (LOW, MEDIUM, HIGH).

Determinism Level:
High. Caching is based on a stable cryptographic hash of the organizational state.

External Dependencies:
- networkx
"""

import networkx as nx
import hashlib
import json
from typing import List, Dict, Any, Tuple
from core.models import Employee
from core.utils.structure import validate_span_of_control, build_reporting_map
from core.utils.skills import compute_skill_redundancy

# Internal Structural Cache
_STRUCTURAL_METRICS_CACHE: Dict[str, Dict[str, Any]] = {}

# Quantitative Thresholds
FRAGILITY_THRESHOLD_HIGH = 60.0
FRAGILITY_THRESHOLD_MEDIUM = 30.0

def _generate_organizational_digest(employees: List[Employee]) -> str:
    """
    Generates a stable cryptographic hash representing the organizational reporting 
    and skill structure to facilitate caching of expensive network analysis.
    
    Args:
        employees: The current employee pool.
        
    Returns:
        str: MD5 hash of the structure.
    """
    # Sort for deterministic hashing
    sorted_employees = sorted(employees, key=lambda x: x.id)
    canonical_repr = ""
    for employee in sorted_employees:
        canonical_repr += f"{employee.id}:{employee.reports_to}:"
        skills_token = ",".join(sorted(employee.skills.keys()))
        canonical_repr += f"{skills_token}|"
    
    return hashlib.md5(canonical_repr.encode()).hexdigest()


def compute_structural_fragility(employees: List[Employee]) -> Dict[str, Any]:
    """
    Performs a full network analysis to determine the organization's Structural Fragility Score.
    Uses caching to avoid re-computing for identical structures.
    
    Args:
        employees: List of employees to analyze.
        
    Returns:
        Dict[str, Any]: Quantitative fragility score and categorical risk level.
    """
    if not employees:
        return {
            "fragility_score": 0.0,
            "risk_level": "LOW",
            "components": {
                "centralization": 0.0, 
                "skill_concentration": 0.0, 
                "span_risk": 0.0, 
                "critical_dependency": 0.0
            }
        }

    structure_digest = _generate_organizational_digest(employees)
    
    # Cache lookup
    if structure_digest in _STRUCTURAL_METRICS_CACHE:
        return _STRUCTURAL_METRICS_CACHE[structure_digest]

    # 1. Network Centralization (30% weight)
    # Analyzes who sits on the most communication paths
    reporting_graph = nx.Graph()
    for emp in employees:
        reporting_graph.add_node(emp.id)
        if emp.reports_to:
            reporting_graph.add_edge(emp.reports_to, emp.id)
    
    centrality_map = nx.betweenness_centrality(reporting_graph)
    max_centralization = max(centrality_map.values()) if centrality_map else 0.0

    # 2. Skill Concentration (30% weight)
    # Measures risk of "Single Points of Failure" for specific skills
    skill_redundancy_map = compute_skill_redundancy(employees)
    if not skill_redundancy_map:
        skill_concentration_index = 0.0
    else:
        critical_skills_count = len([s for s, count in skill_redundancy_map.items() if count <= 1])
        skill_concentration_index = critical_skills_count / len(skill_redundancy_map)

    # 3. Span-of-Control Risk (20% weight)
    # Measures if managers are overloaded (affecting decision quality)
    management_hierarchy = build_reporting_map(employees)
    total_management_nodes = len(management_hierarchy)
    violations = validate_span_of_control(employees)
    span_risk_index = len(violations) / total_management_nodes if total_management_nodes > 0 else 0.0

    # 4. Critical Dependency (20% weight)
    # Measures the concentration of influence in the top 10% of the network
    all_centrality_values = sorted(centrality_map.values(), reverse=True)
    top_tier_count = max(1, int(len(employees) * 0.1))
    top_tier_centrality_avg = sum(all_centrality_values[:top_tier_count]) / top_tier_count if top_tier_count > 0 else 0.0

    # Composite quantitative score (0-100)
    aggregate_fragility_score = (
        (max_centralization * 0.3) +
        (skill_concentration_index * 0.3) +
        (span_risk_index * 0.2) +
        (top_tier_centrality_avg * 0.2)
    ) * 100.0
    
    final_score = round(min(aggregate_fragility_score, 100.0), 2)

    risk_result = {
        "fragility_score": final_score,
        "risk_level": (
            "HIGH" if final_score > FRAGILITY_THRESHOLD_HIGH 
            else "MEDIUM" if final_score > FRAGILITY_THRESHOLD_MEDIUM 
            else "LOW"
        ),
        "components": {
            "centralization": round(max_centralization, 3),
            "skill_concentration": round(skill_concentration_index, 3),
            "span_risk": round(span_risk_index, 3),
            "critical_dependency": round(top_tier_centrality_avg, 3)
        }
    }
    
    # Persistence in cache
    _STRUCTURAL_METRICS_CACHE[structure_digest] = risk_result
    return risk_result
