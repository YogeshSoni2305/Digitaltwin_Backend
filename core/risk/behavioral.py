"""
Business Purpose:
Models the "human" factor of organizational risk, specifically burnout contagion and attrition.
Quantifies the hidden costs of productivity friction and institutional knowledge loss.

Technical Responsibility:
- Implement a Network Contagion Model for burnout propagation.
- Forecast attrition probability based on individual and structural factors.
- Model ramp-up productivity curves for new hires.
- Quantify Knowledge Loss based on network centrality and skill uniqueness.

Determinism Level:
High. All behavioral models use deterministic weight-based propagation.

External Dependencies:
- networkx
"""

import networkx as nx
from typing import List, Dict, Any, Optional
from core.models import Employee

# Business Constants: Contagion & Attrition
CONTAGION_FACTOR_DEFAULT = 0.3
ATTRITION_PROB_LOW = 0.02
ATTRITION_PROB_MEDIUM = 0.08
ATTRITION_PROB_HIGH = 0.25

# Cost Constants
REPLACEMENT_COST_MULTIPLIER = 0.5  # 50% of annual salary

def compute_burnout_contagion(
    organizational_network: nx.Graph, 
    utilization_matrix: Dict[str, float], 
    contagion_weight: float = CONTAGION_FACTOR_DEFAULT
) -> Dict[str, float]:
    """
    Computes a network-aware burnout index using a contagion propagation model.
    Formula: burnout = current_utilization + (contagion_weight * average_neighbor_burnout)
    
    Args:
        organizational_network: Undirected graph of reporting and collaborative ties.
        utilization_matrix: Baseline utilization scores (0.0 to 1.0).
        contagion_weight: Degree to which burnout spreads from peers.
        
    Returns:
        Dict[str, float]: Mapping of employee IDs to propagated burnout scores.
    """
    propagated_burnout = {node: utilization_matrix.get(node, 0.0) for node in organizational_network.nodes()}
    
    # One-pass propagation for deterministic stability
    final_burnout_scores = {}
    for node in organizational_network.nodes():
        peers = list(organizational_network.neighbors(node))
        if not peers:
            final_burnout_scores[node] = propagated_burnout[node]
            continue
            
        peer_burnout_sum = sum(propagated_burnout.get(peer, 0.0) for peer in peers)
        avg_peer_burnout = peer_burnout_sum / len(peers)
        
        # Compound utilization with contagion pressure
        burnout_score = propagated_burnout[node] + (contagion_weight * avg_peer_burnout)
        final_burnout_scores[node] = max(0.0, min(1.0, burnout_score))
        
    return final_burnout_scores


def calculate_individual_attrition_risk(
    burnout_index: float, 
    network_centrality: float, 
    skill_redundancy_count: float
) -> float:
    """
    Forecasts the probability of exit based on individual stress and structural isolation.
    
    Weights: 50% Burnout, 30% Centrality (Burden), 20% Skill Scarcity.
    """
    effective_redundancy = max(skill_redundancy_count, 1.0)
    scarcity_index = 1.0 / effective_redundancy
    
    raw_risk = (burnout_index * 0.5) + (network_centrality * 0.3) + (scarcity_index * 0.2)
    return max(0.0, min(1.0, raw_risk))


def compute_knowledge_loss_impact(
    network_centrality: float, 
    skill_redundancy_count: float
) -> float:
    """
    Quantifies the institutional impact of an employee's departure.
    High impact occurs when a central figure with unique skills leaves.
    """
    effective_redundancy = max(skill_redundancy_count, 1.0)
    scarcity_impact = 1.0 / effective_redundancy
    
    # 60% Centrality (Flow impact), 40% Scarcity (Replacement difficulty)
    impact_index = (network_centrality * 0.6) + (scarcity_impact * 0.4)
    return max(0.0, min(1.0, impact_index))


def compute_behavioral_fragility(
    employees: List[Employee], 
    organizational_network: nx.Graph, 
    utilization_matrix: Dict[str, float], 
    centrality_matrix: Dict[str, float], 
    redundancy_matrix: Dict[str, float]
) -> Dict[str, Any]:
    """
    Aggregates granular behavioral signals into a high-level organizational index.
    
    Args:
        employees: The employee pool.
        organizational_network: Network graph.
        utilization_matrix: Employee utilization data.
        centrality_matrix: Node centrality scores.
        redundancy_matrix: Skill frequency across the org.
        
    Returns:
        Dict[str, Any]: Behavioral analytics including the fragility index.
    """
    burnout_map = compute_burnout_contagion(organizational_network, utilization_matrix)
    
    cumulative_attrition_probability = 0.0
    cumulative_knowledge_loss = 0.0
    
    for emp in employees:
        emp_id = emp.id
        burnout_val = burnout_map.get(emp_id, 0.0)
        centrality_val = centrality_matrix.get(emp_id, 0.0)
        
        # Determine specific skill redundancy (lowest redundancy among held skills)
        skills = list(emp.skills.keys())
        min_redundancy = min([redundancy_matrix.get(s, 1.0) for s in skills]) if skills else 1.0
            
        attrition_prob = calculate_individual_attrition_risk(burnout_val, centrality_val, min_redundancy)
        knowledge_impact = compute_knowledge_loss_impact(centrality_val, min_redundancy)
        
        cumulative_attrition_probability += attrition_prob
        cumulative_knowledge_loss += knowledge_impact
        
    avg_attrition = cumulative_attrition_probability / len(employees) if employees else 0.0
    avg_loss = cumulative_knowledge_loss / len(employees) if employees else 0.0
    
    # Behavioral Fragility Index (0-100)
    fragility_index = (avg_attrition * 60) + (avg_loss * 40)
    
    return {
        "behavioral_fragility_index": round(fragility_index, 2),
        "average_attrition_probability": round(avg_attrition, 3),
        "average_knowledge_loss_index": round(avg_loss, 3),
        "burnout_map": burnout_map
    }


# --- Legacy / Categorical Helpers (Merged from core/risk.py) ---

def convert_burnout_to_attrition_probability(burnout_categories: Dict[str, str]) -> Dict[str, float]:
    """Business logic for categorical-to-probabilistic mapping."""
    mapping = {"LOW": ATTRITION_PROB_LOW, "MEDIUM": ATTRITION_PROB_MEDIUM, "HIGH": ATTRITION_PROB_HIGH}
    return {name: mapping.get(level, ATTRITION_PROB_LOW) for name, level in burnout_categories.items()}


def compute_weighted_attrition_cost(attrition_probabilities: Dict[str, float], employees: List[Employee]) -> float:
    """Calculates currency-based risk exposure for employee turnover."""
    total_expected_exposure = 0.0
    for emp in employees:
        prob = attrition_probabilities.get(emp.id, attrition_probabilities.get(emp.name, 0.0))
        potential_cost = emp.salary * REPLACEMENT_COST_MULTIPLIER
        total_expected_exposure += prob * potential_cost
    return round(total_expected_exposure, 2)
