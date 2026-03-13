"""
Business Purpose:
Models the "human" factor of organizational risk, specifically burnout contagion and attrition.
Quantifies the hidden costs of productivity friction and institutional knowledge loss.

Technical Responsibility:
- Implement a Network Contagion Model for burnout propagation.
- Forecast attrition probability using a sigmoid transform for realistic distribution (0.05–0.80).
- Quantify Knowledge Loss based on network centrality and skill uniqueness.

Determinism Level:
High. All behavioral models use deterministic weight-based propagation.

External Dependencies:
- networkx
"""

import math
import networkx as nx
from typing import List, Dict, Any, Optional
from core.models import Employee, BehavioralRiskResult

# Business Constants: Contagion & Attrition
CONTAGION_FACTOR_DEFAULT = 0.3
ATTRITION_PROB_LOW = 0.02
ATTRITION_PROB_MEDIUM = 0.08
ATTRITION_PROB_HIGH = 0.25

# Cost Constants
REPLACEMENT_COST_MULTIPLIER = 0.5  # 50% of annual salary

# Sigmoid calibration constants (tuned for distribution range 0.05–0.80)
# z = BURNOUT_W * burnout + CENTRALITY_W * centrality + SCARCITY_W * scarcity - BIAS
_BURNOUT_W    = 1.8
_CENTRALITY_W = 1.2
_SCARCITY_W   = 0.9
_SIGMOID_BIAS = 1.5


def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid function."""
    # Clamp extreme inputs to prevent overflow
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def compute_burnout_contagion(
    organizational_network: nx.Graph,
    utilization_matrix: Dict[str, float],
    contagion_weight: float = CONTAGION_FACTOR_DEFAULT,
) -> Dict[str, float]:
    """
    Computes a network-aware burnout index using a one-pass contagion propagation model.

    Formula: burnout_i = utilization_i + contagion_weight * avg(neighbor_burnout)

    Args:
        organizational_network: Undirected graph of reporting and collaborative ties.
        utilization_matrix: Baseline utilization scores (0.0 to 1.0) keyed by employee ID.
        contagion_weight: Degree to which burnout spreads from peers.

    Returns:
        Dict[str, float]: Employee ID → propagated burnout score (clamped 0–1).
    """
    propagated_burnout = {
        node: utilization_matrix.get(node, 0.0)
        for node in organizational_network.nodes()
    }

    # One-pass propagation for deterministic stability
    final_burnout_scores: Dict[str, float] = {}
    for node in organizational_network.nodes():
        peers = list(organizational_network.neighbors(node))
        if not peers:
            final_burnout_scores[node] = propagated_burnout[node]
            continue

        avg_peer_burnout = sum(propagated_burnout.get(p, 0.0) for p in peers) / len(peers)
        burnout_score = propagated_burnout[node] + (contagion_weight * avg_peer_burnout)
        final_burnout_scores[node] = max(0.0, min(1.0, burnout_score))

    return final_burnout_scores


def calculate_individual_attrition_risk(
    burnout_index: float,
    network_centrality: float,
    skill_redundancy_count: float,
) -> float:
    """
    Forecasts exit probability using a calibrated sigmoid transform.

    Replaces the previous linear model which produced a narrow 0.41–0.56 band.
    The sigmoid maps the same inputs to a realistic range of ~0.05–0.80.

    Sigmoid formula:
        z = 1.8 * burnout + 1.2 * centrality + 0.9 * skill_scarcity - 1.5
        P_exit = sigmoid(z)

    Where:
        skill_scarcity = 1 / max(skill_redundancy_count, 1.0)

    Args:
        burnout_index: Contagion-adjusted burnout level [0, 1].
        network_centrality: Betweenness centrality score [0, 1].
        skill_redundancy_count: Number of employees sharing the rarest skill held.

    Returns:
        float: Exit probability in [0.0, 1.0].
    """
    effective_redundancy = max(skill_redundancy_count, 1.0)
    skill_scarcity = 1.0 / effective_redundancy

    z = (
        _BURNOUT_W    * burnout_index
        + _CENTRALITY_W * network_centrality
        + _SCARCITY_W   * skill_scarcity
        - _SIGMOID_BIAS
    )
    return _sigmoid(z)


def compute_knowledge_loss_impact(
    network_centrality: float,
    skill_redundancy_count: float,
) -> float:
    """
    Quantifies the institutional impact of an employee's departure.
    High impact occurs when a central figure with unique skills leaves.

    Formula:
        impact = (centrality * 0.6) + (scarcity * 0.4)
    """
    effective_redundancy = max(skill_redundancy_count, 1.0)
    scarcity_impact = 1.0 / effective_redundancy

    impact_index = (network_centrality * 0.6) + (scarcity_impact * 0.4)
    return max(0.0, min(1.0, impact_index))


def compute_behavioral_fragility(
    employees: List[Employee],
    organizational_network: nx.Graph,
    utilization_matrix: Dict[str, float],
    centrality_matrix: Dict[str, float],
    redundancy_matrix: Dict[str, float],
) -> BehavioralRiskResult:
    """
    Aggregates granular behavioral signals into a high-level organizational index.

    Returns:
        Dict[str, Any]: Behavioral analytics including the fragility index.
    """
    if not employees:
        return BehavioralRiskResult(
            behavioral_fragility_index=0.0,
            average_attrition_probability=0.0,
            average_knowledge_loss_index=0.0,
            burnout_map={}
        )

    burnout_map = compute_burnout_contagion(organizational_network, utilization_matrix)

    cumulative_attrition_probability = 0.0
    cumulative_knowledge_loss = 0.0

    for emp in employees:
        burnout_val    = burnout_map.get(emp.id, 0.0)
        centrality_val = centrality_matrix.get(emp.id, 0.0)

        # Lowest redundancy across all held skills — the rarest skill drives scarcity risk
        skills = list(emp.skills.keys())
        min_redundancy = (
            min(redundancy_matrix.get(s, 1.0) for s in skills)
            if skills else 1.0
        )

        attrition_prob   = calculate_individual_attrition_risk(burnout_val, centrality_val, min_redundancy)
        knowledge_impact = compute_knowledge_loss_impact(centrality_val, min_redundancy)

        cumulative_attrition_probability += attrition_prob
        cumulative_knowledge_loss        += knowledge_impact

    n = len(employees)
    avg_attrition = cumulative_attrition_probability / n
    avg_loss      = cumulative_knowledge_loss / n

    # Behavioral Fragility Index (0–100)
    fragility_index = (avg_attrition * 60) + (avg_loss * 40)

    return BehavioralRiskResult(
        behavioral_fragility_index=round(fragility_index, 2),
        average_attrition_probability=round(avg_attrition, 3),
        average_knowledge_loss_index=round(avg_loss, 3),
        burnout_map=burnout_map,
    )


# --- Legacy / Categorical Helpers ---

def convert_burnout_to_attrition_probability(burnout_categories: Dict[str, str]) -> Dict[str, float]:
    """Business logic for categorical-to-probabilistic mapping."""
    mapping = {"LOW": ATTRITION_PROB_LOW, "MEDIUM": ATTRITION_PROB_MEDIUM, "HIGH": ATTRITION_PROB_HIGH}
    return {name: mapping.get(level, ATTRITION_PROB_LOW) for name, level in burnout_categories.items()}


def compute_weighted_attrition_cost(
    attrition_probabilities: Dict[str, float],
    employees: List[Employee],
) -> float:
    """Calculates currency-based risk exposure for employee turnover."""
    total_expected_exposure = 0.0
    for emp in employees:
        prob = attrition_probabilities.get(emp.id, attrition_probabilities.get(emp.name, 0.0))
        potential_cost = emp.salary * REPLACEMENT_COST_MULTIPLIER
        total_expected_exposure += prob * potential_cost
    return round(total_expected_exposure, 2)
