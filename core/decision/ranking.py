"""
Business Purpose:
Ranks and recommends HR strategies based on a multi-criteria decision model.
Evaluates trade-offs between financial return, organizational health, and various risk indices.

Technical Responsibility:
- Implement Multi-Attribute Utility Theory (MAUT) for strategy scoring.
- Detect Pareto Dominance to identify mathematically superior scenarios.
- Generate qualitative "reasoning" strings for board-level reporting.
- Support dynamic weighting of decision criteria.

Determinism Level:
High. Ranking is reproducible for any given set of scenario metrics and weights.

External Dependencies:
- None (Pure logic)
"""

from typing import List, Dict, Any, Tuple

def normalize_metric(value: float, min_val: float, max_val: float, invert: bool = False) -> float:
    """
    Normalizes a numerical value to a standard 0.0-1.0 scale.
    
    Args:
        value: The raw metric value.
        min_val: Minimum observed value in the sample set.
        max_val: Maximum observed value in the sample set.
        invert: If True, higher raw values result in lower normalized scores (e.g., for risks).
        
    Returns:
        float: Normalized score between 0.0 and 1.0.
    """
    if max_val == min_val:
        return 0.5 # Neutral baseline for zero variance
    
    normalized_value = (value - min_val) / (max_val - min_val)
    return (1.0 - normalized_value) if invert else normalized_value


def compute_aggregate_decision_score(
    normalized_metrics: Dict[str, float],
    importance_weights: Dict[str, float]
) -> float:
    """
    Computes a composite score (0-100) using weighted linear aggregation.
    
    Args:
        normalized_metrics: Map of normalized metric keys to 0-1 values.
        importance_weights: Map of importance weights for each attribute.
        
    Returns:
        float: Aggregate score scaled to 0-100.
    """
    total_weighted_score = 0.0
    
    # Mapping of simulation output keys to configuration weight keys
    WEIGHT_KEY_MAPPING = {
        "profit": "profit",
        "org_health": "org_health",
        "volatility": "volatility",
        "burnout_index": "burnout",
        "fragility_score": "structural_fragility",
        "behavioral_fragility_index": "behavioral_fragility",
        "stability_score": "stability"
    }
    
    # Polarities: Impact on final score (1 for beneficial, -1 for risk/detrimental)
    POLARITY_MAPPING = {
        "profit": 1,
        "org_health": 1,
        "volatility": -1,
        "burnout_index": -1,
        "fragility_score": -1,
        "behavioral_fragility_index": -1,
        "stability_score": 1
    }

    for metric_key, normalized_val in normalized_metrics.items():
        weight_lookup_key = WEIGHT_KEY_MAPPING.get(metric_key)
        if weight_lookup_key in importance_weights:
            attribute_weight = importance_weights[weight_lookup_key]
            polarity = POLARITY_MAPPING.get(metric_key, 1)
            total_weighted_score += normalized_val * attribute_weight * polarity
    
    return round(total_weighted_score * 100, 2)


def detect_dominance_matrix(scenarios: List[Dict[str, Any]]) -> List[bool]:
    """
    Identifies if a strategy is Pareto Dominant (no other strategy is better across 
    all dimensions while being at least as good in one).
    """
    dominance_flags = [False] * len(scenarios)
    
    for i, candidate in enumerate(scenarios):
        is_truly_dominant = True
        for j, rival in enumerate(scenarios):
            if i == j: continue
            
            # Dominance check (Is the rival strictly better or equal in everything?)
            # Benefit criteria (Higher is better)
            benefit_ok = (
                candidate.get('profit', 0) >= rival.get('profit', 0) and
                candidate.get('org_health', 0) >= rival.get('org_health', 0) and
                candidate.get('stability_score', 0) >= rival.get('stability_score', 0)
            )
            # Risk criteria (Lower is better)
            risk_ok = (
                candidate.get('volatility', 0) <= rival.get('volatility', 0) and
                candidate.get('burnout_index', 0) <= rival.get('burnout_index', 0) and
                candidate.get('fragility_score', 0) <= rival.get('fragility_score', 0) and
                candidate.get('behavioral_fragility_index', 0) <= rival.get('behavioral_fragility_index', 0)
            )
            
            if not (benefit_ok and risk_ok):
                is_truly_dominant = False
                break
        
        dominance_flags[i] = is_truly_dominant
    
    return dominance_flags


def rank_strategies(
    scenarios: List[Dict[str, Any]], 
    decision_weights: Dict[str, float]
) -> Dict[str, Any]:
    """
    Ranks a set of simulated strategies and provides qualitative justification.
    
    Args:
        scenarios: List of strategy outcome dictionaries.
        decision_weights: Configuration defining the priority of each metric.
        
    Returns:
        Dict[str, Any]: Ranked list with detailed scoring and reasoning.
    """
    if not scenarios:
        return {"ranked_strategies": [], "governance": {"status": "NO_DATA"}}

    dominance_indicators = detect_dominance_matrix(scenarios)

    # Calculate global min/max for normalization bounds
    metric_keys = [
        'profit', 'org_health', 'volatility', 'burnout_index', 
        'fragility_score', 'behavioral_fragility_index', 'stability_score'
    ]
    global_bounds = {}
    for key in metric_keys:
        values = [s.get(key, 0.0) for s in scenarios if key in s]
        if not values: values = [0.0]
        global_bounds[key] = {'min': min(values), 'max': max(values)}

    processed_rankings = []
    for i, scenario in enumerate(scenarios):
        # Normalize local metrics against global bounds
        normalized_profile = {}
        for key in metric_keys:
            if key in scenario:
                normalized_profile[key] = normalize_metric(
                    scenario[key], global_bounds[key]['min'], global_bounds[key]['max']
                )
        
        # Calculate utility score
        composite_score = compute_aggregate_decision_score(normalized_profile, decision_weights)
        
        # Generate Narrative Reasoning
        narrative_justification = []
        if scenario.get('profit', 0) == global_bounds['profit']['max']: 
            narrative_justification.append("Optimal financial throughput.")
        if scenario.get('org_health', 0) > 85: 
            narrative_justification.append("Strong institutional health preservation.")
        if scenario.get('behavioral_fragility_index', 100) < 25: 
            narrative_justification.append("Resilient behavioral profile.")
        if scenario.get('stability_score', 0) > 0.9: 
            narrative_justification.append("Predictable stochastic behavior.")
        if dominance_indicators[i]: 
            narrative_justification.append("Directly Dominant Strategy.")
        
        # Risk Warnings
        if scenario.get('behavioral_fragility_index', 0) > 60: 
            narrative_justification.append("CRITICAL: High risk of organizational burnout contagion.")

        processed_rankings.append({
            "strategy": scenario['strategy'],
            "decision_score": composite_score,
            "is_dominant": dominance_indicators[i],
            "recommendation_justification": narrative_justification,
            "raw_metrics": scenario
        })

    # Sort by aggregate utility
    processed_rankings.sort(key=lambda x: x['decision_score'], reverse=True)
    
    # Assign ordinal ranks
    for index, item in enumerate(processed_rankings):
        item['rank'] = index + 1

    return {
        "ranked_strategies": processed_rankings,
        "governance": {
            "applied_weights": decision_weights,
            "stability_informed": True,
            "pareto_checked": True
        }
    }
