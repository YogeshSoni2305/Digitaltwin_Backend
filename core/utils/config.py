from pydantic import BaseModel

class SimulationConfig(BaseModel):
    task_variance: float = 0.1
    market_volatility: float = 0.05
    attrition_shock_impact: float = 0.10
    attrition_probs: dict = {
        "LOW": 0.02,
        "MEDIUM": 0.08,
        "HIGH": 0.25
    }