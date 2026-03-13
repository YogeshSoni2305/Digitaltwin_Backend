from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any
from enum import Enum

class StrategyType(str, Enum):
    BASELINE = "baseline"
    NO_REPLACE = "no_replace"
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    PRICE_INCREASE = "price_increase"
    RESTRUCTURE = "restructure"


class SimulationEngineError(Exception):
    """Raised when the simulation engine encounters a logical or runtime failure."""
    pass


class DecisionEngineError(Exception):
    """Raised when the decision matrix or ranking logic fails."""
    pass


class LLMEngineError(Exception):
    """Raised when the LLM service or isolation layer fails."""
    pass


class Skill(BaseModel):
    name: str
    level: float  # 0–1


class Employee(BaseModel):
    id: str
    name: str
    role: str
    department: str
    salary: float
    skills: Dict[str, float] = Field(default_factory=dict)
    capacity_hours_per_week: float
    reports_to: Optional[str] = None
    productivity_multiplier: float = Field(default=1.0, ge=0.1)
class Task(BaseModel):
    id: str
    name: str
    required_skills: Dict[str, float] = Field(default_factory=dict)
    duration_days: int
    assigned_to: Optional[str] = None


class Project(BaseModel):
    id: str
    name: str
    budget: float
    revenue: float
    tasks: List[Task] = Field(default_factory=list)

class ExecutionTask(BaseModel):
    id: str
    name: str
    required_skill: str
    required_level: float
    estimated_hours: float
    dependencies: List[str] = Field(default_factory=list)


class ExecutionProject(BaseModel):
    id: str
    name: str
    tasks: List[ExecutionTask] = Field(default_factory=list)
    base_revenue: float
    decay_rate: float  # market sensitivity


# =====================================================
# Phase 2: Type Safe Response Models
# =====================================================

class SimulationResult(BaseModel):
    duration_weeks: float
    utilization: Dict[str, float]
    burnout_risk: Dict[str, str]
    project_completion_times: Dict[str, float]


class FinanceResult(BaseModel):
    revenue: float
    cost: float
    profit: float


class StructuralComponents(BaseModel):
    centralization: float
    influence_concentration: float
    clustering_cohesion: float
    silo_score: float
    skill_concentration: float
    span_risk: float
    critical_dependency: float


class StructuralRiskResult(BaseModel):
    fragility_score: float
    risk_level: str
    components: StructuralComponents


class BehavioralRiskResult(BaseModel):
    behavioral_fragility_index: float
    average_attrition_probability: float
    average_knowledge_loss_index: float
    burnout_map: Dict[str, float]


class MonteCarloResult(BaseModel):
    model_config = ConfigDict(extra='allow')
    mean_profit: float = 0.0
    p5_profit: float = 0.0
    p95_profit: float = 0.0
    profit_variance: float = 0.0
    probability_of_loss: float = 0.0
    stability_score: float = 0.0
    seed_used: Optional[int] = None
    mean_structural_fragility: float = 0.0
    mean_behavioral_fragility: float = 0.0


class RiskMetrics(BaseModel):
    average_profit: float
    p5_profit: float
    p95_profit: float
    profit_variance: float
    stability_score: float
    risk_probability: float
    mean_structural_fragility: float
    mean_behavioral_fragility: float
    seed_used: Optional[int]


class OrganizationMetrics(BaseModel):
    health_score: float
    structural_fragility_score: float
    behavioral_fragility_index: float
    risk_level: str


class GovernanceMetrics(BaseModel):
    model_version: Optional[str]
    timestamp: str


class ServiceResponse(BaseModel):
    execution: SimulationResult
    financial: FinanceResult
    risk: RiskMetrics
    organization: OrganizationMetrics
    governance: GovernanceMetrics


class RankedStrategy(BaseModel):
    strategy: StrategyType
    decision_score: float
    is_dominant: bool
    recommendation_justification: List[str]
    raw_metrics: Dict[str, Any]
    rank: int


class RankingEngineResult(BaseModel):
    ranked_strategies: List[RankedStrategy]
    governance: Dict[str, Any]


class DecisionResult(BaseModel):
    best_strategy: StrategyType
    comparison_matrix: List[RankedStrategy]
    ranked_strategies: List[RankedStrategy]
    governance: Dict[str, Any]


class ExplanationResult(BaseModel):
    best_strategy: str
    rankings: List[RankedStrategy]
    executive_summary: str
    winning_data_snapshot: Dict[str, Any]