from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

class StrategyType(str, Enum):
    BASELINE = "baseline"
    NO_REPLACE = "no_replace"
    IMMEDIATE = "immediate"
    DELAYED = "delayed"


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
    productivity_multiplier: float = 1.0

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