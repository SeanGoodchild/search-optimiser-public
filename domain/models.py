
from pydantic import BaseModel, Field
from typing import Literal, Optional

class Point(BaseModel):
    point_id: str
    cost: float = Field(ge=0)
    conv: float  = Field(ge=0)
    revenue: float = 0.0

class Entity(BaseModel):
    entity_id: str
    strategy: str
    points: list[Point]

Objective = Literal["max_conv","max_rev","min_cost_for_target"]

class Scenario(BaseModel):
    objective: Objective = "max_conv"
    budget: Optional[float] = None
    target: Optional[float] = None
    increment: Optional[float] = None
