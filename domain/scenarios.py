
from domain.models import Scenario

def default_scenario() -> dict:
    return {"objective": "max_conv", "budget": 10000.0, "target": None, "increment": None}

def scenario_from_form(objective: str, budget: float | None, target: float | None, increment: float | None) -> dict:
    return {"objective": objective, "budget": budget, "target": target, "increment": increment}
