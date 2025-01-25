import toml
from typing import Any, Dict, List

from .funnel_models import FunnelCollection, FunnelDefinition, FunnelStepDefinition

def parse_funnel_defs_toml(toml_str: str) -> FunnelCollection:
    """
    Parse a TOML string representing funnel definitions.
    Example structure:

    [[funnels]]
    funnel_id = "user_onboarding"
    label = "User Onboarding Funnel"

    [[funnels.steps]]
    name = "Signup"
    metric_id = "signups"

    [[funnels.steps]]
    name = "Activation"
    metric_id = "activated_users"
    """
    data = toml.loads(toml_str)
    raw_funnels = data.get("funnels", [])

    funnel_defs = []
    for raw_f in raw_funnels:
        funnel_defs.append(_parse_single_funnel(raw_f))

    return FunnelCollection(funnels=funnel_defs)

def _parse_single_funnel(raw_f: Dict[str, Any]) -> FunnelDefinition:
    funnel_id = raw_f["funnel_id"]
    label = raw_f.get("label", funnel_id)

    steps_raw = raw_f.get("steps", [])
    steps = []
    for step in steps_raw:
        steps.append(FunnelStepDefinition(
            name=step["name"],
            metric_id=step["metric_id"]
        ))

    return FunnelDefinition(
        funnel_id=funnel_id,
        label=label,
        steps=steps
    )
