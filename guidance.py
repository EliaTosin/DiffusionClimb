# Backward-compat shim — real code is in ink_kin_stance.guidance
from ink_kin_stance.guidance import StabilityGuidance, ddim_sample_guided

__all__ = ["StabilityGuidance", "ddim_sample_guided"]
