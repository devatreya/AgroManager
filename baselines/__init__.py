from .conservative_rotation import decide as conservative_rotation
from .greedy_extractor import decide as greedy_extractor
from .weather_aware_rotation import decide as weather_aware_rotation

BASELINES = {
    "greedy_extractor": greedy_extractor,
    "conservative_rotation": conservative_rotation,
    "weather_aware_rotation": weather_aware_rotation,
}

__all__ = ["BASELINES", "greedy_extractor", "conservative_rotation", "weather_aware_rotation"]
