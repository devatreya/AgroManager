from .weather_aware_rotation import decide as weather_aware_rotation

BASELINES = {
    "weather_aware_rotation": weather_aware_rotation,
}

__all__ = ["BASELINES", "weather_aware_rotation"]
