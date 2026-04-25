# Environment Design

AgroManager exposes one environment class, `AgroManager`, and one server entrypoint, `app.py`. The agent manages 4 plots of 100 acres each over 40 quarters. Each quarter has one farm-level capital decision and one crop-input plan per plot.

The dense reward is quarterly P&L scaled by `1e-4` plus soil shaping for plots above soil health `0.55`. The terminal score is the solvency-adjusted cash ratio multiplied by a final-soil factor. The environment is intentionally long-horizon: short-run extraction can look profitable while damaging the final score through soil and bankruptcy risk.

The five tools are:

- `read_farm_state`
- `read_soil_report`
- `read_weather_history`
- `read_price_board`
- `commit_plan`

Tool metadata always includes `tool` and `state`. `commit_plan` also includes `step` and `episode_metrics`.
