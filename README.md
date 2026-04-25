# AgroManager

AgroManager is a hosted OpenReward environment for long-horizon UK arable farm management under live uncertainty. The project prioritizes a deployed environment first: stochastic weather, stochastic prices, weather-driven pest pressure, and noisy soil drift all happen while the agent is acting, and the baseline path uses the actual hosted environment `devatreya/AgroManager`.

The environment models a 400-acre Cambridgeshire farm split into 4 plots over 40 quarters. The main metric is `mean_terminal_score`, which balances solvency and final soil stewardship. The strongest scripted baseline is `weather_aware_rotation`, which adapts to live quarter-by-quarter price boards and recent weather history while protecting weak-soil plots and buying irrigation only when it is economically justified.

## Quickstart

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt

python scripts/fetch_weather.py
python scripts/fetch_prices.py
python scripts/build_tasks.py --scale medium
```

Run the local server for smoke checks:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

Run the hosted smoke test:

```bash
export OPENREWARD_API_KEY="..."
python scripts/test_deployed_env.py --openreward-env-id devatreya/AgroManager
```

Run hosted baselines:

```bash
python eval/run_baselines.py --split train --capture-conversation --openreward-env-id devatreya/AgroManager
python eval/run_baselines.py --split validation --capture-conversation --openreward-env-id devatreya/AgroManager
```

Prepare SFT data from hosted rollouts:

```bash
python scripts/prepare_sft_data.py --openreward-env-id devatreya/AgroManager
```

Run Modal jobs:

```bash
modal run pipeline/train_sft.py
modal run pipeline/train_rl.py --openreward-env-id devatreya/AgroManager
modal run pipeline/eval_compare.py --split validation --max-tasks 16 --openreward-env-id devatreya/AgroManager
modal run pipeline/eval_compare.py --split test --max-tasks 16 --openreward-env-id devatreya/AgroManager
```

## Project Layout

- `app.py`: canonical OpenReward server entrypoint
- `env.py`: hosted environment class `AgroManager`
- `sim.py`: simulator state, stochastic dynamics, reward logic
- `baselines/`: scripted policies, with `weather_aware_rotation` as the main benchmark
- `pipeline/`: hosted session cache, transcript compaction, rollouts, Modal training and eval scaffolds
- `scripts/`: data ingestion, task building, hosted smoke test, SFT prep
- `eval/`: hosted baseline runner
- `docs/`: design, deployment, training, and failure-mode notes
- `data/processed/`: processed Cambridge weather, DEFRA price anchors, and scenario tasks

## Data Sources

- Weather: Open-Meteo archive, aggregated to Cambridge quarterly climate context
- Prices: DEFRA agricultural price indices, chain-linked from the legacy ODS series to the current CSV series

`field_beans` uses a documented historical proxy chain-link before the modern `field_beans` series begins. There is no synthetic weather or price fallback path in the runtime.

## OpenReward Deployment

```bash
orwd create AgroManager \
  --description "Long-horizon UK arable farm benchmark with Qwen + Modal + ART" \
  --namespace devatreya

orwd link devatreya/AgroManager devatreya/AgroManager
orwd upload devatreya/AgroManager ./data/processed/

orwd deployments devatreya/AgroManager
orwd logs devatreya/AgroManager
```

## Notes

- The acceptance path is the hosted environment, not localhost.
- The simulator remains stochastic inside the episode; deterministic replay is not the primary benchmark path.
- `training/sft_result.json`, `training/rl_result.json`, and `eval/*_comparison.json` are written by the corresponding entrypoints.
