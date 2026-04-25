# Modal and ART

The Modal runtime uses Python `3.12`, Debian slim, installs `git` and `procps`, mounts:

- `/vol/art`
- `/vol/hf-cache`
- `/vol/results`

The base model is `Qwen/Qwen2.5-7B-Instruct`.

`pipeline/train_sft.py` runs SFT on Modal via ART local backend and writes `training/sft_result.json`.

`pipeline/train_rl.py` runs RL against the hosted OpenReward environment. Shared ART mode is the default. Dedicated mode is exposed as an opt-in.

`pipeline/eval_compare.py` compares the trained model to `weather_aware_rotation` on hosted tasks and writes `eval/validation_comparison.json` or `eval/test_comparison.json`.

## Frozen Teacher Reference

`weather_aware_rotation` is the frozen scripted teacher for the SFT phase. The last hosted reference sweep before SFT used the live environment `devatreya/AgroManager` with a `12/4/4` sample:

- train `12`: mean terminal cash `1,252,745.52`, mean final soil `0.3555`, bankruptcy rate `0.0`, completion rate `1.0`, mean total reward `114.8662`, mean terminal score `0.1086`
- validation `4`: mean terminal cash `1,308,289.33`, mean final soil `0.3939`, bankruptcy rate `0.0`, completion rate `1.0`, mean total reward `119.9789`, mean terminal score `0.2619`
- test `4`: mean terminal cash `1,274,194.27`, mean final soil `0.4057`, bankruptcy rate `0.0`, completion rate `1.0`, mean total reward `118.0194`, mean terminal score `0.4472`

These numbers are a reference point only. From this stage onward, `test` is not used for tuning.

## Hosted SFT Flow

1. Run a full hosted teacher harvest on `train` and `validation` using:
   - `python eval/run_baselines.py --split train --capture-conversation --openreward-env-id devatreya/AgroManager`
   - `python eval/run_baselines.py --split validation --capture-conversation --openreward-env-id devatreya/AgroManager`
2. Build SFT data from the saved hosted harvest:
   - `python scripts/prepare_sft_data.py --openreward-env-id devatreya/AgroManager`
3. The data-prep step writes:
   - `artifacts/sft/train.jsonl`
   - `artifacts/sft/validation.jsonl`
   - `artifacts/sft/selected_train_trajectories.json`
   - `artifacts/sft/selected_validation_trajectories.json`
   - `artifacts/sft/dataset_summary.json`
4. Train the SFT model on Modal:
   - `modal run pipeline/train_sft.py`
5. Probe the trained model on the hosted validation split before any RL.
