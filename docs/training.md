# Modal and ART

The Modal runtime uses Python `3.12`, Debian slim, installs `git` and `procps`, mounts:

- `/vol/art`
- `/vol/hf-cache`
- `/vol/results`

The base model is `Qwen/Qwen2.5-7B-Instruct`.

`pipeline/train_sft.py` runs SFT on Modal via ART local backend and writes `training/sft_result.json`.

`pipeline/train_rl.py` runs RL against the hosted OpenReward environment. Shared ART mode is the default. Dedicated mode is exposed as an opt-in.

`pipeline/eval_compare.py` compares the trained model to `weather_aware_rotation` on hosted tasks and writes `eval/validation_comparison.json` or `eval/test_comparison.json`.
