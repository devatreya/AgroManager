# Common Failure Modes

- Missing processed files: run `scripts/fetch_weather.py`, `scripts/fetch_prices.py`, and `scripts/build_tasks.py --scale medium`.
- SSL certificate failures: set `SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"`.
- Hosted environment not found or empty: confirm `OPENREWARD_API_KEY`, `orwd link`, and the latest deployment status.
- Task/tool schema drift: the hosted session layer normalizes schemas and prefers metadata over text. If hosted metadata is absent, fix the environment tool outputs first.
- RL rollout tool-budget exhaustion: the default budget is `240`; lower budgets are expected to fail in tests.
