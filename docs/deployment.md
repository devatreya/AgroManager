# Deployment

The runtime image uses `app.py` as the only server entrypoint and starts with:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

Data can live in-repo under `data/processed/` or be uploaded to OpenReward file storage and resolved from `/orwd_data/`.

Hosted deployment flow:

```bash
orwd create AgroManager \
  --description "Long-horizon UK arable farm benchmark with Qwen + Modal + ART" \
  --namespace devatreya

orwd link devatreya/AgroManager devatreya/AgroManager
orwd upload devatreya/AgroManager ./data/processed/
orwd deployments devatreya/AgroManager
orwd logs devatreya/AgroManager
```
