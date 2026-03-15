# RAG Assistant

This repository contains a small Retrieval-Augmented Generation (RAG) demo.

Quickstart
----------

1. Copy the example environment file and fill real secrets:

```bash
cp .env.example .env
# edit .env and fill API_KEY or LLM_VENDER_SECRET, DATABASE_URL, etc.
```

2. Create and activate a virtual environment, install deps:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run the app or scripts as required (example):

```bash
uvicorn main:app --reload
# or use your preferred entrypoint
```

Secrets and repository hygiene
----------------------------

- Do NOT commit `.env` — it is already listed in `.gitignore`.
- Commit `.env.example` (this file) which contains placeholders and allowed defaults.
- For CI / deployment, store secrets in the platform's secret manager (GitHub Actions secrets, Heroku config vars, etc.).

Configuration via Pydantic
-------------------------

This project loads settings using `pydantic` / `pydantic-settings` from a `.env` file via `settings.py`.
Sensitive values are typed as `SecretStr` in `settings.py`; to access the raw string call `settings.api_key.get_secret_value()`.

If you need help adding your keys into `.env`, tell me which variables you have and I can populate the file locally.
