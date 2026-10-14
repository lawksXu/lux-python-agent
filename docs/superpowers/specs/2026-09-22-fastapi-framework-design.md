# FastAPI framework design

The Python service will host Lux's LangChain/LangGraph agents. This first step provides a small, runnable HTTP framework.

## Architecture

Use the locally available Python 3.14 and a single installable `lux_agent` package under `src/`. `lux_agent.main:app` is the ASGI entry point. FastAPI handles routing and OpenAPI; Uvicorn serves HTTP. Dependencies are declared in `pyproject.toml` and installed in a local virtual environment.

A root-level script would require fewer packaging files but makes imports depend on the working directory. A larger layered template would introduce structure without feature requirements. The small installed package gives this service a stable import path while keeping the initial application in one file.

## HTTP behavior

- `GET /health` returns HTTP 200 and `{"status": "ok"}` as JSON.
- The endpoint checks process liveness only and makes no external calls.
- FastAPI provides `/docs`, `/redoc`, and `/openapi.json`.
- Keep FastAPI's standard error handling.

## Scope and verification

Runtime dependencies are FastAPI and Uvicorn. pytest and HTTPX's async ASGI transport provide a health integration check; Ruff provides lint and format checks. The transport does not run lifespan, so verify startup separately with Uvicorn. Add agent libraries, provider calls, persistence, authentication, and deployment configuration when their requirements are defined.

Document setup and run commands, add Python ignore rules, and preserve project guidance in the root `AGENTS.md`. Verify dependency installation, the health integration test, lint, formatting, wheel packaging, and a real local Uvicorn startup. Do not create a Git branch or commit.
