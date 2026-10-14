# FastAPI Framework Implementation Plan

**Goal:** Make the empty Python service runnable with FastAPI and a verified health endpoint.

**Architecture:** A single installed `lux_agent` package exposes `lux_agent.main:app`. Keep the HTTP framework in one module until agent features need further structure.

**Tech stack:** Python 3.14, FastAPI 0.141.1, Uvicorn 0.53.0, pytest 9.1.1, HTTPX 0.28.1, Ruff 0.16.8, setuptools.

## Constraints

- Make changes only in `/Users/xuwenjie/lux-python-agent`.
- Keep dependencies in `pyproject.toml` and install into `.venv`.
- Add no agent integrations, external services, branches, or commits.
- Keep the repository-root `AGENTS.md` available to Git and ignore nested copies.

## Task: Runnable service scaffold

Create `.gitignore`, `.python-version`, `pyproject.toml`, `README.md`, `src/lux_agent/__init__.py`, `src/lux_agent/main.py`, and `tests/test_health.py`.

- [x] Declare the package and dependencies, document commands, and create an initially empty FastAPI application:

  ```python
  from fastapi import FastAPI

  app = FastAPI(title="Lux Python Agent", version="0.1.0")
  ```

- [x] Install the project with its `dev` extra in `.venv`.
- [x] Add and run the health integration test; expect HTTP 404 before adding the route:

  ```python
  import asyncio

  from httpx import ASGITransport, AsyncClient, Response

  from lux_agent.main import app


  def test_health() -> None:
      async def get_health() -> Response:
          async with AsyncClient(
              transport=ASGITransport(app=app), base_url="http://testserver"
          ) as client:
              return await client.get("/health")

      response = asyncio.run(get_health())

      assert response.status_code == 200
      assert response.json() == {"status": "ok"}
  ```

- [x] Add the health route:

  ```python
  @app.get("/health", tags=["health"])
  async def health() -> dict[str, str]:
      return {"status": "ok"}
  ```

- [x] Run `.venv/bin/python -m pytest`, `.venv/bin/python -m ruff check .`, `.venv/bin/python -m ruff format --check .`, and `.venv/bin/python -m pip check`; expect successful exits.
- [x] Build a wheel with `.venv/bin/python -m pip wheel --no-deps --no-cache-dir --wheel-dir dist .` and confirm that it contains the application package.
- [x] Start Uvicorn on an available localhost port, request `/health`, `/docs`, `/redoc`, and `/openapi.json`, and stop the verification server.
- [x] Review the files and report the verified start command and any remaining limitations.

## Verification results

Validated with Python 3.14.3. The health test first failed with HTTP 404 before the route existed, then passed with warnings treated as errors. Ruff lint and formatting checks and pip's dependency consistency check passed. The built wheel contains `lux_agent/__init__.py` and `lux_agent/main.py`. A real Uvicorn process completed startup, returned HTTP 200 for all four documented endpoints, and completed shutdown.

Only direct dependencies are pinned; transitive dependencies are resolved during installation. The folder did not contain a Git repository, and none was initialized. Generated build output is excluded explicitly from Ruff so checks also work before Git initialization.
