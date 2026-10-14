# Agent command runtime implementation plan

**Goal:** Implement the Python side of Lux's existing v1 Agent command protocol, with durable receipts, Run ownership, canonical event replay and cancellation.

**Architecture:** Keep Java's business state and wire contract unchanged. Decode commands into Pydantic types, persist receipt and execution events in a private local SQLite database, and isolate Kafka transport from the asynchronous execution boundary. This iteration owns a single local Agent process; a process lock prevents concurrent writers from executing the same database's Runs.

**Scope:** The previous roadmap authorizes this first execution milestone. Real model/provider calls, LangGraph checkpoints, tools and task-result application are subsequent milestones. Test executors live in tests. Without a configured executor, a Run fails explicitly rather than returning a fabricated answer. Task updates are retained pending context application, and a nonzero required task-update cutoff prevents execution.

**Constraints:** Python 3.14; `src/lux_agent`; named business structures and explicit annotations; English documentation; preserve snake_case and canonical UUIDs; no branches, commits or instruction-file edits. No Java database or Docker changes. Kafka dependency installation needs the user's pending approval.

## Task 1: Wire protocol

Files: `src/lux_agent/protocol.py`, `tests/test_protocol.py`, `pyproject.toml`.

- [x] Write regressions for actual Java command envelopes, discriminated content blocks, canonical IDs, ownership/key consistency and invalid numeric values.
- [x] Run `.venv/bin/python -m pytest tests/test_protocol.py`; confirm missing implementation fails.
- [x] Define `RunStart`, `RunCancel`, `RunStatusQuery`, `TaskUpdated`, their payloads, asset metadata and outgoing lifecycle/text events using Pydantic.
- [x] Run protocol tests. Declare the already-installed Pydantic version directly because production imports it.

```python
command = decode_command(raw_command_bytes)
assert command.partition_key == f"{command.user_id}:{command.session_id}"
```

## Task 2: Durable receipts and event log

Files: `src/lux_agent/store.py`, `tests/test_store.py`, `.gitignore`.

- [x] Write tests for receipt persistence across reopen, changed-content conflicts, duplicate Run identity, owner/session isolation and canonical replay after publication.
- [x] Run `.venv/bin/python -m pytest tests/test_store.py`; confirm missing behavior fails.
- [x] Implement SQLite transactions for inbox, Runs, canonical events and publication attempts. Hold a local process lock for the database lifetime. Never acknowledge a record that could not be retained.
- [x] Verify contiguous sequence allocation, stable event identity on retry, per-session publication order and interrupted execution finalization without re-execution.

```python
receipt = store.receive(record)
assert receipt in (ReceiptState.ACCEPTED, ReceiptState.DUPLICATE, ReceiptState.REJECTED)
store.close()
```

## Task 3: Asynchronous execution and cancellation

Files: `src/lux_agent/runtime.py`, `tests/test_runtime.py`.

- [x] Test text streaming using a test-only executor, duplicate commands, cancellation, shutdown, failure, status replay and task-update waiting.
- [x] Run `.venv/bin/python -m pytest tests/test_runtime.py`; confirm missing behavior fails.
- [x] Implement command application, one active execution per session, cancellation confirmation after the executor stops, and durable events before publication.
- [x] Retain task updates separately from application; do not advance a context watermark that this iteration cannot establish.
- [x] Run runtime and persistence tests together.

## Task 4: Kafka and application lifecycle

Files: `src/lux_agent/kafka.py`, `src/lux_agent/config.py`, `src/lux_agent/main.py`, `tests/test_kafka.py`, `tests/test_lifecycle.py`, `.env.example`, `README.md`.

- [ ] If approved, install only the pinned Kafka wheel into the existing `.venv`; stop on installation failure.
- [ ] Test persistence-before-commit, failed receipt/commit retry without advancing the partition, delivery retry with unchanged envelopes and startup/shutdown cleanup using transport doubles.
- [ ] Implement a dedicated Kafka I/O thread and a separate async execution dispatcher. Workers are opt-in; default health startup opens no database or broker connection.
- [ ] Document environment variables, private local persistence, one-process limits, and incomplete real-model/task-context functionality.
- [ ] Run `.venv/bin/python -m pytest`, `.venv/bin/python -m ruff check .`, `.venv/bin/python -m ruff format --check .` and `.venv/bin/python -m pip check`.
- [ ] Report simulated-boundary verification separately from real Kafka/Java integration, which is not run in this iteration.

## Current implementation boundary

The transport-independent delivery rules are implemented in `src/lux_agent/delivery.py` and verified in `tests/test_delivery.py`. The Kafka wheel installation remains pending authorization. The Kafka adapter, environment configuration and ASGI worker lifespan have not been implemented; do not claim broker interoperability. Existing health startup remains network-free. No graph or paid-model calls have been made.
