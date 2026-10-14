# Lux Python Agent

Lux's local Session execution driver uses LangGraph and native SQLite checkpoints. Java owns immutable Session inputs and business projections; Python owns Run/Turn decisions, model context, execution receipts and canonical output facts. The v2 wire contract is defined in the sibling Java repository's storage design and shared JSON fixtures.

## Setup and verification

Use Python 3.14 from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m pip check
```

Direct dependencies are pinned in `pyproject.toml`. LangGraph and its native SQLite saver provide the execution/checkpoint boundary; aiosqlite supports the saver, langchain-core provides typed model messages, langchain-openai supplies ChatOpenAI, HTTPX provides Java/provider HTTP calls, and aiokafka connects durable command/event delivery. There is no complete transitive dependency lock.

```bash
python -m uvicorn lux_agent.main:app --reload
```

Without configuration this starts only the HTTP application. `GET /health` returns `{"status":"ok"}` and checks process liveness, without broker/model/backend calls. ASGI lifespan starts durable workers only when `LUX_AGENT_STORE` is set. Do not use reload or multiple Uvicorn workers for the enabled execution service.

| Environment variable | Meaning |
| --- | --- |
| `LUX_AGENT_STORE` | Private local SQLite path, for example `data/agent-v2.sqlite3`; absence disables workers |
| `LUX_BACKEND_URL` | Java internal HTTP base URL for plan/Task tools, authoritative plan context, operation reconciliation and input repair |
| `LUX_KAFKA_BOOTSTRAP_SERVERS` | Kafka bootstrap servers; absence disables delivery, while local execution can still process a prepopulated Inbox |
| `LUX_KAFKA_GROUP_ID` | Command consumer group, default `lux-agent-v2` |
| `LUX_OPENAI_ENABLED` | Set exactly `true` (case insensitive) to opt in to direct Agent model execution; default `false` |
| `OPENAI_API_KEY` | Runtime-only OpenAI credential; absence leaves the model executor disabled even when opted in |
| `LUX_OPENAI_ACCOUNT_REF` | Stable non-secret account binding, default `openai-primary`; keep it tied to the same provider account across key rotation/restarts |
| `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` | Optional OpenAI SDK organization/project scope; saved as non-secret receipt bindings and checked on recovery |
| `LUX_AGENT_MEDIA_OBSERVATION_ENABLED` | Defaults `true`; an active executor and configured Java backend enable authorized image/frame observation and require FFmpeg/ffprobe |
| `LUX_AGENT_RECONCILIATION_ENABLED` | Defaults `false`; enables restricted original-response evidence routes |
| `LUX_AGENT_RECONCILIATION_TOKEN` | Runtime-only Bearer token for the restricted routes; never commit it |
| `LUX_AGENT_RECONCILIATION_OPERATOR` | Trusted audit actor from server configuration, never from a request body |
| `LUX_AGENT_RECONCILIATION_ACCOUNTS` | Required comma-separated allowlist of exact non-secret account references when enabled |

Keep the internal HTTP API on a trusted local/private network. Reconciliation routes have separate Bearer authentication; other internal routes still rely on that network boundary. Distributed execution remains outside this single-process implementation. Credentials do not belong in configuration committed to the repository.

## Container runtime

The repository Dockerfile installs the existing package in a build stage and runs
it on `python:3.14-slim-trixie` with FFmpeg/ffprobe. The build verifies their `fd`
protocol support and Python dependency consistency. The image runs as UID/GID
`10001:10001`, binds `0.0.0.0:8000`, and starts one Uvicorn worker without reload.
The build context includes only package source, package metadata and this README;
host virtual environments, credentials, Git state, tests and local data are excluded.
The base tag floats across Python patch releases and transitive dependencies remain
unlocked, so the image is not a fully reproducible dependency snapshot.

Build and check the HTTP application from this repository:

```bash
docker build -t lux-python-agent:local .
docker run --rm --init -p 127.0.0.1:8000:8000 lux-python-agent:local
```

Without runtime configuration this exposes only the HTTP application. The unified
six-service stack and deterministic acceptance override are documented in
[`../lux-java-backend/deploy/local/README.md`](../lux-java-backend/deploy/local/README.md).
Compose supplies service DNS, private runtime configuration and dependency gates;
keep real model execution disabled until separately configured.

For durable execution, set `LUX_AGENT_STORE=/var/lib/lux-agent/agent-v2.sqlite3` and
mount the **whole** `/var/lib/lux-agent` directory. It holds the Store, checkpoint,
lock and SQLite WAL/SHM files together. The image creates the directory with mode
`0700`, owned by UID/GID 10001, so a fresh named volume inherits usable ownership.
Existing or bind-mounted directories must already grant that user access; never
reset or repair historical stores to make a container start. Runtime credentials
must be supplied outside the image. `/tmp` must remain writable for bounded media
decoding. `/health` establishes process liveness, not backend or broker readiness.

## Execution boundary

`create_app(settings, executor=..., model_reconciler=..., tools=..., plan_context_reader=...)` and `AgentRuntime(store, executor=..., model_reconciler=..., tools=..., repair=..., plan_context_reader=...)` accept real adapter implementations. `ModelExecutor` declares supported live output slots and yields named `ModelTextFragment` and one original `ModelResponseObserved`. `ToolExecutor` accepts a named `ToolExecutionRequest`. These injectable boundaries are also used by tests. With the opt-in flag and runtime key, ASGI lifespan wires one `OpenAIModelExecutor` as both executor and reconciler. A received input with no configured executor ends its Run with `MODEL_NOT_CONFIGURED`.

The Agent calls `ChatOpenAI(model="gpt-6-luna", use_responses_api=True, max_retries=0)` against the fixed official OpenAI endpoint. It sends the full Agent-owned message history and the current request's tool schemas, disables provider truncation, and never replaces Session context with a provider conversation or `previous_response_id`. Text and reasoning summaries stream into their declared live slots; complete function calls retain original call IDs and are decoded into existing typed arguments before business tools can run. The original LangChain AIMessage, including provider reasoning extensions, remains in private Agent context. Connector continues to own image/video generation only.

`ModelRequest.tools` exposes `get_model_catalog`, `read_generation_plan`, `create_generation_plan`, `update_generation_plan`, `change_generation_target`, `clone_generation_plan`, `focus_generation_plan`, `submit_generation_plan`, and `get_task`. Tool schemas contain only model arguments. The transport injects owner/Session scope and uses the original TOOL_CALL Item UUID as the operation identity for every plan tool, including reads. The old opaque `submit_generation` call is retained only for historical decoding and recovery.

`BackendClient` posts typed invocations to `/internal/v2/users/{owner}/sessions/{session}/generation-plan-tools/{tool_name}`. Uncertain outcomes are reconciled by querying `/generation-plan-tool-operations/{original_operation_id}`; a missing receipt leaves the operation UNKNOWN and blocks further business work. Complete original receipts are persisted before checkpoint completion, with operation scope, canonical argument hash and child Task membership verified. Conversation Item payloads contain the receipt result, preserving historical snapshots, ordered Task lists, default explanations, null values and explicit false. Legacy uncertain submissions still query `/tasks/by-operation/{operation_id}`. A SYSTEM/TASK_EVENT notice contains only the task ID; a new `get_task` call obtains its results.

The runtime checks the entire response before invoking Java: dependent mutations/submissions and focus-changing groups receive `DEPENDENT_TOOL_BATCH` with proven nonexecution. Independent reads and explicitly intended submissions of the observed focused FROZEN revision may run concurrently. Mutate, observe the new revisions, and submit in a later response.

When configured with `plan_context_reader` (the configured Java backend supplies this), the runtime fetches `/generation-plan-context` after assembling history and before each new model invocation. Its named `GenerationPlanContext` contains authoritative focus, a required nullable focused plan snapshot, and catalog. Hydration failure prevents model invocation; summaries and old tool receipts never substitute for current authority. New plan calls require this observation. Custom adapters receive this context and nine tool definitions in `ModelRequest`; provider response decoding remains their responsibility. Java supplies four disabled-by-default BytePlus catalog entries; this Agent registers no image/video generation providers.

One Turn is one model response, all its tool calls, and postprocessing. Inputs are received durably at any time but enter model context only at the next frozen Turn boundary. All prior real tool calls must have their original confirmed result or a proven uninvoked error before another model request. Known failures do not conceal unknown external outcomes.

STOP_SESSION completes an already started Turn, then ends that Run. It does not start an empty Turn or erase pending/context inputs. A later first-received input/task notice/CONTINUE_SESSION wakes a new Run; a later stop suppresses an earlier wake. Admission order comes from the local Inbox ordinal, while `input_sequence_no` orders context and detects gaps. Missing inputs are repaired from Java's retained original InputAccepted envelopes through the maximum admitted input sequence.

## Authorized media observation

Agent obtains Asset metadata/content only from Java's owner-scoped internal endpoints. It verifies owner, Asset ID, original Task/version source, MIME, immutable length/SHA-256 and decoded media, then freezes compact typed descriptors. Original Items and ToolResults remain unchanged. Successful descriptors survive tool turns and native checkpoint reopening; failed new media does not block later input.

At model invocation, descriptors become transient verified JPEG `input_image` blocks with source labels and video timestamps. Presentation Base64/URLs never enter durable messages/checkpoints. The window holds at most eight presentations, keeps recent successful observations, and reserves space for fresh Assets. Images support PNG/JPEG up to 10 MiB; videos support MP4/MOV up to 100 MiB and 120 seconds, at most eight uniformly sampled frames. Decoded source pixels are limited to 40 million; each JPEG is at most 1024 pixels on its longest edge and 1 MiB. FFmpeg/ffprobe subprocesses use 20-second budgets, one thread and fd/pipe-only input protocols. Audio observation is unsupported.

Provider receipt ownership is recorded before invocation-side hydration with `dispatch_started=false`, then durably switches true immediately before the SDK request. Cancelled/crashed false-phase work settles `MODEL_NOT_SUBMITTED` without network or resubmission. Existing receipts upgrade with true to preserve prior uncertainty. Once dispatch starts, missing IDs remain unresolved; known IDs recover only by original GET without media refetch.

## Original model evidence reconciliation

When explicitly enabled with a token, trusted operator and allowed account references, GET `/internal/model-reconciliation` lists unresolved provider executions; GET `/{execution_id}` reads a compact scoped view; POST `/{execution_id}/original-receipt` attaches a proven original response ID. The request contains the unchanged binding, run/turn IDs, response ID, evidence reference and reason. Authenticated server configuration supplies the audit actor. Original binding/hash/account/org/project must match; undispatched work, terminal/conflicting evidence and body actor fields are rejected. Receipt and immutable audit commit together; exact replay is idempotent.

The running dispatcher resumes original recovery on its normal interval. Attachment does not invoke a new model request or directly confirm success. Operator attestation remains necessary; local hashes do not prove provider idempotency or discover a lost ID. Without original evidence, acceptance remains unresolved.

## Durable storage

`AgentStore(Path(...))` owns nine application tables: logical command Inbox, physical Kafka receipts, Run identity registry, Session event counter, canonical events, publication queue, execution receipts, model provider identity receipts, and reconciliation audits. The last two tables and provider dispatch phase are added when opening an existing owned v2 database, without resetting its data or checkpoints. The paired `<store>.checkpoints` file belongs to `AsyncSqliteSaver`; the application does not duplicate native checkpoint schema. The store's OS lock and an in-process driver guard cover the paired lifecycle. Use one local process and one database pair per consumer workload; separate copies are not distributed fencing.

Legacy v1 or unrecognized databases are refused without deleting them. Select a fresh v2 path. Store and checkpoint files contain private prompts and responses; retain them together, with local owner-only filesystem access. Canonical events and original inputs are retained without automatic cleanup.

The native graph runs with `durability="sync"` at preparation and lifecycle barriers. Its state contains typed Session context, pending inputs, counters, execution bindings, stop effects and pending outgoing intents. Kafka receipt ingestion only writes the durable Inbox, never a concurrently running graph state.

Checkpoint and AgentStore writes use separate transactions. A checkpoint incorporating input precedes its Inbox APPLIED marker; replay repairs a missing marker using stable Item IDs. Outgoing intents remain in the checkpoint until canonical event/counter/publication commits, and their later removal is checkpointed. Streaming callbacks journal observed starts/fragments directly before publication, with transactionally allocated Session `event_seq` and Item `stream_seq`; they do not prove model/Turn completion.

The complete original model response and stable output bindings are saved in the execution receipt before checkpoint incorporation. ModelResponseCompleted publishes complete snapshots of original Item identities, before tools execute; it does not finish a Turn or Run. Supported live slots publish only when actually observed. Additional output forms are buffered until the full durable response binds them. Unobserved candidates stay private. Each live fragment corresponds to one temporary text block; complete snapshots replace the whole payload.

PREPARED execution receipts permit invocation; IN_FLIGHT/UNKNOWN receipts require reconciliation of the original execution. CONFIRMED/FAILED outcomes are reused after crashes rather than issuing another model call or Task. UNKNOWN keeps Run RUNNING with a recovery phase and Turn INTERRUPTED, and blocks successors. Partial Items remain open until the original response is recovered or a failure is confirmed. Process shutdown cancels local work and leaves uncertain external outcomes for recovery.

The OpenAI adapter saves execution/owner/Session/request-hash/account/model/endpoint/organization/project bindings before submission. The SDK-supported `X-Client-Request-Id` header carries the original execution UUID for operator/vendor correlation only; it does not provide idempotency or lookup by trace. In the pinned `langchain-openai==1.6.0`, the public `response.created` AIMessageChunk exposes the response ID through `response_metadata["id"]`; the adapter commits it before yielding dependent fragments. No private LangChain converter or stream hook is used. The complete original SDK message is then retained using the existing execution receipt boundary. This is a normalized SDK archive, not a byte-for-byte archive of raw provider SSE events.

Recovery only calls `GET /v1/responses/{saved_original_id}` with retries disabled. A retrieved completed response can seal the original partial Items; a confirmed failed/cancelled/incomplete response or invalid complete business payload is persisted as FAILED by the runtime. A definitive HTTP 400/401/403 rejection of the original create request, before any accepted response ID, becomes `MODEL_PROVIDER_REJECTED` and FAILED; provider error details are not retained. A missing ID after uncertain transport/server failure, missing/expired provider record, retrieval error, or still-running response remains UNKNOWN. Once an accepted response ID has been observed, later errors cannot be reclassified as pre-acceptance rejection. Changed request/account/project/organization bindings are rejected before retrieval. An account reference does not authenticate account identity: operators must keep the reference bound to credentials for the original account. SDK presence alone does not guarantee recovery: acceptance followed by loss before the public created chunk is delivered and locally committed has no retrievable identity here and never triggers another submission. Provider retention, model access, account permissions and live behavior remain unverified.

Kafka commands use `lux.agent.commands.v2`, events use `lux.agent.events.v2`, schema version 2, with owner/Session partition keys. Broker ACK follows durable receipt, including diagnosed invalid records. Failed receipt/offset commits cannot be skipped by a later offset. Publication retries reuse the original immutable event; explicit replay increments `dispatch_generation` so an older publish ACK cannot clear a new replay request. This is at-least-once delivery, not a Kafka/database transaction.

The coordinated baseline has a 32 MiB Kafka record/request capacity, a 64 MiB consumer fetch capacity, and a 31 MiB serialized execution-envelope budget measured in UTF-8 bytes, reserving 1 MiB for transport overhead. Model/tool adapters must budget the complete fact envelope, including multi-Item response bundles, and each live fragment envelope. The canonical append transaction rejects an oversized fact before committing its counter/event/publication, without truncation. An already observed complete response retains its original confirmed receipt and checkpoint intent and blocks subsequent publication/tool execution until transport limits are repaired together; it is not a provider failure or a reason to repeat the model/tool invocation. A live journaling failure before the complete original response is durably observed follows the existing UNKNOWN recovery rule.

## Source synchronization

`GET /internal/v2/users/{owner}/sessions/{session}/execution?after_event_seq=0&limit=100` returns named `SessionExecutionSync` with a fixed committed event high watermark, bounded original event page, latest canonical Run/Turn/observed Item snapshots, and an optional page cursor. `through_event_seq` bounds the event page. Private context and unobserved stream candidates are excluded. Initial snapshots include the full retained entity set; this local baseline does not implement entity pagination or retention.

QUERY_SESSION_EXECUTION requeues existing facts idempotently; REPLAY_SESSION_EVENTS requeues the requested existing sequence range. Neither invents lifecycle facts. Java may project facts out of order using stable identities and independent source revisions.

Tests use real local SQLite and the native graph/saver, injected model/tool adapters, HTTPX MockTransport and ASGI lifespan. They cover input/stop boundaries, original receipts, streaming, gaps, UNKNOWN recovery and broker acknowledgment ordering. They do not prove live Kafka, deployed Java integration or external provider behavior.
