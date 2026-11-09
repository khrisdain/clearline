# Clearline

**A permission-aware RAG agent for regulated teams.** Staff ask questions in plain language and get answers drawn from their organisation's documents, with citations, but only ever from documents they are cleared to read. Built for the public sector, where a Protected B budget draft must never leak into an answer for someone who only holds Unclassified access.

```
$ clearline ask "What property tax increase is forecast for 2027?" --as ben     # finance, Protected B
→ "…a residential property tax increase of 2.9 percent, which adds about $118 a year to the average home. [1]"
   cites: 2027 Operating Budget Forecast › Property tax (Protected B)

$ clearline ask "What property tax increase is forecast for 2027?" --as ana     # staff, Unclassified
→ "I couldn't find this in the documents you have access to."
```

## What it does

- **Answers with span-level citations.** A tool-using agent searches (and re-searches with its own phrasing), opens whole documents when it needs to, and answers only from passages it was shown. With Cohere, passages go to Command A as documents and come back as *native* citations: which characters of the answer each passage supports. Other models cite with `[n]` markers, which are turned into the same spans. Either way, citations to passages the model never saw are dropped, the share of the answer that is cited is reported as `coverage`, and an answer with no valid citation is flagged `grounded: false`.
- **Enforces access in the database, not the prompt.** Every chunk carries its document's classification (Unclassified, Protected A, Protected B) and groups. The access check is part of the same SQL statement as the vector search, so restricted text never leaves Postgres for someone without access: it can't reach the reranker, the model, or the logs. A forbidden document is indistinguishable from one that doesn't exist.
- **Hybrid, two-stage retrieval.** Dense search (pgvector HNSW, cosine) and keyword search (BM25 in memory, Postgres full text in production), each under the same access filter, merged with reciprocal rank fusion, then narrowed by a reranker (Cohere Rerank) to the few passages worth the model's attention, with a relevance floor so weak matches lead to "not found" rather than a guess. Keyword search catches exact identifiers, form numbers and rare terms that embeddings blur.
- **Runs where the data lives.** Swap Cohere for a local Ollama server with one setting, for on-prem and air-gapped installs where nothing may leave the network. The Kubernetes manifests deny all egress except to the database, the identity provider and the model server.
- **Signs people in with the organisation's identity provider.** OIDC bearer tokens from Keycloak, verified against the realm's JWKS (issuer, audience, expiry, RS256). Clearance comes from a user attribute, groups from group membership, admin/auditor rights from realm roles.
- **Leaves an audit trail that can't be quietly edited.** Every question, ingestion and deletion is logged with who, when and which documents an answer drew on. Entries are hash-chained, so changing or deleting any past entry is detected (`GET /v1/audit/verify`).
- **Masks personal information.** Social Insurance Numbers (Luhn-checked), emails and phone numbers are redacted before a question reaches the model or the audit log.
- **Measured, not assumed.** An eval harness scores retrieval (hit@k, MRR), cited answers, correct refusals and cross-boundary leakage on a fixed question set, and CI fails the build if a single restricted passage leaks. A [retrieval study](#retrieval-study) compares first-stage methods and embedding sizes and precisions with bootstrap confidence intervals.

## Architecture

```mermaid
flowchart LR
    U[Staff member] -->|OIDC token| API[FastAPI]
    KC[Keycloak] -. JWKS .-> API
    API --> AG[Agent loop]
    AG -->|search_documents / read_document| R[Retriever]
    R -->|embed query| P[(Provider:<br/>Cohere or Ollama)]
    R -->|vector search<br/>+ access filter in SQL| PG[(Postgres + pgvector)]
    R -->|rerank| P
    AG -->|chat with tools| P
    AG --> CITE[Citation check]
    API --> AUD[(Hash-chained<br/>audit log)]
```

| Piece | Where |
|---|---|
| Chunking (heading-aware, overlap, front-matter access labels; Markdown, text, PDF) | [`ingest.py`](src/clearline/ingest.py) |
| Access model: clearance levels × groups | [`models.py`](src/clearline/models.py) |
| pgvector store with the access filter in SQL | [`store/postgres.py`](src/clearline/store/postgres.py) |
| Retrieval + agent loop + citation checking | [`agent.py`](src/clearline/agent.py) |
| Providers: Cohere (Embed, Rerank, Command), Ollama, offline fake | [`providers/`](src/clearline/providers) |
| OIDC auth | [`auth.py`](src/clearline/auth.py) |
| Audit log | [`audit.py`](src/clearline/audit.py) |
| PII redaction | [`redact.py`](src/clearline/redact.py) |
| Eval harness | [`evals/run.py`](evals/run.py) |
| Compose stack, Keycloak realm, Kubernetes | [`docker-compose.yml`](docker-compose.yml), [`deploy/`](deploy) |

## Run it

**Quickest (no keys, no Docker):** a deterministic offline provider and an in-memory store.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
clearline ask "How many days a week can I telework?" --as ana --load sample_data
```

**The full stack** (Postgres + pgvector, Keycloak, the API):

```bash
docker compose up --build
# Get a token for a dev user (ana, ben, chloe, admin; local-only password in deploy/keycloak)
TOKEN=$(curl -s http://localhost:8081/realms/clearline/protocol/openid-connect/token \
  -d grant_type=password -d client_id=clearline-cli -d username=ben -d password=dev-only-password | jq -r .access_token)
curl -s localhost:8000/v1/ask -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question": "How much is drawn from the tax stabilisation reserve?"}' | jq
```

Then open **http://localhost:8000** for the web page: sign in through Keycloak (as any dev user), ask questions, browse the documents you can read, add or delete documents as `admin`, and read and verify the audit log as `chloe`.

**With Cohere models:** `CLEARLINE_PROVIDER=cohere COHERE_API_KEY=... docker compose up --build`

**Fully on-prem with Ollama:** `CLEARLINE_PROVIDER=ollama CLEARLINE_EMBEDDING_DIM=768 docker compose --profile onprem up --build`, then pull the models once: `docker compose exec ollama ollama pull nomic-embed-text && docker compose exec ollama ollama pull llama3.1`.

**Kubernetes:** `kubectl apply -k deploy/k8s` (after creating the secret from `deploy/k8s/secret.example.yaml`). Two stateless API replicas, a pod disruption budget, restricted pod security, and a default-deny network policy.

## API

| Method | Path | Who | |
|---|---|---|---|
| `POST` | `/v1/ask` | anyone signed in | `{"question": "..."}` → answer, citations, agent steps, `grounded`, latency |
| `GET` | `/ui/config` | public | how the page signs in, and which models and retrieval mode are running |
| `GET` | `/v1/me` | anyone signed in | who you are: clearance, groups, roles |
| `GET` | `/v1/documents` | anyone signed in | the documents *you* can read |
| `POST` | `/v1/documents` | `clearline-admin` | add or replace a document with its classification and groups |
| `DELETE` | `/v1/documents/{id}` | `clearline-admin` | |
| `GET` | `/v1/audit` | `clearline-auditor` | latest audit entries |
| `GET` | `/v1/audit/verify` | `clearline-auditor` | is the hash chain intact? |
| `GET` | `/healthz`, `/readyz` | probes | |

Interactive docs at `/docs` when the API is running.

## Evaluation

```bash
python evals/run.py                                   # offline, what CI runs
CLEARLINE_PROVIDER=cohere COHERE_API_KEY=... python evals/run.py --out evals/results/cohere.md
```

26 questions asked as three different people: 19 they should get answered, and 7 they should be refused (4 restricted above their access, 3 not covered by any document). Offline results with the deterministic fake provider:

| Metric | Fake provider |
|---|---|
| Retrieval hit@5 | 100% |
| Retrieval MRR | 1.00 |
| Answered with the right citation | 15/19 |
| Correctly declined | 7/7 |
| **Leaked passages** | **0** |

The fake "model" is a word-overlap stand-in that keeps CI deterministic and keyless; it misses paraphrases such as *mileage* vs *kilometre*, which is what the four misses are. Run the same script with `CLEARLINE_PROVIDER=cohere` for real model quality.

## Retrieval study

[`evals/retrieval_study.py`](evals/retrieval_study.py) answers two design questions with numbers instead of defaults:

1. **Which first stage?** BM25, dense, hybrid (RRF), and each followed by Rerank.
2. **How big and how precise should the vectors be?** Embed v4 is trained Matryoshka-style, so 256, 512, 1024 and 1536 dimensions all work, and the API returns float, int8 and binary embeddings in one call. The study crosses size with precision (float32, int8, binary, and binary shortlisting with float rescoring) and prints the storage each costs per million passages next to its retrieval quality.

Metrics are nDCG@10, Recall@20 (what the reranker gets to see) and MRR@10, with 95% bootstrap confidence intervals over queries. Embeddings and rerank results are cached in `evals/.cache`, so re-runs and new configurations don't pay twice, and rate limits are retried with backoff.

```bash
python evals/retrieval_study.py                                     # offline: checks the harness
CLEARLINE_PROVIDER=cohere COHERE_API_KEY=... python evals/retrieval_study.py --out evals/results/retrieval-lakeview.md

# A public benchmark (BEIR SciFact: 5,183 abstracts, 300 test queries) for numbers worth comparing
curl -LO https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip && unzip -q scifact.zip -d data
CLEARLINE_PROVIDER=cohere COHERE_API_KEY=... python evals/retrieval_study.py --beir data/scifact --out evals/results/retrieval-scifact.md
```

The Lakeview sample (7 documents, 19 answerable questions) is too small to separate the methods; every one finds the right document. It is there to exercise the pipeline. Use SciFact (or any BEIR-format folder) for the comparison.

## Tests

```bash
pytest                                                             # 63 tests, offline
docker compose up -d db && CLEARLINE_TEST_DATABASE_URL=postgresql://clearline:clearline@localhost:5432/clearline pytest
```

Covers chunking, the access matrix for every dev user against every question, BM25 and rank fusion, span citations (markers and native Cohere ones), the retrieval study's metrics and quantisation, the agent loop (multi-search, step limit, dropped hallucinated citations, PII masking), token verification (expired, wrong audience, wrong issuer, forged signature), audit tamper detection, the API's role checks, the Cohere and Ollama adapters (against stand-in clients), and the same access guarantees against real Postgres + pgvector. CI runs lint (ruff), types (mypy), the tests against a pgvector service, the eval gate and a Docker build.

## Design notes

- **Filter before ranking, not after.** Post-filtering search results (or asking the model to ignore restricted text) leaks: the restricted passage has already reached the reranker or the prompt. Filtering inside the vector query means it never leaves the store. With selective filters HNSW can return fewer than `k` rows, so the search enables pgvector's iterative index scans.
- **A forbidden document looks exactly like a missing one,** in search, in `read_document` and in the document list, so the system doesn't confirm that something sensitive exists.
- **Citations are checked, not trusted.** The model can only cite numbers it was handed; anything else is removed and the answer is marked ungrounded so a client can show a warning.
- **Dev sign-in can't reach production.** `CLEARLINE_AUTH_MODE=dev` is refused at startup when `CLEARLINE_ENV=production`.

The sample documents describe the City of Lakeview, a fictional municipality. Nothing in them is real.
