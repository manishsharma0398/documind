# DocuMind

A retrieval-augmented Q&A service for large personal knowledge bases — ask questions
against your own notes and get answers grounded in them, with citations.

Built around a corpus of ~370 markdown documents spanning 9 technical domains, which is
large enough that retrieval quality is a real engineering problem rather than a formality.

> **Status:** ingestion and search work end to end — 373 documents into 5,154 chunks in
> about a minute, and re-running costs nothing. `/retrieve` returns ranked, cited chunks;
> grounded answer generation is the next piece. Numbers in the evaluation table come from
> measured runs only; empty cells mean not yet measured.

---

## The problem this solves

Semantic search over a personal corpus fails in a specific way: the same term means
different things in different contexts. A query about **caching** could reasonably be about
HTTP caches, database query caches, build caches, CDN behaviour, or embedding caches —
and a dense-only retriever will happily return a confident mix of all of them.

Terms with that property in this corpus:

| Term | Distinct contexts it appears in |
|---|---|
| caching | HTTP, database, build, CDN, application, embedding |
| retry | HTTP clients, job queues, database transactions, infrastructure provisioning |
| health check | load balancers, reverse proxies, application endpoints |
| connection pool | databases, HTTP clients, reverse proxies |
| rate limit | API gateways, application middleware, third-party quotas |

These are the queries the system is tuned and measured against, because they are where
naive retrieval quietly degrades.

---

## Architecture

```
                  ┌──────────────┐
   POST /ingest ──▶│   ingest     │──▶ chunk (token boundaries + overlap)
                  └──────────────┘         │
                                           ▼
                                   batched embedding ──▶ ┌─────────┐
                                                          │ Qdrant  │
                                   dense + sparse vectors │ :6333   │
                                                          └─────────┘
                                                               ▲
                  ┌──────────────┐                             │
  POST /retrieve ─▶│  retrieve    │──▶ query rewrite ──▶ hybrid search
                  └──────────────┘         │              (RRF fusion)
                                           ▼                   │
                                    cross-encoder rerank ◀──────┘
                                           │
                                           ▼
                                  context assembly + citations
                                           │
                                           ▼
                                    grounded answer
```

Dense retrieval is built end to end. Sparse vectors, query rewriting, reranking and
answer generation are the remaining work.

**Design decisions worth calling out:**

- **Chunking on token boundaries, not characters,** with overlap — character splits cut
  mid-token and produce embeddings for fragments that mean nothing.
- **Idempotent re-indexing.** Each chunk carries a hash of its source and the number of
  chunks that file produced, so a re-ingest rebuilds only what changed — and a run that
  died halfway is detected rather than mistaken for a finished one. On the test corpus a
  second run touches nothing and costs nothing.
- **Batched embedding with cost tracking**, streamed end to end. Chunks are generated,
  embedded a batch at a time, upserted and dropped, so peak memory tracks the batch rather
  than the corpus. Every run reports tokens billed against tokens counted locally; a
  mismatch means something drifted.
- **A similarity floor with context-only prompting.** If nothing clears the threshold, the
  system refuses rather than answering from parametric knowledge. Refusal accuracy is
  measured, not assumed.

---

## Evaluation

Retrieval improvements are easy to claim and hard to substantiate, so each technique is
added **one at a time with the full eval re-run after each** — the per-technique delta is
attributable, including the cost and latency it costs you.

The golden set is deliberately mixed:

| Type | What it tests | Share |
|---|---|---|
| `factual` | one document clearly answers it | ~1/3 |
| `ambiguous-term` | a term spanning several unrelated contexts | ~1/3 |
| `multi-hop` | requires combining two documents from different areas | ~1/5 |
| `absent` | deliberately not in the corpus — measures refusal accuracy | ~5 questions |

**Faithfulness is scored separately from correctness.** A system can be perfectly faithful
to bad retrieval — conflating the two hides which half is actually broken.

| Stage | hit@5 | MRR | Faithfulness | Refusal acc. | p95 | $/1k queries |
|---|---|---|---|---|---|---|
| Baseline (dense only) | — | — | — | — | — | — |
| + query rewriting | — | — | — | — | — | — |
| + hybrid (BM25 + dense, RRF) | — | — | — | — | — | — |
| + cross-encoder rerank | — | — | — | — | — | — |

A note on corpus size: retrieval metrics saturate on small corpora. If `top_k=5` retrieves
a double-digit percentage of the whole index, hit@5 approaches 100% before any tuning, and
every subsequent improvement measures as zero. The corpus here is sized so the metrics have
somewhere to move.

The baseline is frozen once generated — regenerating it against a changed corpus makes
every later delta meaningless.

---

## Running it

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev --group lint
uv run pre-commit install
```

Qdrant, via Docker:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

The API:

```bash
uv run fastapi dev src/main.py
```

Interactive docs at `http://127.0.0.1:8000/docs`.

Copy `.env.example` to `.env` and fill in your keys. `INGEST_ROOT` is the one that matters
most — every ingest request must resolve inside it. Settings are read once at startup, so
changing `.env` needs a restart.

Then index something:

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'content-type: application/json' \
  -d '{"folder_path": "/path/to/notes", "extensions": [".md"], "exclude": ["drafts/**"]}'
```

Run it again and it will report everything skipped.

---

## Development

Formatting, linting, typing and security checks run as pre-commit hooks — black and isort at
line-length 88, flake8, pyright, bandit over `src/`, and `uv lock --check` to keep the
lockfile honest. Run them across the whole tree with:

```bash
uv run pre-commit run --all-files
```

Requests are logged in and out as JSON with a correlation id on every line, including lines
from deep in the call stack, so one request's whole trace can be pulled out at once. The id
comes back as `x-request-id`.

---

## Versioning and releases

Versions are derived, never hand-edited.

**Branch builds** get a SemVer pre-release identifier and a PEP 440 equivalent, so the same
commit has a valid identifier for both container tags and Python packaging:

| Branch | Commit | SemVer | PEP 440 |
|---|---|---|---|
| `feat/hybrid-search` | 1st | `0.1.0-hybrid-search.1` | `0.1.0.dev1+hybrid.search` |
| `feat/hybrid-search` | 3rd | `0.1.0-hybrid-search.3` | `0.1.0.dev3+hybrid.search` |
| `master` | — | `0.1.0` | `0.1.0` |

The counter is `git rev-list --count origin/master..HEAD` — the commits the branch adds.
Deriving it from git rather than a CI counter keeps it reproducible from any clone, with no
build-server state involved. CI validates both strings against the official SemVer regex and
PEP 440 before anything downstream consumes them.

**Releases** are automated with [release-please](https://github.com/googleapis/release-please).
Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/), which
determines the bump — `fix:` patch, `feat:` minor, `!` major. A rolling release PR
accumulates the changelog; merging it tags the version and bumps `pyproject.toml`. The
convention is enforced by a `commit-msg` hook, since a malformed message is only fixable by
rewriting history once pushed.

## Stack

Python 3.13 · FastAPI · Qdrant · LangChain · OpenAI · Pydantic · uv
