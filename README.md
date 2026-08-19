# DocuMind

A retrieval-augmented Q&A service for large personal knowledge bases — ask questions
against your own notes and get answers grounded in them, with citations.

Built around a corpus of ~410 markdown documents (~484k words) spanning 14 technical
domains, which is large enough that retrieval quality is a real engineering problem rather
than a formality.

> **Status:** FastAPI skeleton is up; ingestion and retrieval are in progress. Metrics in
> the evaluation table are filled in from measured runs only — empty cells mean not yet
> measured, not zero.

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
| health check | load balancers, container orchestration, application endpoints |
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
       /retrieve ─▶│  retrieve    │──▶ query rewrite ──▶ hybrid search
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

**Design decisions worth calling out:**

- **Chunking on token boundaries, not characters,** with overlap — character splits cut
  mid-token and produce embeddings for fragments that mean nothing.
- **Idempotent re-indexing.** Re-ingesting an unchanged corpus is a no-op; updates delete
  by payload filter before reinsert, so repeated ingestion never silently duplicates chunks
  and skews retrieval.
- **Batched embedding with cost tracking** — embedding calls dominate ingestion cost, and
  batch size is the main lever on both spend and wall time.
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

Copy `.env.example` to `.env` and fill in your keys. `.env` is gitignored.

---

## Development

Formatting, linting and security checks run as pre-commit hooks — black and isort at
line-length 88, flake8, bandit over `src/`, and `uv lock --check` to keep the lockfile
honest. Run them across the whole tree with:

```bash
uv run pre-commit run --all-files
```

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
