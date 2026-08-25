"""Run the golden set against /retrieve and record the raw results.

Deliberately records rather than scores. Each question costs an embedding, so
metrics are computed offline by `report.py` — trying a different score floor or
a different k should never mean another run.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx2
import yaml

EVALS = Path(__file__).parent
RESULTS = EVALS / "results"


def load(path: Path) -> Any:
    """Read a YAML file, failing loudly if it is missing."""
    if not path.exists():
        raise SystemExit(f"{path} not found")
    return yaml.safe_load(path.read_text())


def preflight(client: httpx2.Client, url: str) -> None:
    """Fail before spending embeddings if the service is not up."""
    try:
        client.get(f"{url}/docs", timeout=5.0)
    except httpx2.HTTPError as exc:
        raise SystemExit(f"cannot reach {url}: {exc}")


def ask(
    client: httpx2.Client, url: str, question: str, top_k: int, floor: float
) -> dict[str, Any]:
    """One /retrieve call. Failures are recorded, not raised.

    One bad question should not lose the other twenty-nine, and a run that dies
    halfway has still paid for every embedding it made.
    """
    try:
        response = client.post(
            f"{url}/retrieve",
            json={"question": question, "top_k": top_k, "score_threshold": floor},
        )
    except httpx2.HTTPError as exc:
        return {"error": type(exc).__name__, "body": str(exc)[:200]}

    if response.status_code != 200:
        return {"error": f"HTTP {response.status_code}", "body": response.text[:200]}
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EVALS / "config.yaml")
    parser.add_argument("--golden", type=Path, default=EVALS / "golden_set.yaml")
    parser.add_argument("--label", default="", help="suffix for the result filename")
    args = parser.parse_args()

    config = load(args.config)
    questions = load(args.golden)["questions"]
    if not questions:
        raise SystemExit("golden set is empty")

    records = []
    with httpx2.Client(timeout=60.0) as client:
        preflight(client, config["api_url"])
        for i, item in enumerate(questions, 1):
            body = ask(
                client,
                config["api_url"],
                item["question"],
                config["top_k"],
                config["score_threshold"],
            )
            hits = [
                {
                    "source": r["source"],
                    "score": r["score"],
                    "chunk_index": r["chunk_index"],
                    "section": r["section"],
                }
                for r in body.get("results", [])
            ]
            records.append(
                {
                    "id": item["id"],
                    "type": item["type"],
                    "question": item["question"],
                    "expected_sources": item.get("expected_sources") or [],
                    "hits": hits,
                    # What the service actually applied, not what we asked for.
                    "applied_top_k": body.get("top_k"),
                    "applied_score_threshold": body.get("score_threshold"),
                    "error": body.get("error"),
                }
            )
            flag = "!" if body.get("error") else " "
            print(f"{flag}{i:>3}/{len(questions)}  {len(hits):>2} hits  {item['id']}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}{'-' + args.label if args.label else ''}.json"
    out = RESULTS / name
    out.write_text(json.dumps({"config": config, "results": records}, indent=2) + "\n")

    failed = sum(1 for r in records if r["error"])
    print(f"\nwrote {out.relative_to(EVALS.parent)}", end="")
    print(f"  ({failed} failed)" if failed else "")


if __name__ == "__main__":
    main()
