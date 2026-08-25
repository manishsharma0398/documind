"""Score a recorded eval run.

Reads a result file from `run_eval.py` and computes retrieval metrics. Separate
from the runner on purpose: the expensive part is querying, and the floor should
be chosen by looking at the whole score distribution rather than by guessing a
value and re-running.
"""

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
KS = (1, 3, 5, 10)
FLOORS = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def first_hit_rank(record) -> int | None:
    """1-based rank of the first hit from an expected source, if any."""
    expected = set(record["expected_sources"])
    for rank, hit in enumerate(record["hits"], 1):
        if hit["source"] in expected:
            return rank
    return None


def top_score(record) -> float:
    """Best score returned, or 0 when nothing came back."""
    return record["hits"][0]["score"] if record["hits"] else 0.0


def best_correct_score(record) -> float:
    """Score of the highest-ranked correct hit, or 0 if it was never retrieved."""
    expected = set(record["expected_sources"])
    scores = [h["score"] for h in record["hits"] if h["source"] in expected]
    return max(scores) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result",
        type=Path,
        nargs="?",
        help="result file (default: the most recent in results/)",
    )
    args = parser.parse_args()

    path = args.result
    if path is None:
        runs = sorted(RESULTS.glob("*.json"))
        if not runs:
            raise SystemExit("no result files in results/")
        path = runs[-1]

    data = json.loads(path.read_text())
    records = data["results"]
    config = data["config"]

    errored = [r for r in records if r.get("error")]
    scored = [r for r in records if not r.get("error")]
    answerable = [r for r in scored if r["type"] != "absent"]
    absent = [r for r in scored if r["type"] == "absent"]

    print(f"{path.name}")
    print(f"  model {config['embedding_model']}  collection {config['collection']}")
    print(f"  {len(answerable)} answerable, {len(absent)} absent", end="")
    print(f", {len(errored)} errored" if errored else "")

    if not answerable:
        raise SystemExit("\nno answerable questions to score")

    ranks = [first_hit_rank(r) for r in answerable]

    print("\nretrieval")
    for k in KS:
        hits = sum(1 for rank in ranks if rank is not None and rank <= k)
        print(f"  hit@{k:<3} {hits / len(answerable):.3f}  ({hits}/{len(answerable)})")
    mrr = sum(1 / rank for rank in ranks if rank) / len(answerable)
    print(f"  MRR    {mrr:.3f}")

    missed = [r["id"] for r, rank in zip(answerable, ranks, strict=True) if not rank]
    if missed:
        print(f"  never retrieved: {', '.join(missed)}")

    # By type, since ambiguous-term is the one Chapter 8 has to move.
    types = sorted({r["type"] for r in answerable})
    if len(types) > 1:
        print("\nhit@5 by type")
        for t in types:
            group = [
                (r, rank)
                for r, rank in zip(answerable, ranks, strict=True)
                if r["type"] == t
            ]
            hits = sum(1 for _, rank in group if rank and rank <= 5)
            print(f"  {t:<15} {hits / len(group):.3f}  ({hits}/{len(group)})")

    # The floor trade-off: refusing more junk costs correct answers eventually.
    # The best floor is the highest one that has not started costing them.
    print("\nscore floor sweep")
    print("  floor   refused-absent   kept-correct")
    for floor in FLOORS:
        refused = sum(1 for r in absent if top_score(r) < floor)
        kept = sum(1 for r in answerable if best_correct_score(r) >= floor)
        refusal = f"{refused / len(absent):.3f}" if absent else "  n/a"
        print(
            f"  {floor:.2f}    {refusal:>10}       "
            f"{kept / len(answerable):.3f}  ({kept}/{len(answerable)})"
        )

    if not absent:
        print("  (no absent questions: refusal accuracy is unmeasured)")


if __name__ == "__main__":
    main()
