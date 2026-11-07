"""Score Clearline on a fixed question set.

    python evals/run.py                      # offline, deterministic (fake provider) - what CI runs
    CLEARLINE_PROVIDER=cohere COHERE_API_KEY=... python evals/run.py --out evals/results/cohere.md

For each question, asked as a specific user:
- retrieval: is the expected document among the passages retrieved (hit@k), and how high (MRR)?
- answers: is the answer grounded and does it cite the expected document?
- refusals: for questions the user may not see, or that no document answers, does it decline?
- leakage: did any retrieved passage or citation come from a document the user may not read?
  This must be zero; the run fails otherwise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from clearline.auth import DEV_USERS
from clearline.config import Settings
from clearline.ingest import load_directory
from clearline.models import Classification, Principal
from clearline.service import Clearline

ROOT = Path(__file__).resolve().parent.parent
EVERYONE = Principal("eval", Classification.PROTECTED_B, frozenset({"staff", "finance", "security", "it"}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "evals" / "questions.jsonl")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-hit", type=float, default=0.0, help="fail below this retrieval hit rate")
    args = parser.parse_args()

    settings = Settings(store="memory")
    clearline = Clearline(settings)
    for document in load_directory(ROOT / "sample_data"):
        clearline.ingest(document, "eval")
    labels = {d.id: (d.title, d.classification, d.groups) for d, _ in clearline.store.documents(EVERYONE)}
    title_of = {i: t for i, (t, _, _) in labels.items()}

    rows, ranks, answered, refused, latencies, leaks = [], [], [], [], [], 0
    for line in args.questions.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        user = DEV_USERS[case["user"]]
        hits = clearline.retriever.search(case["question"], user, settings.context_chunks)
        answer = clearline.ask(case["question"], user)
        latencies.append(answer.latency_ms)
        seen = [h.chunk.document_id for h in hits] + [c.document_id for c in answer.citations]
        leaked = [title_of[d] for d in seen if not user.can_read(*labels[d][1:])]
        leaks += len(leaked)
        cited = {title_of[c.document_id] for c in answer.citations}
        if case["expect"]:
            titles = [title_of[h.chunk.document_id] for h in hits]
            rank = titles.index(case["expect"]) + 1 if case["expect"] in titles else None
            ranks.append(rank)
            ok = answer.grounded and case["expect"] in cited
            answered.append(ok)
            verdict = f"retrieved @{rank}" if rank else "missed"
        else:
            ok = not answer.grounded
            refused.append(ok)
            verdict = "declined" if ok else f"answered from {', '.join(sorted(cited))}"
        rows.append((case["user"], case["question"], case["expect"] or f"decline ({case.get('why', '')})",
                     verdict, "✅" if ok and not leaked else "❌"))

    hit_rate = sum(r is not None for r in ranks) / len(ranks)
    mrr = statistics.fmean(1 / r if r else 0 for r in ranks)
    p95 = sorted(latencies)[max(0, round(0.95 * len(latencies)) - 1)]
    summary = [
        ("Provider", f"{clearline.provider.name}"),
        ("Questions", f"{len(rows)} ({len(ranks)} answerable, {len(refused)} should be declined)"),
        (f"Retrieval hit@{settings.context_chunks}", f"{hit_rate:.0%}"),
        ("Retrieval MRR", f"{mrr:.2f}"),
        ("Answered with the right citation", f"{sum(answered)}/{len(answered)}"),
        ("Correctly declined", f"{sum(refused)}/{len(refused)}"),
        ("Leaked passages (must be 0)", f"{leaks}"),
        ("Latency p50 / p95", f"{statistics.median(latencies):.0f} ms / {p95} ms"),
    ]
    report = "\n".join([
        "| Metric | Result |", "|---|---|", *(f"| {k} | {v} |" for k, v in summary), "",
        "| User | Question | Expected | Outcome | |", "|---|---|---|---|---|",
        *(f"| {u} | {q} | {e} | {v} | {s} |" for u, q, e, v, s in rows),
    ])
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
    if leaks:
        print(f"\nFAIL: {leaks} passage(s) leaked across access boundaries", file=sys.stderr)
        return 1
    if hit_rate < args.min_hit:
        print(f"\nFAIL: hit rate {hit_rate:.0%} below {args.min_hit:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
