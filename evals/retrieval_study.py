"""Retrieval study: which first stage, and which embedding size and precision, for Clearline?

    python evals/retrieval_study.py                                   # offline stand-in models, Lakeview corpus
    CLEARLINE_PROVIDER=cohere COHERE_API_KEY=... python evals/retrieval_study.py --out evals/results/retrieval-cohere.md
    python evals/retrieval_study.py --beir data/scifact               # any BEIR-format dataset (download separately)

Part 1 compares first-stage retrieval: BM25, dense, hybrid (reciprocal rank fusion of both), and
each followed by the reranker. Part 2 holds retrieval to dense only and varies the embedding:
Embed v4's Matryoshka output sizes (256-1536) x precision (float32, int8, binary, and binary with
float rescoring), against the bytes each vector costs to store.

Metrics are nDCG@10, Recall@20 (what the reranker gets to see) and MRR@10, with 95% bootstrap
confidence intervals over queries. Embeddings are cached under evals/.cache, so re-runs and new
configurations don't pay for the same API calls twice.

Lakeview (the sample corpus) is tiny: 7 documents, 19 answerable questions. It checks the harness
end to end; differences there are mostly noise and the intervals say so. Use a BEIR dataset such
as SciFact (5k abstracts, 300 test queries) for numbers worth comparing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import numpy as np

from clearline.agent import passage, reciprocal_rank_fusion
from clearline.config import Settings
from clearline.ingest import chunk_document, load_directory
from clearline.models import Chunk, Classification, Document, Hit, Principal
from clearline.providers import build_provider
from clearline.providers.base import EmbedKind, Provider
from clearline.store.memory import MemoryStore

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "evals" / ".cache"
EVERYONE = Principal("study", Classification.PROTECTED_B, frozenset({"staff", "finance", "security", "it"}))
Ranking = list[str]  # corpus ids, best first
T = TypeVar("T")


def with_retry(call: Callable[[], T], attempts: int = 6) -> T:
    """Rate limits (a Cohere trial key allows only a few rerank calls a minute): back off and retry."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception as error:  # the SDK raises its own TooManyRequestsError; match on the message
            if attempt == attempts - 1 or "429" not in str(error) and "rate" not in str(error).lower():
                raise
            wait = min(60, 2 ** attempt * 5)
            print(f"  rate limited, retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def cached_rerank(provider: Provider, query: str, documents: list[str]) -> list[tuple[int, float]]:
    key = hashlib.sha256(json.dumps([provider.name, provider.rerank_model, query, documents]).encode()).hexdigest()[:24]
    path = CACHE / f"{provider.name}-rerank-{key}.json"
    if path.exists():
        return [(int(i), float(s)) for i, s in json.loads(path.read_text())]
    order = with_retry(lambda: provider.rerank(query, documents, len(documents)))
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(order))
    return order


# ---- Data -----------------------------------------------------------------------------------------

@dataclass
class Corpus:
    name: str
    chunks: list[Chunk]  # what gets retrieved
    parent: dict[str, str]  # chunk id -> the id relevance is judged on (a document for Lakeview)
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]  # query id -> {judged id: relevance}
    texts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.texts = [passage(c) for c in self.chunks]


def lakeview() -> Corpus:
    documents = load_directory(ROOT / "sample_data")
    by_title = {d.title: d.id for d in documents}
    chunks = [c for d in documents for c in chunk_document(d)]
    queries, qrels = {}, {}
    for i, line in enumerate((ROOT / "evals" / "questions.jsonl").read_text().splitlines()):
        case = json.loads(line) if line.strip() else {}
        if case.get("expect"):
            queries[f"q{i}"] = case["question"]
            qrels[f"q{i}"] = {by_title[case["expect"]]: 1}
    return Corpus("Lakeview sample documents", chunks, {c.id: c.document_id for c in chunks}, queries, qrels)


def beir(path: Path, max_queries: int | None) -> Corpus:
    """A BEIR-format folder: corpus.jsonl, queries.jsonl and qrels/test.tsv."""
    qrels: dict[str, dict[str, int]] = {}
    for line in (path / "qrels" / "test.tsv").read_text().splitlines()[1:]:
        qid, doc, score = line.split("\t")
        if int(score) > 0:
            qrels.setdefault(qid, {})[doc] = int(score)
    queries = {}
    for line in (path / "queries.jsonl").read_text().splitlines():
        q = json.loads(line)
        if q["_id"] in qrels:
            queries[q["_id"]] = q["text"]
    if max_queries:
        queries = dict(sorted(queries.items())[:max_queries])
        qrels = {q: qrels[q] for q in queries}
    chunks = []
    for line in (path / "corpus.jsonl").read_text().splitlines():
        d = json.loads(line)
        chunks.append(Chunk(id=d["_id"], document_id=d["_id"], title=d.get("title") or "", section="", text=d["text"],
                            classification=Classification.UNCLASSIFIED, groups=frozenset(), position=0))
    return Corpus(f"BEIR {path.name}", chunks, {c.id: c.id for c in chunks}, queries, qrels)


# ---- Embeddings (cached), quantised -----------------------------------------------------------------

def embed_typed(provider: Provider, texts: list[str], kind: EmbedKind, dim: int) -> dict[str, np.ndarray]:
    """float32, int8 and packed-binary embeddings. Cohere returns all three natively in one call;
    for other providers the int8 and binary versions are quantised here from the floats."""
    key = hashlib.sha256(json.dumps([provider.name, provider.embed_model, kind, dim, texts]).encode()).hexdigest()[:24]
    path = CACHE / f"{provider.name}-{kind}-{dim}-{key}.npz"
    if path.exists():
        with np.load(path) as cached:
            return {k: cached[k] for k in cached.files}
    native = getattr(provider, "embed_typed", None)
    if native is not None:
        out = with_retry(lambda: native(texts, kind, dim))
    else:
        floats = np.asarray(embed_at(provider, texts, kind, dim), dtype=np.float32)
        scale = np.abs(floats).max(axis=1, keepdims=True)
        scale[scale == 0] = 1
        out = {"float": floats, "int8": np.round(floats / scale * 127).astype(np.int8),
               "ubinary": np.packbits(floats > 0, axis=1)}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def embed_at(provider: Provider, texts: list[str], kind: EmbedKind, dim: int) -> list[list[float]]:
    if hasattr(provider, "dim"):  # the offline fake: its vector size is a constructor argument
        provider.dim = dim
    elif hasattr(provider, "settings"):
        provider.settings.embedding_dim = dim
    out: list[list[float]] = []
    for start in range(0, len(texts), 96):
        out.extend(with_retry(lambda: provider.embed(texts[start:start + 96], kind)))  # noqa: B023 - called at once
    return out


def normalise(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def dense_scores(precision: str, q: dict[str, np.ndarray], d: dict[str, np.ndarray], rescore_k: int = 100) -> np.ndarray:
    """(queries x corpus) similarity under one storage precision."""
    if precision == "float32":
        return normalise(q["float"]) @ normalise(d["float"]).T
    if precision == "int8":
        return q["int8"].astype(np.float32) @ d["int8"].astype(np.float32).T
    hamming = np.stack([POPCOUNT[np.bitwise_xor(row, d["ubinary"])].sum(axis=1) for row in q["ubinary"]])
    scores = -hamming.astype(np.float32)
    if precision == "binary":
        return scores
    # binary + rescore: shortlist by Hamming distance, then re-score the shortlist with the float query
    # against the documents' ±1 bits (no float document vectors needed).
    bits = np.unpackbits(d["ubinary"], axis=1)[:, : q["float"].shape[1]].astype(np.float32) * 2 - 1
    rescored = np.full_like(scores, -np.inf)
    for i, row in enumerate(scores):
        shortlist = np.argsort(-row)[:rescore_k]
        rescored[i, shortlist] = bits[shortlist] @ q["float"][i]
    return rescored


BYTES = {"float32": lambda dim: dim * 4, "int8": lambda dim: dim, "binary": lambda dim: dim // 8,
         "binary + rescore": lambda dim: dim // 8}


# ---- Metrics ---------------------------------------------------------------------------------------

def judged(corpus: Corpus, ranking: Ranking) -> Ranking:
    """Chunk ranking -> ranking of judged ids (first occurrence wins)."""
    return list(dict.fromkeys(corpus.parent[c] for c in ranking))


def ndcg(ranking: Ranking, rel: dict[str, int], k: int = 10) -> float:
    gain = sum(rel.get(d, 0) / np.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sum(r / np.log2(i + 2) for i, r in enumerate(sorted(rel.values(), reverse=True)[:k]))
    return float(gain / ideal) if ideal else 0.0


def recall(ranking: Ranking, rel: dict[str, int], k: int = 20) -> float:
    return len(set(ranking[:k]) & set(rel)) / len(rel)


def mrr(ranking: Ranking, rel: dict[str, int], k: int = 10) -> float:
    return next((1 / (i + 1) for i, d in enumerate(ranking[:k]) if d in rel), 0.0)


def bootstrap(values: list[float], rounds: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    data = np.asarray(values)
    means = data[rng.integers(0, len(data), (rounds, len(data)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@dataclass
class Result:
    name: str
    per_query: dict[str, list[float]]
    ms_per_query: float = 0.0
    extra: dict[str, str] = field(default_factory=dict)

    def mean(self, metric: str) -> float:
        return float(np.mean(self.per_query[metric]))


def score(corpus: Corpus, name: str, rankings: dict[str, Ranking], seconds: float = 0.0, **extra: str) -> Result:
    per: dict[str, list[float]] = {"ndcg@10": [], "recall@20": [], "mrr@10": []}
    for qid, ranking in rankings.items():
        docs, rel = judged(corpus, ranking), corpus.qrels[qid]
        per["ndcg@10"].append(ndcg(docs, rel))
        per["recall@20"].append(recall(docs, rel))
        per["mrr@10"].append(mrr(docs, rel))
    return Result(name, per, 1000 * seconds / max(len(rankings), 1), extra)


# ---- Part 1: first stage and reranking ---------------------------------------------------------------

def part1(corpus: Corpus, provider: Provider, dim: int, candidates: int) -> list[Result]:
    qids = list(corpus.queries)
    ids = [c.id for c in corpus.chunks]
    started = time.perf_counter()
    q, d = embed_typed(provider, [corpus.queries[i] for i in qids], "query", dim), embed_typed(provider, corpus.texts, "document", dim)
    sims = dense_scores("float32", q, d)
    dense = {qid: [ids[j] for j in np.argsort(-sims[i])[:candidates]] for i, qid in enumerate(qids)}
    dense_seconds = time.perf_counter() - started

    store = MemoryStore()  # Clearline's own BM25, the code that runs in production for the in-memory store
    for chunk in corpus.chunks:
        store.upsert(Document(id=chunk.id, title=chunk.title, text=chunk.text), [chunk])
    started = time.perf_counter()
    keyword = {qid: [h.chunk.id for h in store.keyword_search(corpus.queries[qid], EVERYONE, candidates)] for qid in qids}
    keyword_seconds = time.perf_counter() - started

    by_id = {c.id: c for c in corpus.chunks}
    text_of = dict(zip(ids, corpus.texts, strict=True))

    def fuse(qid: str) -> Ranking:
        lists = [[Hit(by_id[i], 0.0) for i in ranking] for ranking in (dense[qid], keyword[qid]) if ranking]
        return [h.chunk.id for h in reciprocal_rank_fusion(lists)][:candidates]

    hybrid = {qid: fuse(qid) for qid in qids}

    def reranked(first: dict[str, Ranking]) -> tuple[dict[str, Ranking], float]:
        out, started = {}, time.perf_counter()
        for qid, ranking in first.items():
            order = cached_rerank(provider, corpus.queries[qid], [text_of[i] for i in ranking])
            ranked = {i for i, _ in order}
            out[qid] = [ranking[i] for i, _ in order] + [c for j, c in enumerate(ranking) if j not in ranked]
        return out, time.perf_counter() - started

    dense_rr, dense_rr_s = reranked(dense)
    hybrid_rr, hybrid_rr_s = reranked(hybrid)
    return [
        score(corpus, "BM25", keyword, keyword_seconds),
        score(corpus, "Dense", dense, dense_seconds),
        score(corpus, "Hybrid (RRF)", hybrid, dense_seconds + keyword_seconds),
        score(corpus, "Dense → rerank", dense_rr, dense_seconds + dense_rr_s),
        score(corpus, "Hybrid → rerank", hybrid_rr, dense_seconds + keyword_seconds + hybrid_rr_s),
    ]


# ---- Part 2: embedding size x precision --------------------------------------------------------------

def part2(corpus: Corpus, provider: Provider, dims: list[int], candidates: int) -> list[Result]:
    qids = list(corpus.queries)
    ids = [c.id for c in corpus.chunks]
    results = []
    for dim in dims:
        q = embed_typed(provider, [corpus.queries[i] for i in qids], "query", dim)
        d = embed_typed(provider, corpus.texts, "document", dim)
        for precision in BYTES:
            started = time.perf_counter()
            sims = dense_scores(precision, q, d)
            rankings = {qid: [ids[j] for j in np.argsort(-sims[i])[:candidates]] for i, qid in enumerate(qids)}
            size = BYTES[precision](dim)
            results.append(score(corpus, f"{dim} × {precision}", rankings, time.perf_counter() - started,
                                 bytes=str(size), per_million=f"{size * 1e6 / 2**30:.2f} GiB"))
    return results


# ---- Report ------------------------------------------------------------------------------------------

def table(results: list[Result], extra_columns: list[tuple[str, str]], baseline: Result | None = None) -> list[str]:
    head = ["Configuration", "nDCG@10 (95% CI)", "Recall@20", "MRR@10", *[c for c, _ in extra_columns]]
    if baseline:
        head.insert(2, "vs best float")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        low, high = bootstrap(r.per_query["ndcg@10"])
        cells = [r.name, f"{r.mean('ndcg@10'):.3f} ({low:.3f}–{high:.3f})", f"{r.mean('recall@20'):.3f}", f"{r.mean('mrr@10'):.3f}",
                 *[r.extra.get(key, f"{r.ms_per_query:.1f}") for _, key in extra_columns]]
        if baseline:
            cells.insert(2, f"{r.mean('ndcg@10') / baseline.mean('ndcg@10'):.1%}" if baseline.mean("ndcg@10") else "–")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--beir", type=Path, help="a BEIR-format dataset folder instead of the Lakeview sample")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--dims", default="256,512,1024,1536")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    settings = Settings(store="memory")
    provider = build_provider(settings)
    corpus = beir(args.beir, args.max_queries) if args.beir else lakeview()
    dims = [int(x) for x in args.dims.split(",")]
    print(f"{corpus.name}: {len(corpus.chunks)} passages, {len(corpus.queries)} queries; provider {provider.name}", file=sys.stderr)

    first = part1(corpus, provider, settings.embedding_dim, args.candidates)
    second = part2(corpus, provider, dims, args.candidates)
    best_float = max((r for r in second if r.name.endswith("float32")), key=lambda r: r.mean("ndcg@10"))

    report = [
        f"# Retrieval study: {corpus.name}",
        "",
        f"Provider **{provider.name}** (embed: {provider.embed_model}; rerank: {provider.rerank_model}). "
        f"{len(corpus.chunks)} passages, {len(corpus.queries)} queries, {args.candidates} candidates per stage. "
        "95% bootstrap intervals over queries.",
        "",
        f"## 1. First stage and reranking (embeddings at {settings.embedding_dim} dims, float32)",
        "",
        *table(first, [("ms / query", "")]),
        "",
        "## 2. Embedding size × precision (dense only, no rerank)",
        "",
        *table(second, [("Bytes / vector", "bytes"), ("Per 1M passages", "per_million")], baseline=best_float),
        "",
    ]
    if provider.name == "fake":
        report += ["> Offline stand-in models (hashed bag-of-words embeddings, term-overlap reranker): this run checks the "
                   "harness, not model quality. Rerun with `CLEARLINE_PROVIDER=cohere` for real numbers.", ""]
    text = "\n".join(report)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        args.out.with_suffix(".json").write_text(json.dumps({
            "corpus": corpus.name, "provider": provider.name, "embed": provider.embed_model, "rerank": provider.rerank_model,
            "results": [{"part": part, "name": r.name, **{m: r.mean(m) for m in r.per_query}, "ms_per_query": r.ms_per_query,
                         **r.extra} for part, rs in ((1, first), (2, second)) for r in rs],
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
