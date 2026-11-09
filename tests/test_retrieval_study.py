"""The retrieval study's metrics, quantisation and BEIR loader (on a tiny fixture; no downloads)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("retrieval_study", Path(__file__).parent.parent / "evals" / "retrieval_study.py")
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
sys.modules["retrieval_study"] = study  # dataclasses look their module up while it loads
SPEC.loader.exec_module(study)


def test_ranking_metrics():
    rel = {"a": 1, "b": 1}
    assert study.ndcg(["a", "b", "c"], rel) == pytest.approx(1.0)
    assert study.ndcg(["c", "a"], {"a": 1}) == pytest.approx(1 / np.log2(3))
    assert study.recall(["x", "a"], rel, k=2) == 0.5
    assert study.mrr(["x", "y", "b"], rel) == pytest.approx(1 / 3)
    low, high = study.bootstrap([0.0, 1.0] * 50)
    assert low < 0.5 < high


def test_binary_rescoring_recovers_the_float_ranking():
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(200, 64)).astype(np.float32)
    queries = docs[:20] + 0.05 * rng.normal(size=(20, 64)).astype(np.float32)  # each query is near its document

    def typed(x):
        return {"float": x, "int8": np.round(x / np.abs(x).max(axis=1, keepdims=True) * 127).astype(np.int8),
                "ubinary": np.packbits(x > 0, axis=1)}

    q, d = typed(queries), typed(docs)
    for precision in ("float32", "int8", "binary + rescore"):
        top = study.dense_scores(precision, q, d).argmax(axis=1)
        assert (top == np.arange(20)).mean() == 1.0, precision
    assert study.BYTES["binary"](1024) == 128 and study.BYTES["float32"](1024) == 4096


def test_beir_loader(tmp_path: Path):
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in [
        {"_id": "d1", "title": "Snow", "text": "Arterial roads are cleared first."},
        {"_id": "d2", "title": "", "text": "Libraries open at nine."}]))
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in [
        {"_id": "q1", "text": "which roads are cleared first"}, {"_id": "q2", "text": "unjudged"}]))
    (tmp_path / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\n")
    corpus = study.beir(tmp_path, None)
    assert list(corpus.queries) == ["q1"] and corpus.qrels == {"q1": {"d1": 1}}
    assert len(corpus.chunks) == 2 and corpus.parent["d2"] == "d2"


def test_the_whole_study_runs_offline(capsys, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(study, "CACHE", tmp_path / "cache")
    monkeypatch.setattr("sys.argv", ["retrieval_study.py", "--dims", "256,512", "--out", str(tmp_path / "r.md")])
    assert study.main() == 0
    report = (tmp_path / "r.md").read_text()
    assert "Hybrid → rerank" in report and "512 × binary + rescore" in report
    assert len(json.loads((tmp_path / "r.json").read_text())["results"]) == 5 + 2 * 4
