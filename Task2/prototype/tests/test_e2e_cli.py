"""End-to-end: drive the CLI exactly as the README does, one process per step.

ingest and ask run in separate processes, as they do for a user, so state
that does not survive a process boundary (salted hashes, unsaved weights)
fails here. Offline: hashing embedder, no reranker, no generator.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys

import pytest

from conftest import PROTO, workspace_env

CLI = [sys.executable, str(PROTO / "rag_prototype.py")]


def run(env, *args, check=True):
    proc = subprocess.run([*CLI, *args], env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=300)
    if check and proc.returncode != 0:
        raise AssertionError(f"{args} exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return proc


@pytest.fixture(scope="module")
def ingested(corpus_dir):
    env = workspace_env(corpus_dir)
    out = run(env, "ingest", "--embedder", "hashing")
    return env, json.loads(out.stdout)


def ask(env, question, *extra):
    out = run(env, "ask", question, "--llm", "none", "--no-rerank", "--json", *extra)
    return json.loads(out.stdout)


def test_ingest_writes_manifest_for_fixture_corpus(ingested):
    _, manifest = ingested
    assert manifest["n_docs"] == 2
    assert manifest["embedder_fallback"] is True
    assert manifest["n_chunks"] >= 5


def test_ask_returns_cited_answer_from_gold_page(ingested):
    env, _ = ingested
    a = ask(env, "What does a sudden drop in draft pressure indicate?")
    assert a["status"] == "extractive"
    assert "blocked inlet duct" in a["answer"]
    assert {"doc": "fan_manual.pdf", "page": 1}.items() <= a["citations"][0].items()


@pytest.mark.parametrize(
    "question, doc, pages",
    [
        ("What does the rotary airlock valve do?", "fan_manual.pdf", {2}),
        # safety_sop p.3 repeats the opening of p.2, so either page is a correct source
        ("Why does dust on the fan vanes cause vibration and noise?", "safety_sop.pdf", {2, 3}),
    ],
)
def test_ask_retrieves_across_documents(ingested, question, doc, pages):
    env, _ = ingested
    a = ask(env, question)
    assert a["status"] == "extractive"
    top = a["citations"][0]
    assert top["doc"] == doc and top["page"] in pages


def test_ask_refuses_out_of_scope_question(ingested):
    env, _ = ingested
    a = ask(env, "Who won the football world cup in 1998?")
    assert a["status"] == "refused"
    assert a["citations"] == []
    assert a["config_hash"]


def test_ask_blocks_sensitive_question_before_retrieval(ingested):
    env, _ = ingested
    a = ask(env, "How do I bypass the interlock on the access door?")
    assert a["status"] == "blocked"
    assert a["guardrails"] == ["safety_bypass"]


def test_ask_rejects_oversized_question_with_usage_error(ingested):
    env, _ = ingested
    proc = run(env, "ask", "x" * 5000, "--llm", "none", "--no-rerank", check=False)
    assert proc.returncode == 2
    assert "question" in proc.stderr.lower()


def test_check_labels_passes_on_fixture_eval_set(ingested):
    env, _ = ingested
    out = run(env, "eval", "--check-labels")
    assert "0 unsupported" in out.stdout


def test_eval_writes_retrieval_metrics_to_configured_path(ingested, corpus_dir):
    env, _ = ingested
    run(env, "eval", "--no-answer-eval", "--no-rerank", "--llm", "none")
    with (corpus_dir / "evaluation.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    configs = {r["config"] for r in rows}
    assert configs == {"dense-only", "bm25-only", "hybrid-rrf"}
    recall5 = {r["config"]: float(r["value"]) for r in rows if r["metric"] == "recall@5"}
    assert recall5["hybrid-rrf"] == 1.0


def test_eval_does_not_touch_committed_results(ingested):
    """Tests must never overwrite the measured evaluation.csv in the repo."""
    committed = (PROTO / "evaluation.csv").read_bytes()
    env, _ = ingested
    run(env, "eval", "--no-answer-eval", "--no-rerank", "--llm", "none")
    assert (PROTO / "evaluation.csv").read_bytes() == committed
