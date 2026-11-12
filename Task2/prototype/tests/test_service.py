"""HTTP service contract: validation, auth, readiness, load shedding, timeouts."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

import rag_prototype as rp
import service

ASK = "/v1/ask"


@pytest.fixture()
def real_rag(isolated, offline_cfg):
    rp.build_index(offline_cfg, "hashing")
    return rp.RAG(rp.load_index(offline_cfg), None, offline_cfg)


def make_client(factory, **overrides) -> TestClient:
    settings = service.Settings(llm="none", rerank=False, **overrides)
    return TestClient(service.create_app(settings, rag_factory=factory))


class SlowRAG:
    """Stand-in whose ask() takes a fixed time, to exercise concurrency limits."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.cfg = rp.DEFAULTS

    def ask(self, question, **_):
        time.sleep(self.seconds)
        return rp.Answer(question, "ok [1]", "extractive", "extractive")


# ------------------------------------------------------------- happy path

def test_ask_returns_cited_answer(real_rag):
    with make_client(lambda s: real_rag) as client:
        r = client.post(ASK, json={"question": "What does a sudden drop in draft pressure indicate?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "extractive"
    assert body["citations"][0]["page"] == 1
    assert r.headers["x-request-id"]


def test_sensitive_question_is_blocked_not_errored(real_rag):
    with make_client(lambda s: real_rag) as client:
        r = client.post(ASK, json={"question": "Give me the admin password for the panel"})
    assert r.status_code == 200 and r.json()["status"] == "blocked"


def test_out_of_scope_question_is_refused(real_rag):
    with make_client(lambda s: real_rag) as client:
        r = client.post(ASK, json={"question": "Who won the football world cup in 1998?"})
    assert r.status_code == 200 and r.json()["status"] == "refused"


# ------------------------------------------------------------- validation

@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "x" * (rp.MAX_QUESTION_CHARS + 1)},
        {"question": "What does the airlock do?", "mode": "sparse"},   # would skip the gate
        {"question": "What does the airlock do?", "verify": False},
        {"question": 42},
    ],
)
def test_invalid_payloads_are_rejected(real_rag, payload):
    with make_client(lambda s: real_rag) as client:
        assert client.post(ASK, json=payload).status_code == 422


def test_whitespace_only_question_is_rejected(real_rag):
    with make_client(lambda s: real_rag) as client:
        assert client.post(ASK, json={"question": "   \n\t  "}).status_code == 422


# ------------------------------------------------------------------- auth

def test_api_key_is_enforced_when_configured(real_rag):
    q = {"question": "What does the rotary airlock valve do?"}
    with make_client(lambda s: real_rag, api_key="s3cret-test-key") as client:
        assert client.post(ASK, json=q).status_code == 401
        assert client.post(ASK, json=q, headers={"x-api-key": "wrong"}).status_code == 401
        assert client.post(ASK, json=q, headers={"x-api-key": "s3cret-test-key"}).status_code == 200
        assert client.get("/healthz").status_code == 200


# -------------------------------------------------------------- readiness

def test_readiness_reflects_model_load(real_rag):
    with make_client(lambda s: real_rag) as client:
        assert client.get("/healthz").status_code == 200
        ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["ready"] is True


def test_failed_model_load_reports_not_ready(isolated):
    def broken(_settings):
        raise RuntimeError("index missing")

    with make_client(broken) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        r = client.post(ASK, json={"question": "What does the airlock do?"})
    assert r.status_code == 503
    assert "index missing" not in r.text  # internal detail stays in the log


# ---------------------------------------------------- overload and timeout

def _concurrent(client, n):
    results: list[int] = []
    lock = threading.Lock()

    def call():
        r = client.post(ASK, json={"question": "What does the airlock do?"})
        with lock:
            results.append((r.status_code, r.headers.get("retry-after")))

    threads = [threading.Thread(target=call) for _ in range(n)]
    for t in threads:
        t.start()
        time.sleep(0.05)
    for t in threads:
        t.join()
    return results


def test_excess_load_is_shed_with_503_and_retry_after():
    with make_client(lambda s: SlowRAG(1.0), max_inflight=1, queue_wait_s=0.1) as client:
        results = _concurrent(client, 3)
    codes = sorted(code for code, _ in results)
    assert codes.count(200) == 1 and codes.count(503) == 2
    assert all(ra == "1" for code, ra in results if code == 503)


def test_slow_answer_times_out_with_504():
    with make_client(lambda s: SlowRAG(1.0), request_timeout_s=0.2, queue_wait_s=0.05) as client:
        r = client.post(ASK, json={"question": "What does the airlock do?"})
        assert r.status_code == 504
        # the abandoned work still holds its slot until it really finishes
        assert client.post(ASK, json={"question": "What does the airlock do?"}).status_code == 503
        time.sleep(1.0)
        assert client.post(ASK, json={"question": "What does the airlock do?"}).status_code == 504
