#!/usr/bin/env python3
"""
HTTP service for the RAG prototype: one process, models loaded once.

    uvicorn service:app --host 127.0.0.1 --port 8000

The CLI (`rag_prototype.py ask`) loads three models per question; a service
loads them once at startup and answers many questions. This file adds only
the serving concerns and reuses the pipeline unchanged:

* input validation at the boundary (length bounds, no unknown fields),
* optional API-key auth (`RAG_API_KEY`), compared in constant time,
* readiness that stays false until the index and models are loaded,
* a fixed number of in-flight answers; excess load gets 503 + Retry-After
  instead of an unbounded queue,
* a per-request timeout (504),
* one structured log line per request, without the question text.

Settings come from the environment, see `Settings.from_env`.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

import rag_prototype as rp

log = logging.getLogger("rag.service")


@dataclass(frozen=True)
class Settings:
    llm: str = rp.DEFAULTS["llm_model"]
    rerank: bool = True
    max_inflight: int = 1          # answers computed at once per process
    queue_wait_s: float = 2.0      # how long a request may wait for a slot
    request_timeout_s: float = 60.0
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            llm=env.get("RAG_LLM", cls.llm),
            rerank=env.get("RAG_RERANK", "1") != "0",
            max_inflight=max(1, int(env.get("RAG_MAX_INFLIGHT", cls.max_inflight))),
            queue_wait_s=float(env.get("RAG_QUEUE_WAIT_S", cls.queue_wait_s)),
            request_timeout_s=float(env.get("RAG_REQUEST_TIMEOUT_S", cls.request_timeout_s)),
            api_key=env.get("RAG_API_KEY") or None,
        )


def default_rag_factory(settings: Settings) -> rp.RAG:
    cfg = dict(rp.DEFAULTS)
    llm = None if settings.llm == "none" else settings.llm
    return rp.make_rag(cfg, llm, need_reranker=settings.rerank)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Retrieval mode is deliberately not a request field: sparse-only retrieval
    # skips the reranker and with it the calibrated relevance gate, so letting
    # clients pick it would let them switch a guardrail off.
    question: str = Field(min_length=rp.MIN_QUESTION_CHARS, max_length=rp.MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def _clean(cls, v: str) -> str:
        return rp.clean_question(v)   # ValueError -> 422


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    status: str
    mode: str
    citations: list[dict[str, Any]]
    guardrails: list[str]
    dropped_claims: int
    support: float
    top_score: float | None
    latency_ms: dict[str, float]
    config_hash: str


class _State:
    def __init__(self) -> None:
        self.rag: rp.RAG | None = None
        self.load_error: str | None = None
        self.slots: asyncio.Semaphore | None = None
        self.pool: ThreadPoolExecutor | None = None


def create_app(settings: Settings | None = None,
               rag_factory: Callable[[Settings], Any] | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    factory = rag_factory or default_rag_factory
    state = _State()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.slots = asyncio.Semaphore(settings.max_inflight)
        state.pool = ThreadPoolExecutor(max_workers=settings.max_inflight,
                                        thread_name_prefix="rag")
        try:
            state.rag = await asyncio.to_thread(factory, settings)
            log.info(json.dumps({"event": "ready", **_public(settings)}))
        except Exception as exc:  # stay up, report not-ready, keep the cause in the log
            state.load_error = type(exc).__name__
            log.exception("model/index load failed")
        yield
        state.pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Guarded RAG over technical manuals", version="1.0", lifespan=lifespan)

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key is None:
            return
        if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key):
            raise HTTPException(status_code=401, detail="missing or invalid API key")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        ready = state.rag is not None
        body = {"ready": ready, "load_error": state.load_error}
        return JSONResponse(body, status_code=200 if ready else 503)

    @app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
    async def ask(req: AskRequest, request: Request, response: Response) -> AskResponse:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        response.headers["x-request-id"] = request_id
        if state.rag is None:
            raise HTTPException(status_code=503, detail="index or models not loaded",
                                headers={"Retry-After": "5"})
        t0 = time.perf_counter()
        answer = await _bounded(state, settings, lambda: state.rag.ask(
            req.question, mode="hybrid", rerank=settings.rerank))
        log.info(json.dumps({
            "event": "ask", "request_id": request_id, "status": answer.status,
            "mode": answer.mode, "n_citations": len(answer.citations),
            "question_chars": len(req.question), "config_hash": answer.config_hash,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }))
        return AskResponse(request_id=request_id, **asdict(answer))

    return app


async def _bounded(state: _State, settings: Settings, fn: Callable[[], rp.Answer]) -> rp.Answer:
    """Run fn on the worker pool with a slot limit and a deadline.

    A request that cannot get a slot within queue_wait_s is shed with 503.
    On timeout the caller gets 504, but the worker thread cannot be killed:
    it runs to completion and keeps its slot until then, so a stuck model
    sheds new load instead of piling more threads onto the CPU.
    """
    try:
        await asyncio.wait_for(state.slots.acquire(), timeout=settings.queue_wait_s)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="server busy, retry shortly",
                            headers={"Retry-After": "1"}) from None

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(state.pool, fn)
    future.add_done_callback(lambda _: state.slots.release())
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout=settings.request_timeout_s)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="answer took too long") from None


def _public(settings: Settings) -> dict[str, Any]:
    d = asdict(settings)
    d["api_key"] = bool(settings.api_key)   # never log the key itself
    return d


logging.basicConfig(level=os.environ.get("RAG_LOG_LEVEL", "INFO"), format="%(message)s")
app = create_app()   # models load in lifespan, not at import
