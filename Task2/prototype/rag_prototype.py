#!/usr/bin/env python3
"""
Minimal, runnable RAG prototype for technical documentation Q&A.

Pipeline
--------
PDF -> page text -> boilerplate strip -> page-anchored chunks -> hybrid index
    (dense FAISS + BM25) -> RRF fusion -> cross-encoder rerank -> guarded answer
    (flan-t5-base, verified against the cited passages, with an extractive
    fallback) -> cited answer.

Design rules baked into the code
--------------------------------
1. Every answer is tied to page-level provenance. Citations are not decoration:
   an answer whose sentences are not supported by the cited passages is
   discarded and replaced by a verbatim extract, or by an explicit refusal.
2. The system refuses rather than guesses. A retrieval gate stops the LLM from
   being called when the corpus does not contain the answer.
3. Everything is measured. `eval` writes precision@k / recall@k / MRR / nDCG,
   citation-support rate, false-answer rate and latency, per configuration.

Usage
-----
    python rag_prototype.py ingest
    python rag_prototype.py ask "What causes vibration and noise from the fan?"
    python rag_prototype.py eval
    python rag_prototype.py eval --check-labels

Free and open source only: PyMuPDF, FAISS, sentence-transformers (Apache-2.0
models), flan-t5-base, plus numpy. No API keys, no paid services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
import time
import unicodedata
import zlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
# Paths default to this folder; the RAG_* variables let tests and services
# point the same code at another corpus or index without editing it.
DOCS_DIR = Path(os.environ.get("RAG_DOCS_DIR", HERE / "docs"))
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", HERE / "index"))
RUNS_DIR = Path(os.environ.get("RAG_RUNS_DIR", HERE / "runs"))
EVAL_SET = Path(os.environ.get("RAG_EVAL_SET", HERE / "eval_set.jsonl"))
EVAL_CSV = Path(os.environ.get("RAG_EVAL_CSV", HERE / "evaluation.csv"))

MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 1000       # longer inputs are pastes, not questions
QUERY_CACHE_MAX = 2048          # bounded so a long-running service cannot grow without limit

# --------------------------------------------------------------------------- #
# 0. Configuration
# --------------------------------------------------------------------------- #

DEFAULTS: dict[str, Any] = {
    # retrieval
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "chunk_size": 900,       # characters, ~150-190 English words
    "chunk_overlap": 180,    # ~20% of chunk_size, sentence-aligned
    "min_chunk": 220,
    "dense_k": 20,
    "sparse_k": 20,
    "final_k": 5,
    "rrf_k": 60,
    # gates (calibrated on the eval set, see notes.md section 4)
    "min_dense_score": 0.28,
    "min_rerank_score": 2.84,   # calibrated, see `calibrate` command and notes.md
    "min_support_ratio": 0.60,
    # generation
    "llm_model": "google/flan-t5-base",
    "max_new_tokens": 96,
    "max_prompt_chars": 3200,
}


# Curated document titles. First-line extraction is unreliable on these PDFs
# (5 of 11 yield a URL, an address, a version label or the Hindi letterhead),
# so sources would be cited as "Manual version:" instead of a document name.
DOC_TITLES: dict[str, str] = {
    "172-65231ma.pdf": "TLV Cyclone Separator DC7 (172-65231MA)",
    "43-86_Cyclone-Separator_IandO.pdf": "Cyclone Separator & Sight Flow Indicator - Installation and Piping",
    "654b45bb8292f586313470.pdf": "Cyclone Separator - Instructions for Use",
    "B_12a.pdf": "CAM Technical Guidance B.12: Cyclones",
    "Cyclone_Emission_Control.pdf": "Woodmaster Commercial Series Cyclone Separator (PN 7994-405)",
    "Cyclone_Manual.pdf": "Kice Cyclone Collector - Operators Manual (ATEX Zone 22)",
    "D13.pdf": "Design of Cyclone Dust Collector - B.Tech Project Report",
    "Operating-and-maintenance-manual-cyclone-separator.pdf": "Dustair Cyclonic Dust Separator - Operating and Maintenance",
    "Parker_DustHog_C_Series_User_Guide.pdf": "Parker DustHog C Series High Efficiency Cyclone - User Guide",
    "SOP-Cyclone-22022023.pdf": "DG Shipping Cyclone Disaster Management SOP (2023)",
    "sect7a.pdf": "Introduction to Cyclones",
}


def fingerprint(cfg: dict[str, Any]) -> str:
    """Hash of the settings that change what is indexed or how it is scored."""
    keys = ["chunk_size", "chunk_overlap", "min_chunk", "embed_model", "rerank_model"]
    blob = json.dumps({k: cfg[k] for k in keys}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 1. Ingestion and chunking
# --------------------------------------------------------------------------- #

_CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\ufffd]")
_WS = re.compile(r"[ \t\u00a0]+")
_PAGENO = re.compile(r"^\s*(page\s+)?\d{1,3}\s*(of\s+\d{1,3})?\s*$", re.I)
_HEADING_NUM = re.compile(r"^\s*(\d+(\.\d+){0,3})[.)]?\s+[A-Z(\"]")
_HEADING_CAPS = re.compile(r"^[A-Z0-9 ,.\-/&()'\"]{8,90}$")


def normalize_text(raw: str) -> str:
    """NFKC, drop control/replacement chars, de-hyphenate line breaks."""
    t = unicodedata.normalize("NFKC", raw)
    t = _CTRL.sub(" ", t)
    t = t.replace("\u00ad", "")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = _WS.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _is_heading(line: str) -> bool:
    s = line.strip()
    if len(s) < 4 or len(s) > 90:
        return False
    if s.endswith((".", ";", ",")) and not _HEADING_CAPS.match(s):
        return False
    return bool(_HEADING_NUM.match(s) or _HEADING_CAPS.match(s))


def strip_boilerplate(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove lines repeated across most pages of a document (headers/footers)."""
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for p in pages:
        by_doc.setdefault(p["doc"], []).append(p)

    cleaned: list[dict[str, Any]] = []
    for doc, plist in by_doc.items():
        counts: dict[str, int] = {}
        for p in plist:
            seen = {
                ln.strip().lower()
                for ln in p["text"].splitlines()
                if 0 < len(ln.strip()) <= 90
            }
            for ln in seen:
                counts[ln] = counts.get(ln, 0) + 1
        threshold = max(3, int(0.4 * len(plist)))
        boiler = {ln for ln, c in counts.items() if c >= threshold}

        for p in plist:
            kept = []
            for ln in p["text"].splitlines():
                low = ln.strip().lower()
                if not low:
                    kept.append(ln)
                    continue
                if low in boiler or _PAGENO.match(low):
                    continue
                kept.append(ln)
            cleaned.append({"doc": doc, "page": p["page"], "text": "\n".join(kept)})
    return cleaned


def _pymupdf():
    try:
        import pymupdf  # PyMuPDF >= 1.24.3; the legacy `fitz` name is deprecated
    except ImportError:
        import fitz as pymupdf
    return pymupdf


def load_pages(docs_dir: Path) -> list[dict[str, Any]]:
    fitz = _pymupdf()

    pages: list[dict[str, Any]] = []
    for pdf in sorted(docs_dir.glob("*.pdf")):
        with fitz.open(pdf) as doc:
            for i, page in enumerate(doc):
                text = normalize_text(page.get_text())
                if len(text) < 40:          # scanned page with no text layer
                    continue
                pages.append({"doc": pdf.name, "page": i + 1, "text": text})
    return strip_boilerplate(pages)


def _doc_title(pages: list[dict[str, Any]], doc: str) -> str:
    """Curated title when the file is known, else the best first-page line."""
    if doc in DOC_TITLES:
        return DOC_TITLES[doc]
    for p in pages:
        if p["doc"] != doc:
            continue
        for line in p["text"].splitlines():
            s = line.strip()
            if 12 <= len(s) <= 110 and not _PAGENO.match(s):
                return re.sub(r"\s+", " ", s)
    return re.sub(r"[_-]+", " ", Path(doc).stem).strip()


def chunk_pages(pages: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Page-anchored, heading-aware, sentence-aligned sliding-window chunks."""
    size, overlap, min_size = cfg["chunk_size"], cfg["chunk_overlap"], cfg["min_chunk"]
    titles = {d: _doc_title(pages, d) for d in {p["doc"] for p in pages}}
    chunks: list[dict[str, Any]] = []

    for p in pages:
        paras = [s.strip() for s in re.split(r"\n\s*\n", p["text"]) if s.strip()]
        buf: list[str] = []
        buf_len = 0
        section = ""
        counter = 0

        def flush() -> None:
            nonlocal buf, buf_len, counter
            body = "\n".join(buf).strip()
            if (len(body) < min_size and chunks and chunks[-1]["doc"] == p["doc"]
                    and chunks[-1]["page"] == p["page"]):
                # glue a short tail onto the previous chunk of the same page only;
                # gluing across pages would cite page N for text from page N+1
                chunks[-1]["text"] += "\n" + body
                buf, buf_len = [], 0
                return
            if not body:
                return
            counter += 1
            chunks.append(
                {
                    "chunk_id": f"{Path(p['doc']).stem}::p{p['page']}::c{counter}",
                    "doc": p["doc"],
                    "doc_title": titles[p["doc"]],
                    "page": p["page"],
                    "section": section,
                    "text": body,
                    "n_chars": len(body),
                }
            )
            # sentence-aligned overlap: keep the tail as the next chunk's seed
            tail = body[-overlap:] if overlap else ""
            cut = tail.find(". ")
            if tail and cut != -1:
                tail = tail[cut + 2 :]
            buf = [tail] if len(tail) > 60 else []
            buf_len = len(tail) if buf else 0

        for para in paras:
            if _is_heading(para.split("\n")[0]):
                head = para.split("\n")[0].strip()
                section = head[:120]
            if buf_len and buf_len + len(para) > size:
                flush()
            if len(para) > size:                      # long table or dense paragraph
                sentences = re.split(r"(?<=[.!?])\s+", para)
                piece = ""
                for s in sentences:
                    if len(piece) + len(s) > size and piece:
                        if buf:
                            buf.append(piece)
                            buf_len += len(piece)
                            flush()
                        else:
                            buf, buf_len = [piece], len(piece)
                            flush()
                        piece = s
                    else:
                        piece = (piece + " " + s).strip()
                if piece:
                    buf.append(piece)
                    buf_len += len(piece)
                continue
            buf.append(para)
            buf_len += len(para) + 1

        if buf:
            flush()

    for i, c in enumerate(chunks):
        c["ordinal"] = i
    return chunks


# --------------------------------------------------------------------------- #
# 2. Sparse retrieval: BM25
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    """a an the of to in for on at by with from as is are was were be been being
    and or but if then than that this these those it its into over under between
    can could should would may might must do does did not no nor so such which who
    what when where why how all any both each few more most other some only own
    same too very s t will just don now""".split()
)


def _stem(w: str) -> str:
    for suf in ("ing", "ies", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            base = w[: -len(suf)]
            if suf == "ies":
                return base + "y"
            return base
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _TOKEN.findall(text.lower()) if w not in _STOP and len(w) > 1]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: list[list[str]] = []
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avglen = 1.0

    def fit(self, texts: Iterable[str]) -> "BM25":
        self.docs = [tokenize(t) for t in texts]
        self.tf = [dict(sorted({w: d.count(w) for w in set(d)}.items())) for d in self.docs]
        df: dict[str, int] = {}
        for tf in self.tf:
            for w in tf:
                df[w] = df.get(w, 0) + 1
        n = max(1, len(self.docs))
        self.df = df
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.avglen = statistics.fmean(len(d) for d in self.docs) if self.docs else 1.0
        return self

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.docs), dtype=np.float32)
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                f = tf[w]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avglen)
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / denom
            out[i] = s
        return out


# --------------------------------------------------------------------------- #
# 3. Dense retrieval: FAISS + sentence-transformers
# --------------------------------------------------------------------------- #

class HashingEmbedder:
    """Deterministic offline fallback: hashed char+word n-grams, L2-normalised.

    Not competitive with a trained encoder; it exists so the demo runs with no
    model download and so the eval can quantify what the encoder adds.
    """

    name = "hashing-ngram"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.idf: dict[str, float] = {}

    def _feats(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{w} {v}" for w, v in zip(words, words[1:])]
        feats += [f"c:{w[i:i + 5]}" for w in words if len(w) > 4 for i in range(len(w) - 4)]
        return feats

    def _bucket(self, feat: str) -> int:
        # crc32, not hash(): built-in str hashing is salted per process, so the
        # index (built by `ingest`) and the query (embedded by `ask`) would land
        # in different buckets.
        return zlib.crc32(feat.encode("utf-8")) % self.dim

    def fit(self, texts: list[str]) -> "HashingEmbedder":
        df: dict[int, int] = {}
        for t in texts:
            for h in {self._bucket(f) for f in self._feats(t)}:
                df[h] = df.get(h, 0) + 1
        n = max(1, len(texts))
        self.idf = {h: math.log((n + 1) / (c + 1)) + 1.0 for h, c in df.items()}
        return self

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f in self._feats(t):
                h = self._bucket(f)
                out[i, h] += self.idf.get(h, 1.0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, np.maximum(norms, 1e-9), out=out)
        return out


def load_embedder(cfg: dict[str, Any], kind: str):
    if kind == "hashing":
        return HashingEmbedder()
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(cfg["embed_model"], device="cpu", local_files_only=True)
    except Exception:
        return SentenceTransformer(cfg["embed_model"], device="cpu")


class VectorIndex:
    """FAISS IndexFlatIP over L2-normalised embeddings (exact cosine search)."""

    def __init__(self, dim: int = 0) -> None:
        self.dim = dim
        self.index = None

    def build(self, emb: np.ndarray) -> "VectorIndex":
        import faiss

        emb = np.ascontiguousarray(emb.astype(np.float32))
        self.dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)
        return self

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(q.astype(np.float32))
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        return scores[0], ids[0]


# --------------------------------------------------------------------------- #
# 4. Retrieval layer
# --------------------------------------------------------------------------- #

@dataclass
class Hit:
    idx: int
    chunk: dict[str, Any]
    dense: float = 0.0
    sparse: float = 0.0
    rrf: float = 0.0
    rerank: float | None = None

    @property
    def score(self) -> float:
        return self.rerank if self.rerank is not None else self.rrf


class Retriever:
    def __init__(self, chunks, emb, cfg, embedder, bm25: BM25, reranker=None):
        self.chunks = chunks
        self.cfg = cfg
        self.embedder = embedder
        self.bm25 = bm25
        self.vectors = VectorIndex().build(emb)
        self._reranker = reranker
        self._query_cache: dict[str, np.ndarray] = {}

    # -- stage 1a ---------------------------------------------------------- #
    def dense(self, question: str, k: int) -> list[tuple[int, float]]:
        q = self._query_cache.get(question)
        if q is None:
            q = self.embedder.encode([question], normalize_embeddings=True)
            q = np.asarray(q, dtype=np.float32)
            if len(self._query_cache) >= QUERY_CACHE_MAX:
                self._query_cache.clear()
            self._query_cache[question] = q
        scores, ids = self.vectors.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids, scores) if i >= 0]

    # -- stage 1b ---------------------------------------------------------- #
    def sparse(self, question: str, k: int) -> list[tuple[int, float]]:
        s = self.bm25.scores(question)
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]

    # -- stage 2 ----------------------------------------------------------- #
    def retrieve(self, question: str, mode: str = "hybrid", rerank: bool = True) -> list[Hit]:
        cfg = self.cfg
        pool: dict[int, Hit] = {}

        if mode in {"dense", "hybrid"}:
            for rank, (i, s) in enumerate(self.dense(question, cfg["dense_k"])):
                h = pool.setdefault(i, Hit(i, self.chunks[i]))
                h.dense = s
                h.rrf += 1.0 / (cfg["rrf_k"] + rank + 1)
        if mode in {"sparse", "hybrid"}:
            for rank, (i, s) in enumerate(self.sparse(question, cfg["sparse_k"])):
                h = pool.setdefault(i, Hit(i, self.chunks[i]))
                h.sparse = s
                h.rrf += 1.0 / (cfg["rrf_k"] + rank + 1)

        hits = sorted(pool.values(), key=lambda h: h.rrf, reverse=True)
        if rerank and mode != "sparse" and self.reranker is not None:
            cand = hits[: max(cfg["final_k"] * 4, 20)]
            pairs = [(question, h.chunk["text"]) for h in cand]
            scores = self.reranker.predict(pairs)
            for h, s in zip(cand, np.atleast_1d(scores)):
                h.rerank = float(s)
            hits = sorted(cand, key=lambda h: h.rerank, reverse=True) + hits[len(cand):]
        return hits[: cfg["final_k"]]

    @property
    def reranker(self):
        if self._reranker is None and self.cfg.get("_rerank_model"):
            from sentence_transformers import CrossEncoder

            try:
                self._reranker = CrossEncoder(self.cfg["rerank_model"], device="cpu", local_files_only=True)
            except Exception:
                self._reranker = CrossEncoder(self.cfg["rerank_model"], device="cpu")
        return self._reranker


# --------------------------------------------------------------------------- #
# 5. Guardrails
# --------------------------------------------------------------------------- #

@dataclass
class GuardrailVerdict:
    blocked: bool
    rule: str = ""
    message: str = ""


_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "credential_request",
        re.compile(r"\b(password|passcode|pass ?phrase|api[_ -]?key|secret|access token|credentials?|otp|private key)\b", re.I),
        "I can't retrieve or reproduce credentials. Ask your IT/controls owner for access through the approved process.",
    ),
    (
        "safety_bypass",
        re.compile(
            r"\b(bypass|disable|override|defeat|skip|circumvent|turn off)\b.{0,60}"
            r"\b(interlock|interlocks|guard|trip|alarm|lockout|tagout|loto|safety|shutdown|protection|rotary valve)\b",
            re.I,
        ),
        "I can't advise on disabling interlocks, guards or lockout/tagout. Use the manufacturer's procedure and your site permit-to-work system.",
    ),
    (
        "prompt_injection",
        re.compile(r"(ignore (all )?(previous|prior|above) instructions|system prompt|you are now|reveal your instructions)", re.I),
        "That request asks me to change my operating rules, which I won't do. Ask a documentation question instead.",
    ),
    (
        "personal_data",
        re.compile(r"\b(home address|aadhaar|social security number|personal phone number|date of birth)\b", re.I),
        "I don't hold or return personal data. Contact the data owner for that request.",
    ),
    (
        "regulated_advice",
        re.compile(r"\b(legal advice|medical advice|investment advice|prescribe|diagnose me)\b", re.I),
        "That is regulated professional advice, which is out of scope for this documentation assistant.",
    ),
]


def clean_question(raw: str) -> str:
    """Boundary check for user input: strip control characters, bound the length."""
    if not isinstance(raw, str):
        raise ValueError("question must be text")
    q = " ".join(_CTRL.sub(" ", raw).split())
    if len(q) < MIN_QUESTION_CHARS:
        raise ValueError(f"question must be at least {MIN_QUESTION_CHARS} characters")
    if len(q) > MAX_QUESTION_CHARS:
        raise ValueError(f"question must be at most {MAX_QUESTION_CHARS} characters")
    return q


def screen_query(question: str) -> GuardrailVerdict:
    for rule, pattern, message in _RULES:
        if pattern.search(question):
            return GuardrailVerdict(True, rule, message)
    return GuardrailVerdict(False)


_INJECTION_LINE = re.compile(r"(ignore (all )?(previous|prior) instructions|system\s*:|assistant\s*:)", re.I)


def sanitize_passage(text: str) -> str:
    """Retrieved text is untrusted data: drop instruction-like lines."""
    kept = [ln for ln in text.splitlines() if not _INJECTION_LINE.search(ln)]
    return "\n".join(kept).strip()


def _digits(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def support_ratio(sentence: str, passage_tokens: set[str], passage_text: str) -> float:
    toks = set(tokenize(sentence))
    if not toks:
        return 0.0
    overlap = len(toks & passage_tokens) / len(toks)
    # every number in a claim must exist verbatim in the supporting passage
    nums = _digits(sentence)
    if nums and not nums.issubset(_digits(passage_text)):
        return 0.0
    return overlap


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass
class Verified:
    answer: str
    kept: list[str]
    dropped: list[str]
    supported: bool
    support: float
    supporting: list[int] = field(default_factory=list)


def verify_answer(
    draft: str, passages: list[dict[str, Any]], threshold: float
) -> Verified:
    """Keep only sentences that a passage actually supports.

    The LLM never writes citation markers: for each sentence we find the
    best-supporting passage by content-token overlap (numbers must match
    verbatim), keep the sentence only above `threshold`, and derive the
    citation list from that. Unsupported sentences are dropped and counted.
    """
    draft = draft.strip()
    if not draft:
        return Verified("", [], [], False, 0.0, [])
    sentences = [s.strip() for s in _SENT_SPLIT.split(draft) if s.strip()]
    kept, dropped, ratios, supporting = [], [], [], []
    for s in sentences:
        best_i, best_r = 0, 0.0
        for i, p in enumerate(passages, start=1):
            r = support_ratio(s, set(tokenize(p["text"])), p["text"])
            if r > best_r:
                best_i, best_r = i, r
        ratios.append(best_r)
        if best_r >= threshold:
            kept.append(s)
            if best_i and best_i not in supporting:
                supporting.append(best_i)
        else:
            dropped.append(s)
    support = statistics.fmean(ratios) if ratios else 0.0
    return Verified(" ".join(kept), kept, dropped, bool(kept), support, sorted(supporting))


# --------------------------------------------------------------------------- #
# 6. Generation
# --------------------------------------------------------------------------- #

PROMPT = """Answer the question using only the passages. Give a short answer of at most three sentences.
Copy numbers exactly as written. If the passages do not contain the answer, reply exactly NOT_ANSWERABLE.

{passages}

Question: {question}
Answer:"""

REFUSALS = {
    "sensitive": "I can't help with that request. {detail}",
    "no_context": (
        "I couldn't find this in the indexed documents. Closest indexed topics: {topics}. "
        "If the answer should be there, the manual may not be in the corpus yet."
    ),
    "not_answerable": (
        "The retrieved passages don't answer that question, so I won't guess. "
        "Try naming the equipment, symptom and document section."
    ),
}


class FlanT5Generator:
    def __init__(self, model_path: str, max_new_tokens: int = 96) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = Path(model_path).name
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.calls = 0

    def generate(self, prompt: str) -> str:
        ids = self.tok(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids
        with self.torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=self.max_new_tokens, num_beams=1, do_sample=False
            )
        self.calls += 1
        return self.tok.decode(out[0], skip_special_tokens=True).strip()


class ExtractiveGenerator:
    """No-LLM baseline: the best-matching sentence, quoted verbatim."""

    name = "extractive"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        raise RuntimeError("extractive mode bypasses prompting")


def extractive_answer(question: str, passages: list[dict[str, Any]]) -> tuple[str, int]:
    """Best-matching sentence, quoted verbatim, plus its passage number."""
    q = set(tokenize(question))
    best, best_score, best_i = "", 0.0, 0
    for i, p in enumerate(passages, start=1):
        for s in _SENT_SPLIT.split(p["text"]):
            toks = set(tokenize(s))
            if len(toks) < 5:
                continue
            score = len(toks & q) / (len(q) or 1)
            if score > best_score:
                best, best_score, best_i = s.strip(), score, i
    if best_score < 0.25:
        return "", 0
    return best, best_i


# --------------------------------------------------------------------------- #
# 7. Orchestration: guarded answer
# --------------------------------------------------------------------------- #

@dataclass
class Answer:
    question: str
    answer: str
    status: str                     # answered | extractive | refused | blocked
    mode: str                       # llm | extractive | none
    citations: list[dict[str, Any]] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    dropped_claims: int = 0
    support: float = 0.0
    top_score: float | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    config_hash: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def build_prompt(question: str, hits: list[Hit], max_chars: int) -> str:
    blocks, used = [], 0
    for i, h in enumerate(hits, start=1):
        text = sanitize_passage(h.chunk["text"])
        if used + len(text) > max_chars:
            text = text[: max(0, max_chars - used)]
        if not text:
            break
        used += len(text)
        blocks.append(f"Passage from {h.chunk['doc']}, page {h.chunk['page']}:\n{text}")
    return PROMPT.format(passages="\n\n".join(blocks), question=question)


def is_usable_answer(question: str, text: str) -> tuple[bool, str]:
    """Reject degenerate drafts before they are shown to anyone.

    Two cheap, explainable checks:
      * enough content to be an answer (5+ content tokens), and
      * a quantitative question must get a number (how often / what pressure ...).
    Everything else falls through to the verbatim extract.
    """
    toks = tokenize(text)
    if len(toks) < 5:
        return False, "too_short"
    quantitative = re.search(
        r"\b(how (often|many|much|long|far|hot|cold)|what (pressure|temperature|speed|voltage|"
        r"capacity|diameter|weight|ratio|frequency|size)|at what)\b", question, re.I)
    if quantitative and not re.search(r"\d", text):
        return False, "quantitative_without_number"
    return True, ""


class RAG:
    def __init__(self, retriever: Retriever, generator=None, cfg: dict[str, Any] | None = None):
        self.r = retriever
        self.g = generator
        self.cfg = cfg or retriever.cfg

    def ask(self, question: str, mode: str = "hybrid", rerank: bool = True,
            verify: bool = True, allow_llm: bool = True, select: bool = True) -> Answer:
        t0 = time.perf_counter()
        cfg = self.cfg
        verdict = screen_query(question)
        if verdict.blocked:
            return Answer(question, REFUSALS["sensitive"].format(detail=verdict.message),
                          "blocked", "none", guardrails=[verdict.rule],
                          config_hash=fingerprint(cfg))

        hits = self.r.retrieve(question, mode=mode, rerank=rerank)
        t1 = time.perf_counter()
        if not hits:
            return Answer(question, REFUSALS["no_context"].format(topics="none"), "refused", "none",
                          guardrails=["retrieval_gate"],
                          latency_ms={"retrieve": _ms(t0, t1), "total": _ms(t0, t1)},
                          config_hash=fingerprint(cfg))

        top = hits[0].score
        if hits[0].rerank is not None:
            gate_ok = top >= cfg["min_rerank_score"]
        elif mode in {"dense", "hybrid"}:
            # RRF scores are rank-based and not comparable to a cut-off, so without
            # the reranker the gate falls back to the best cosine among the hits
            # instead of switching off.
            gate_ok = max(h.dense for h in hits) >= cfg["min_dense_score"]
        else:
            gate_ok = True   # sparse-only: no hits already refuses above; BM25 has no absolute scale
        if not gate_ok:
            topics = ", ".join(sorted({h.chunk["doc_title"] for h in hits[:3]}))
            return Answer(question, REFUSALS["no_context"].format(topics=topics), "refused", "none",
                          guardrails=["retrieval_gate"], top_score=top,
                          latency_ms={"retrieve": _ms(t0, t1), "total": _ms(t0, t1)},
                          config_hash=fingerprint(cfg))

        # Passage numbers are per request. They are not written into h.chunk:
        # chunks are shared by every request in a long-running service.
        citations = [
            {
                "n": i,
                "doc": h.chunk["doc"],
                "page": h.chunk["page"],
                "section": h.chunk["section"],
                "chunk_id": h.chunk["chunk_id"],
                "score": round(float(h.score), 4),
            }
            for i, h in enumerate(hits, start=1)
        ]

        def pick(numbers: list[int]) -> list[dict[str, Any]]:
            return [c for c in citations if c["n"] in numbers]

        draft = ""
        mode_used = "none"
        if allow_llm and self.g is not None and not isinstance(self.g, ExtractiveGenerator):
            draft = self.g.generate(build_prompt(question, hits, cfg["max_prompt_chars"]))
            mode_used = "llm"
        t2 = time.perf_counter()

        refused_by_llm = re.search(r"NOT_ANSWERABLE|not answerable", draft, re.I) is not None
        if not verify:
            # Unguarded baseline for the eval only: ship the raw draft with the
            # top passage as a naive citation, exactly what a plain RAG does.
            if draft and not refused_by_llm:
                return Answer(question, draft, "answered", mode_used, citations=pick([1]),
                              guardrails=["verification_disabled"], support=0.0,
                              top_score=round(float(top), 4),
                              latency_ms={"retrieve": _ms(t0, t1), "generate": _ms(t1, t2),
                                          "total": _ms(t0, t2)},
                              config_hash=fingerprint(cfg))
            answer, status, mode_used = REFUSALS["not_answerable"], "refused", "none"
            return Answer(question, answer, status, mode_used, citations=[],
                          guardrails=["llm_refused"], top_score=round(float(top), 4),
                          latency_ms={"retrieve": _ms(t0, t1), "generate": _ms(t1, t2),
                                      "total": _ms(t0, t2)},
                          config_hash=fingerprint(cfg))

        checked = verify_answer(draft, [h.chunk for h in hits[:3]], cfg["min_support_ratio"]) if (draft and not refused_by_llm) else Verified("", [], [], False, 0.0)

        llm_text, llm_citations = "", []
        if checked.supported and not refused_by_llm:
            usable, _ = is_usable_answer(question, checked.answer)
            if usable:
                llm_text = checked.answer
                llm_citations = pick(checked.supporting)

        extract, src = extractive_answer(question, [h.chunk for h in hits[:2]])

        # Two candidates, one decision: whichever answers the question better
        # according to the cross-encoder. The 250M model is not trusted to win
        # by default; on this corpus it often should not.
        chosen, notes = "none", []
        if llm_text and extract and select and self.r.reranker is not None:
            s = np.atleast_1d(self.r.reranker.predict([(question, llm_text), (question, extract)]))
            if float(s[1]) > float(s[0]):
                chosen = "extractive"
                notes.append("extract_preferred_over_llm")
            else:
                chosen = "llm"
        elif llm_text:
            chosen = "llm"
        elif extract:
            chosen = "extractive"

        if chosen == "llm":
            answer = f"{llm_text} {''.join(f'[{n}]' for n in checked.supporting)}".strip()
            citations, status, mode_used = llm_citations or pick(checked.supporting), "answered", "llm"
        elif chosen == "extractive":
            answer, status, mode_used = f"{extract} [{src}]", "extractive", "extractive"
            citations = pick([src])
        else:
            answer, status, mode_used = REFUSALS["not_answerable"], "refused", "none"
            citations = []
        t3 = time.perf_counter()

        return Answer(
            question=question,
            answer=answer,
            status=status,
            mode=mode_used,
            citations=citations,
            guardrails=(["llm_refused"] if refused_by_llm else []) + (["claims_dropped"] if checked.dropped else []) + notes,
            dropped_claims=len(checked.dropped),
            support=round(checked.support, 3),
            top_score=round(float(top), 4),
            latency_ms={"retrieve": _ms(t0, t1), "generate": _ms(t1, t2), "verify": _ms(t2, t3),
                        "total": _ms(t0, t3)},
            config_hash=fingerprint(cfg),
        )


def _ms(a: float, b: float) -> float:
    return round((b - a) * 1000, 1)


def format_answer(a: Answer) -> str:
    lines = [a.answer]
    if a.status in {"answered", "extractive"} and a.citations:
        lines.append("")
        lines.append("Sources:")
        for c in a.citations:
            if c["n"] in _cited_numbers(a.answer):
                sec = f" — {c['section']}" if c["section"] else ""
                lines.append(f"  [{c['n']}] {c['doc']}, p.{c['page']}{sec}")
    if a.guardrails:
        lines.append("")
        lines.append(f"Guardrails: {', '.join(a.guardrails)}; dropped unsupported claims: {a.dropped_claims}")
    lines.append("")
    lines.append(f"status={a.status} mode={a.mode} "
                 f"latency={a.latency_ms.get('total', 0)}ms support={a.support} cfg={a.config_hash}")
    return "\n".join(lines)


def _cited_numbers(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", text)}


# --------------------------------------------------------------------------- #
# 8. Index build / load
# --------------------------------------------------------------------------- #

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_index(cfg: dict[str, Any], embedder_kind: str = "st") -> dict[str, Any]:
    pages = load_pages(DOCS_DIR)
    if not pages:
        raise SystemExit(f"No text-layer PDFs found in {DOCS_DIR}")
    chunks = chunk_pages(pages, cfg)
    embedder = load_embedder(cfg, embedder_kind)
    if isinstance(embedder, HashingEmbedder):
        embedder.fit([c["text"] for c in chunks])
    emb = np.asarray(
        embedder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                        batch_size=32, show_progress_bar=False),
        dtype=np.float32,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", emb)
    with (INDEX_DIR / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_hash": fingerprint(cfg),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "embedder": getattr(embedder, "name", cfg["embed_model"]),
        "embedder_fallback": isinstance(embedder, HashingEmbedder),
        "n_docs": len({c["doc"] for c in chunks}),
        "n_pages": len(pages),
        "n_chunks": len(chunks),
        "avg_chunk_chars": round(statistics.fmean(c["n_chars"] for c in chunks), 1),
        "corpus_sha256": sha256_files(list(DOCS_DIR.glob("*.pdf"))),
        "python": platform.python_version(),
        "versions": _versions(),
    }
    (INDEX_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _versions() -> dict[str, str]:
    out = {}
    for mod in ("numpy", "faiss", "sentence_transformers", "transformers", "torch", "pymupdf"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = "not installed"
    return out


def load_index(cfg: dict[str, Any], embedder_kind: str = "st", need_reranker: bool = False) -> Retriever:
    chunks_path = INDEX_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit("No index found. Run: python rag_prototype.py ingest")
    manifest = json.loads((INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_hash"] != fingerprint(cfg):
        print(f"[warn] index built with a different config hash "
              f"({manifest['config_hash']} != {fingerprint(cfg)}); rerun ingest if chunking changed.",
              file=sys.stderr)
    chunks = read_jsonl(chunks_path)
    emb = np.load(INDEX_DIR / "embeddings.npy")
    embedder = load_embedder(cfg, "hashing" if manifest.get("embedder_fallback") else embedder_kind)
    if isinstance(embedder, HashingEmbedder):
        # the idf weights are derived from the chunks, so refit them exactly as ingest did
        embedder.fit([c["text"] for c in chunks])
    bm25 = BM25().fit([c["text"] for c in chunks])
    return Retriever(chunks, emb, cfg, embedder, bm25, reranker=_LazyReranker(cfg) if need_reranker else None)


class _LazyReranker:
    """Created on first use so retrieval-only runs never pay the load cost."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._m = None

    @property
    def model(self):
        if self._m is None:
            from sentence_transformers import CrossEncoder

            try:
                self._m = CrossEncoder(self.cfg["rerank_model"], device="cpu", local_files_only=True)
            except Exception:
                self._m = CrossEncoder(self.cfg["rerank_model"], device="cpu")
        return self._m

    def predict(self, pairs):
        return self.model.predict(pairs)


GENERATOR_FILES = ["*.json", "*.model", "model.safetensors"]   # not the flax/tf/.bin copies


def download_models(cfg: dict[str, Any], llm: str | None) -> int:
    """Fetch the three models once, so later runs need no network.

    The generator goes to ./models/<name> with only the files transformers
    loads; a plain snapshot of google/flan-t5-base would also pull the Flax,
    TensorFlow and PyTorch .bin copies (~3 GB more).
    """
    from huggingface_hub import snapshot_download
    from sentence_transformers import CrossEncoder, SentenceTransformer

    for label, loader, model_id in (("embedder", SentenceTransformer, cfg["embed_model"]),
                                    ("reranker", CrossEncoder, cfg["rerank_model"])):
        loader(model_id, device="cpu")
        print(f"{label:9s} {model_id}: cached")
    if llm and llm != "none" and not Path(llm).exists():
        target = HERE / "models" / Path(llm).name
        snapshot_download(llm, local_dir=str(target), allow_patterns=GENERATOR_FILES)
        print(f"generator {llm}: {target}")
    return 0


def make_rag(cfg: dict[str, Any], llm: str | None, need_reranker: bool = True) -> RAG:
    retriever = load_index(cfg, need_reranker=need_reranker)
    generator = None
    if llm and llm != "none":
        path = llm
        if path.startswith("google/flan-t5"):
            local = HERE / "models" / Path(path).name
            path = str(local) if local.exists() else path
        generator = FlanT5Generator(path, cfg["max_new_tokens"])
    return RAG(retriever, generator, cfg)


# --------------------------------------------------------------------------- #
# 9. Evaluation
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def token_f1(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return 0.0
    pt, gt = p.split(), g.split()
    common = {}
    for w in pt:
        common[w] = min(pt.count(w), gt.count(w))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pt), overlap / len(gt)
    return 2 * precision * recall / (precision + recall)


def ndcg_at_k(relevances: list[int], k: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(relevances[:k]))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _is_gold(hit: Hit, item: dict[str, Any]) -> bool:
    return hit.chunk["doc"] == item["gold_doc"] and hit.chunk["page"] in item["gold_pages"]


def score_retrieval(item: dict[str, Any], hits: list[Hit]) -> dict[str, float]:
    gold = [1 if _is_gold(h, item) else 0 for h in hits]
    first = next((i for i, g in enumerate(gold, start=1) if g), None)
    return {
        "recall@1": float(bool(gold[:1] and gold[0])),
        "recall@5": float(any(gold[:5])),
        "recall@10": float(any(gold[:10])),
        "precision@5": sum(gold[:5]) / max(1, min(5, len(gold))),
        "mrr@10": (1.0 / first) if first and first <= 10 else 0.0,
        "ndcg@5": ndcg_at_k(gold, 5),
    }


def calibrate(cfg: dict[str, Any], mode: str = "hybrid", rerank: bool = True) -> int:
    """Pick the relevance-gate threshold from the eval data, not from taste.

    Sweep candidate cut-offs on top-1 scores; choose the one that maximises
    balanced accuracy between answerable and unanswerable questions.
    """
    items = [i for i in read_jsonl(EVAL_SET) if i["type"] != "sensitive"]
    retriever = load_index(cfg, need_reranker=rerank)
    rows = []
    for item in items:
        hits = retriever.retrieve(item["question"], mode=mode, rerank=rerank)
        h = hits[0]
        rows.append({"id": item["id"], "type": item["type"],
                     "dense": round(h.dense, 4),
                     "rerank": None if h.rerank is None else round(h.rerank, 3),
                     "rrf": round(h.rrf, 5)})

    pos = [r for r in rows if r["type"] == "answerable"]
    neg = [r for r in rows if r["type"] != "answerable"]
    for field_name in ("dense", "rerank"):
        if any(r[field_name] is None for r in rows):
            continue
        p = sorted(r[field_name] for r in pos)
        n = sorted(r[field_name] for r in neg)
        cands = sorted({round(v, 2) for v in p + n})
        best = max(
            cands,
            key=lambda t: 0.5 * (sum(v >= t for v in p) / len(p) + sum(v < t for v in n) / len(n)),
        )
        tpr = sum(v >= best for v in p) / len(p)
        tnr = sum(v < best for v in n) / len(n)
        print(f"{field_name:7s} answerable p05={p[max(0, len(p)//20)]:7.3f} "
              f"min={p[0]:7.3f} | unanswerable max={n[-1]:7.3f} | "
              f"cut={best:7.2f} TPR={tpr:.2f} TNR={tnr:.2f}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(RUNS_DIR / "calibration.csv", rows)
    return 0


def run_eval(cfg, args) -> dict[str, Any]:
    items = read_jsonl(EVAL_SET)
    answerable = [i for i in items if i["type"] == "answerable"]
    unanswerable = [i for i in items if i["type"] != "answerable"]

    configs = [
        ("dense-only", "dense", False),
        ("bm25-only", "sparse", False),
        ("hybrid-rrf", "hybrid", False),
        ("hybrid+rerank", "hybrid", True),
    ]
    use_rerank = not getattr(args, "no_rerank", False)
    if not use_rerank:   # offline/CI run: skip configs that need the cross-encoder
        configs = [c for c in configs if not c[2]]
    rows: list[dict[str, Any]] = []
    retriever = load_index(cfg, need_reranker=use_rerank)

    for name, mode, rerank in configs:
        per_q = []
        for item in answerable:
            t0 = time.perf_counter()
            hits = retriever.retrieve(item["question"], mode=mode, rerank=rerank)
            dt = (time.perf_counter() - t0) * 1000
            m = score_retrieval(item, hits)
            m.update({"latency_ms": dt, "id": item["id"]})
            per_q.append(m)
        for key in ("recall@1", "recall@5", "recall@10", "precision@5", "mrr@10", "ndcg@5"):
            rows.append({"config": name, "stage": "retrieval", "metric": key,
                         "value": round(statistics.fmean(q[key] for q in per_q), 4),
                         "n": len(per_q)})
        rows.append({"config": name, "stage": "retrieval", "metric": "latency_ms_p50",
                     "value": round(statistics.median(q["latency_ms"] for q in per_q), 1), "n": len(per_q)})
        (RUNS_DIR).mkdir(parents=True, exist_ok=True)
        _write_csv(RUNS_DIR / f"retrieval_{name.replace('+', '_')}.csv", per_q)

    # answer-level eval: guarded vs unguarded, same retriever
    if not args.no_answer_eval:
        generator = None
        if args.llm and args.llm != "none":
            path = args.llm
            if path.startswith("google/flan-t5"):
                local = HERE / "models" / Path(path).name
                path = str(local) if local.exists() else path
            generator = FlanT5Generator(path, cfg["max_new_tokens"])
        rag = RAG(retriever, generator, cfg)
        page_text = {(p["doc"], p["page"]): p["text"] for p in load_pages(DOCS_DIR)}

        for label, verify, select in (
            ("hybrid+rerank+llm-no-guardrail", False, False),
            ("hybrid+rerank+llm-verified", True, False),
            ("hybrid+rerank+llm", True, True),
        ):
            if generator is None:
                break
            per_q = []
            for item in answerable + unanswerable:
                a = rag.ask(item["question"], mode="hybrid", rerank=True,
                            verify=verify, select=select)
                per_q.append(_score_answer(item, a, page_text))
            _summarise_answers(rows, label, per_q, len(answerable), len(unanswerable))
            (RUNS_DIR).mkdir(parents=True, exist_ok=True)
            _write_csv(RUNS_DIR / f"answers_{label.replace('+', '_')}.csv", per_q)

    _write_csv(EVAL_CSV, rows)
    return {"rows": rows, "path": str(EVAL_CSV)}


def _score_answer(item: dict[str, Any], a: Answer,
                  page_text: dict[tuple[str, int], str]) -> dict[str, Any]:
    row = {
        "id": item["id"],
        "type": item["type"],
        "status": a.status,
        "mode": a.mode,
        "answer": a.answer,
        "f1": 0.0,
        "contains_gold": 0.0,
        "fact_coverage": 0.0,
        "fully_answered": 0.0,
        "citation_hit": 0.0,
        "support_rate": 0.0,
        "unsupported": 0.0,
        "dropped": a.dropped_claims,
        "latency_ms": a.latency_ms.get("total", 0.0),
        "refused": float(a.status in {"refused", "blocked"}),
    }
    if item["type"] == "answerable":
        facts = item.get("gold_facts") or []
        found = [f for f in facts if _norm(f) in _norm(a.answer)]
        row["facts_found"], row["facts_total"] = len(found), len(facts)
        row["fact_coverage"] = round(len(found) / len(facts), 4) if facts else 0.0
        row["fully_answered"] = float(bool(facts) and len(found) == len(facts))
        row["f1"] = round(token_f1(a.answer, item["gold_answer"]), 4)
        row["contains_gold"] = float(_norm(item["gold_answer"]) in _norm(a.answer))
        cited = _cited_numbers(a.answer)
        row["citation_hit"] = float(
            any(c["doc"] == item["gold_doc"] and c["page"] in item["gold_pages"] and c["n"] in cited
                for c in a.citations)
        )

    # Independent faithfulness audit: does each sentence of the final answer
    # hold up against the page it cites? This is measured for every mode,
    # including the unguarded baseline, so the comparison is like-for-like.
    ctx = "\n".join(page_text.get((c["doc"], c["page"]), "") for c in a.citations)
    sentences = [s for s in _SENT_SPLIT.split(a.answer) if len(tokenize(s)) >= 3]
    if a.status in {"answered", "extractive"} and ctx and sentences:
        ok = [support_ratio(s, set(tokenize(ctx)), ctx) >= 0.6 for s in sentences]
        row["support_rate"] = round(sum(ok) / len(ok), 4)
        row["unsupported"] = float(not all(ok))
    return row


def _summarise_answers(rows: list[dict[str, Any]], label: str, per_q: list[dict[str, Any]],
                       n_ans: int, n_unans: int) -> None:
    ans = [q for q in per_q if q["type"] == "answerable"]
    un = [q for q in per_q if q["type"] != "answerable"]
    answered = [q for q in ans if q["status"] in {"answered", "extractive"}]
    audi = [q for q in per_q if q["support_rate"] > 0]
    metrics = {
        "fact_coverage": statistics.fmean(q["fact_coverage"] for q in ans),
        "fully_answered_rate": statistics.fmean(q["fully_answered"] for q in ans),
        "answer_f1": statistics.fmean(q["f1"] for q in ans),
        "citation_precision": statistics.fmean(q["citation_hit"] for q in ans),
        "answer_rate_answerable": len(answered) / max(1, len(ans)),
        "refusal_rate_unanswerable": statistics.fmean(q["refused"] for q in un) if un else 0.0,
        "false_answer_rate": 1 - (statistics.fmean(q["refused"] for q in un) if un else 0.0),
        "mean_sentence_support": statistics.fmean(q["support_rate"] for q in audi) if audi else 0.0,
        "hallucinated_answer_rate": statistics.fmean(q["unsupported"] for q in audi) if audi else 0.0,
        "llm_answer_share": sum(q["mode"] == "llm" for q in answered) / max(1, len(answered)),
        "latency_ms_p50": statistics.median(q["latency_ms"] for q in per_q),
    }
    for k, v in metrics.items():
        rows.append({"config": label, "stage": "answer", "metric": k, "value": round(float(v), 4),
                     "n": n_ans + n_unans})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def check_labels() -> int:
    """Fail loudly if a gold answer is not actually on its cited page."""
    pages = {(p["doc"], p["page"]): p["text"] for p in load_pages(DOCS_DIR)}
    bad = 0
    for item in read_jsonl(EVAL_SET):
        if item["type"] != "answerable":
            continue
        text = " ".join(pages.get((item["gold_doc"], pg), "") for pg in item["gold_pages"])
        gold_tokens = tokenize(item["gold_answer"])
        page_tokens = set(tokenize(text))
        missing = [k for k in gold_tokens if k not in page_tokens]
        if len(gold_tokens) < 2 or len(missing) > max(1, int(0.35 * len(gold_tokens))):
            bad += 1
            print(f"FAIL {item['id']}: gold_answer not supported on "
                  f"{item['gold_doc']} p.{item['gold_pages']} (missing {missing})")
    total = sum(1 for i in read_jsonl(EVAL_SET) if i["type"] == "answerable")
    print(f"checked {total} answerable labels, {bad} unsupported")
    return 1 if bad else 0


# --------------------------------------------------------------------------- #
# 10. CLI
# --------------------------------------------------------------------------- #

def _cfg_from_args(args) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if getattr(args, "llm", None):
        cfg["llm_model"] = args.llm
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Minimal guarded RAG prototype")
    p.add_argument("command", choices=["download-models", "ingest", "ask", "eval", "show-index", "calibrate"])
    p.add_argument("question", nargs="*", help="question for the ask command")
    p.add_argument("--mode", default="hybrid", choices=["dense", "sparse", "hybrid"])
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-verify", action="store_true", help="disable citation verification (eval only)")
    p.add_argument("--llm", default=DEFAULTS["llm_model"],
                   help="'none' for retrieval-only, or a HF id / local model directory")
    p.add_argument("--embedder", default="st", choices=["st", "hashing"])
    p.add_argument("--no-answer-eval", action="store_true")
    p.add_argument("--check-labels", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cfg = _cfg_from_args(args)
    t0 = time.perf_counter()

    if args.command == "download-models":
        return download_models(cfg, args.llm)

    if args.command == "ingest":
        manifest = build_index(cfg, args.embedder)
        print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "show-index":
        m = INDEX_DIR / "manifest.json"
        print(m.read_text(encoding="utf-8") if m.exists() else "no index; run ingest")
        return 0

    if args.command == "calibrate":
        return calibrate(cfg)

    if args.command == "eval":
        if args.check_labels:
            return check_labels()
        result = run_eval(cfg, args)
        print(f"wrote {result['path']} in {time.perf_counter() - t0:.1f}s")
        for r in result["rows"]:
            if r["stage"] == "retrieval" and r["metric"] in {"recall@5", "mrr@10", "ndcg@5"}:
                print(f"  {r['config']:22s} {r['metric']:10s} {r['value']}")
            if r["stage"] == "answer" and r["metric"] in {
                "fact_coverage", "fully_answered_rate", "citation_precision",
                "hallucinated_answer_rate", "refusal_rate_unanswerable"}:
                print(f"  {r['config']:32s} {r['metric']:28s} {r['value']}")
        return 0

    question = " ".join(args.question).strip() or "What causes vibration and noise from the cyclone fan?"
    try:
        question = clean_question(question)
    except ValueError as exc:
        p.error(str(exc))          # argparse usage error: exit status 2

    rag = make_rag(cfg, args.llm if args.llm != "none" else None, need_reranker=not args.no_rerank)
    a = rag.ask(question, mode=args.mode, rerank=not args.no_rerank, verify=not args.no_verify)
    print(a.to_json() if args.json else format_answer(a))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
