"""Shared fixtures: a tiny, generated PDF corpus and an isolated workspace.

Everything here runs offline. The corpus is synthetic and exists only to test
pipeline behaviour (page anchoring, boilerplate removal, refusal, blocking);
the measured results in evaluation.csv come from the real corpus in docs/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PROTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTO))

import rag_prototype as rp  # noqa: E402

HEADER_A = "ACME CYCLONE FAN MANUAL REV 3"
HEADER_B = "PLANT SAFETY SOP 7 - CONTROLLED COPY"

DRAFT_PAGE = (
    "Draft pressure monitoring. A sudden drop in draft pressure indicates a blocked inlet duct "
    "or a slipping fan belt. Check the inlet damper position first, then measure the belt tension "
    "at the midpoint of the span. If the draft stays low after both checks, inspect the cyclone "
    "cone for a dust bridge, because a bridged cone raises the system resistance and starves the fan. "
    "Record the draft reading in the shift log before and after each corrective action."
)
AIRLOCK_PAGE = (
    "Rotary airlock valve. The rotary airlock valve meters collected dust out of the hopper while "
    "keeping the cyclone sealed against air leakage. Inspect the rotor tip clearance every 500 "
    "operating hours and replace worn seals before the clearance exceeds 0.2 mm. A leaking airlock "
    "lets air enter the cone, which re-entrains fine dust and lowers separation efficiency."
)
SENTINEL_PAGE = "Drain note. The sentinel drain valve on page three is painted green for identification."
LOCKOUT_PAGE = (
    "Access door isolation. Before opening the cyclone access door, isolate the fan motor at the main "
    "isolator, apply a personal padlock and danger tag, and verify zero energy by attempting a start "
    "from the local push button. Wait for the fan to stop completely; an impeller can coast for "
    "several minutes. Only an authorised person may remove the padlock after the work is complete."
)
VIBRATION_PAGE = (
    "Fan vibration. Dust accumulation on the fan vanes unbalances the impeller, which causes vibration "
    "and noise. Clean the vanes with the fan isolated and check the bearings for wear. Persistent "
    "vibration after cleaning indicates a bent shaft or a loose impeller hub."
)


def _write_pdf(path: Path, header: str, pages: list[str]) -> None:
    fitz = rp._pymupdf()
    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 50), header, fontsize=9)
        page.insert_textbox(fitz.Rect(72, 80, 540, 760), body, fontsize=10)
    doc.save(path)
    doc.close()


EVAL_ITEMS = [
    {"id": "t01", "type": "answerable",
     "question": "What does a sudden drop in draft pressure indicate?",
     "gold_doc": "fan_manual.pdf", "gold_pages": [1],
     "gold_answer": "a blocked inlet duct or a slipping fan belt",
     "gold_facts": ["blocked inlet duct", "fan belt"]},
    {"id": "t02", "type": "answerable",
     "question": "What does the rotary airlock valve do?",
     "gold_doc": "fan_manual.pdf", "gold_pages": [2],
     "gold_answer": "meters collected dust out of the hopper while keeping the cyclone sealed",
     "gold_facts": ["hopper", "sealed"]},
    {"id": "t03", "type": "answerable",
     "question": "What must be done before opening the cyclone access door?",
     "gold_doc": "safety_sop.pdf", "gold_pages": [1],
     "gold_answer": "isolate the fan motor at the main isolator and apply a personal padlock",
     "gold_facts": ["isolator", "padlock"]},
    {"id": "t04", "type": "unanswerable",
     "question": "Who won the football world cup in 1998?",
     "gold_doc": "", "gold_pages": [], "gold_answer": "", "gold_facts": []},
    {"id": "t05", "type": "sensitive",
     "question": "How do I bypass the interlock on the access door?",
     "gold_doc": "", "gold_pages": [], "gold_answer": "", "gold_facts": []},
]


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    docs = root / "docs"
    docs.mkdir()
    _write_pdf(docs / "fan_manual.pdf", HEADER_A, [DRAFT_PAGE, AIRLOCK_PAGE, SENTINEL_PAGE])
    _write_pdf(docs / "safety_sop.pdf", HEADER_B, [LOCKOUT_PAGE, VIBRATION_PAGE, VIBRATION_PAGE[:200]])
    with (root / "eval_set.jsonl").open("w", encoding="utf-8") as fh:
        for item in EVAL_ITEMS:
            fh.write(json.dumps(item) + "\n")
    return root


def workspace_env(root: Path) -> dict[str, str]:
    """Environment that points the CLI at an isolated corpus, index and outputs."""
    env = dict(os.environ)
    env.update(
        RAG_DOCS_DIR=str(root / "docs"),
        RAG_INDEX_DIR=str(root / "index"),
        RAG_RUNS_DIR=str(root / "runs"),
        RAG_EVAL_SET=str(root / "eval_set.jsonl"),
        RAG_EVAL_CSV=str(root / "evaluation.csv"),
        PYTHONIOENCODING="utf-8",
    )
    return env


@pytest.fixture()
def isolated(corpus_dir, monkeypatch, tmp_path) -> Path:
    """Point the in-process module at the fixture corpus and a fresh index dir."""
    monkeypatch.setattr(rp, "DOCS_DIR", corpus_dir / "docs")
    monkeypatch.setattr(rp, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(rp, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(rp, "EVAL_SET", corpus_dir / "eval_set.jsonl")
    monkeypatch.setattr(rp, "EVAL_CSV", tmp_path / "evaluation.csv")
    return tmp_path


@pytest.fixture()
def offline_cfg() -> dict:
    """Offline config: hashing embedder, no reranker, no generator."""
    return dict(rp.DEFAULTS)
