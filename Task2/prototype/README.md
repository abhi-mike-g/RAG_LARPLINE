# Minimal guarded RAG prototype

A runnable, dependency-light RAG system over the 11-document cyclone corpus in
`docs/`: hybrid retrieval (FAISS + BM25), cross-encoder reranking, a small open
LLM, and guardrails that can refuse. No API keys, no paid services, CPU only.

```
docs/*.pdf -> text -> chunks -> [dense index | BM25] -> RRF -> rerank -> LLM
                                                              -> verify -> cited answer / refusal
```

## What it does

- **Ingests** PDFs page by page, strips repeated headers and footers, and builds
  900-character chunks with 180 characters of overlap, anchored to page numbers.
- **Indexes** each chunk twice: a 384-d dense vector (`all-MiniLM-L6-v2`) in
  FAISS, and a BM25 term index.
- **Retrieves** with both, fuses with Reciprocal Rank Fusion, and reranks the
  top candidates with a cross-encoder.
- **Answers** with `flan-t5-base` — or refuses before calling it when the
  retrieval score says the corpus does not cover the question.
- **Verifies** every sentence of the draft against the cited passages. Unsupported
  sentences are dropped; if nothing survives, the answer becomes a verbatim
  extract or an explicit refusal.
- **Measures** itself: `evaluation.csv` holds recall@k, precision@k, MRR, nDCG@5,
  citation precision, refusal rate and latency for four retrieval configurations
  and a guarded-vs-unguarded answer comparison.

## Requirements

- Python 3.10+ (developed on 3.14, Windows, CPU only)
- ~2 GB free disk for the two models
- No GPU, no network at query time (after the models are downloaded)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Models to download (free, open weights)

| Purpose | Model | Size | Licence |
|---|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 90 MB | Apache-2.0 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 90 MB | Apache-2.0 |
| Generation | `google/flan-t5-base` | 990 MB | Apache-2.0 |

One command fetches all three (about 1.2 GB) so later runs need no network:

```bash
python rag_prototype.py download-models
```

The embedder and reranker go to the Hugging Face cache; the generator goes to
`./models/flan-t5-base` with only the files transformers loads
(`model.safetensors`, tokenizer, configs). Do not use a bare
`snapshot_download('google/flan-t5-base')`: that repo also holds Flax,
TensorFlow and PyTorch `.bin` copies of the weights and pulls about 3 GB extra.
`--llm` with another Hub id downloads that model instead.

> On Windows without symlink privileges, the default Hugging Face cache fails
> with `WinError 1314` on some `huggingface_hub` versions; the generator is
> unaffected (it uses `local_dir`). Enable Developer Mode if the embedder fails.

Other models work by pointing `--llm` at any seq2seq checkpoint directory or
Hub id: `--llm google/flan-t5-small` (fastest), `--llm google/flan-t5-large`
(better prose, slower on CPU), `--llm ./models/mt5-small`.

## Run

```bash
# 1. build the index (~30 s on one CPU core)
python rag_prototype.py ingest

# 2. ask a question (eval item q10; answers from Cyclone_Emission_Control.pdf p.13)
python rag_prototype.py ask "Why does dust accumulation on fan vanes cause vibration and noise?"

# 3. no generator: extractive answer only (eval item q04). The embedder and
#    reranker still load; see "Fully offline" below for a zero-download run.
python rag_prototype.py ask "What is the purpose of the rotary airlock valve on the cyclone?" --llm none

# 4. full evaluation: ablations + guardrail comparison (3–5 min on CPU with flan-t5-base)
python rag_prototype.py eval

# 5. verify the gold labels really sit on the pages they cite
python rag_prototype.py eval --check-labels
```

`ask` prints the answer, the sources it used (document, page, section) and the
run record: status, mode, latency, support score, dropped-claim count and the
config hash. Add `--json` for machine-readable output.

### Useful flags

| Flag | Effect |
|---|---|
| `--mode dense\|sparse\|hybrid` | retrieval method, default `hybrid` |
| `--no-rerank` | skip the cross-encoder |
| `--no-verify` | ship the raw LLM draft (eval baseline only — never production) |
| `--llm none` | no generator, extractive answers |
| `--embedder hashing` | (`ingest` only) offline hashing embedder, no model download; `ask` picks it up from the manifest |
| `--json` | JSON output for the answer record |

Without `--no-rerank` the relevance gate uses the calibrated rerank score
(2.84); every guardrail number in `evaluation.csv` is measured in that mode.
With `--no-rerank` the gate falls back to the best dense cosine among the hits
(0.28) instead of switching off. That refuses clearly off-topic questions, but
calibration shows dense scores cannot separate near-domain out-of-scope
questions (`notes.md` §3), so `--no-rerank` is for debugging, not for answers.

### Fully offline (no model downloads at all)

```bash
python rag_prototype.py ingest --embedder hashing
python rag_prototype.py ask "What is the purpose of the rotary airlock valve on the cyclone?" --llm none --no-rerank
```

Every step of the pipeline runs (ingest, retrieval, gate, extractive answer
with a page citation, refusal, blocking), but this is a smoke-test fallback,
not the measured configuration. Checked on the real corpus: q04 is answered
from the right page, q10 is wrongly refused, and the out-of-scope price
question u01 is wrongly answered with warranty text. Use the default models
for real answers. The offline path is what CI runs.

### Environment overrides

| Variable | Default | Purpose |
|---|---|---|
| `RAG_DOCS_DIR` | `./docs` | corpus to ingest |
| `RAG_INDEX_DIR` | `./index` | where the index is written and read |
| `RAG_RUNS_DIR` | `./runs` | per-question eval outputs |
| `RAG_EVAL_SET` | `./eval_set.jsonl` | labelled questions |
| `RAG_EVAL_CSV` | `./evaluation.csv` | metric table written by `eval` |

## Outputs

| File | Contents |
|---|---|
| `index/manifest.json` | model ids, chunk parameters, corpus hash, package versions |
| `evaluation.csv` | metric table: 4 retrieval configs + guarded/unguarded answers |
| `runs/retrieval_*.csv` | per-question retrieval scores |
| `runs/answers_*.csv` | per-question answers, statuses and support flags |

## Guardrails in this prototype

1. **Query screen** — credentials, interlock/guard bypass, prompt injection,
   personal data and regulated advice are blocked before retrieval.
2. **Relevance gate** — cosine < 0.28 or rerank < 2.84 refuses without an LLM
   call. The rerank threshold is calibrated by `python rag_prototype.py calibrate`
   on the labelled set, not chosen by hand.
3. **Citation by verification** — a sentence is kept only if its content tokens
   overlap the supporting passage and every number it states appears there
   verbatim. The LLM never writes citation markers.
4. **Fallback ladder** — verified answer, else a verbatim extract with its page,
   else an explicit refusal naming the closest indexed topics.
5. **Untrusted context** — instruction-like lines inside retrieved passages are
   stripped before prompting.

## Known limits

- Generation is the weak link, not retrieval. With `flan-t5-base`, 27 of 30
  answerable questions get an answer, but only 1 of those 27 comes from the LLM:
  the guardrail rejects the draft and the verbatim extract ships. That is the
  guardrail working, not a hidden failure — `runs/answers_*.csv` records the
  mode and the dropped-claim count per question. A 7B instruct model is the
  next lever; the eval measures whatever `--llm` points at.
- Three answerable questions (q13, q26, q28) are refused by the relevance
  gate, one of them (q28) with its correct page ranked first. Losing them is the
  price of the zero false-answer rate: the highest-scoring out-of-scope question
  (u01, 2.837) sits above all three, so any cut low enough to answer them also
  answers u01.
- Document titles are curated in `DOC_TITLES`; for unknown files the first-page
  heuristic can pick a URL or an address, so the source list always shows the
  filename as well.
- 12 pages of the corpus (of 195) are scanned images with no text layer and are
  skipped; add OCR (Tesseract) before production.
- The corpus mixes three meanings of "cyclone" (dust separator, maritime storm,
  mineral classifier). The eval includes all three on purpose; a production
  deployment should filter by equipment or site at retrieval time.
- Page-level recall is the retrieval metric here. Chunk-level and answer-level
  scoring are in `runs/`.

## Run as an HTTP service

The CLI loads the models for every question. `service.py` loads them once and
serves many questions; it reuses the same pipeline code.

```bash
pip install -r requirements.txt -r requirements-dev.txt
python rag_prototype.py ingest
uvicorn service:app --host 127.0.0.1 --port 8000

curl -s localhost:8000/readyz
curl -s localhost:8000/v1/ask -H "content-type: application/json" \
     -d '{"question": "What is the purpose of the rotary airlock valve on the cyclone?"}'
```

| Endpoint | Behaviour |
|---|---|
| `GET /healthz` | 200 while the process is alive |
| `GET /readyz` | 200 once the index and models are loaded, 503 before or if loading failed |
| `POST /v1/ask` | `{"question": str (3–1000 chars)}`; unknown fields are rejected (422), including `mode` and `verify`, so clients cannot switch off the reranker gate or citation verification. Always hybrid + rerank. Returns the answer record from `ask --json` plus `request_id` |

| Setting | Default | Effect |
|---|---|---|
| `RAG_LLM` | `google/flan-t5-base` | `none` for extractive-only |
| `RAG_RERANK` | `1` | `0` disables the cross-encoder (not recommended, see above) |
| `RAG_MAX_INFLIGHT` | `1` | answers computed at once per process; scale with processes, not threads |
| `RAG_QUEUE_WAIT_S` | `2` | wait for a free slot, then 503 + `Retry-After: 1` |
| `RAG_REQUEST_TIMEOUT_S` | `60` | then 504; the abandoned computation keeps its slot until it ends, so a stuck model sheds load instead of stacking threads |
| `RAG_API_KEY` | unset | if set, `/v1/ask` requires header `x-api-key` (constant-time compare) |

Each request logs one JSON line (request id, status, mode, citation count,
latency, config hash). The question text is not logged.

Measured with `python tools/load_test.py --users 100 --requests 200` on a
laptop CPU: one process serves about 0.55 answers/s and sheds the rest with
503 within seconds (details in `notes.md` §7). Scale by running more
processes behind a load balancer, one model copy each.

## Tests

```bash
pip install -r requirements-dev.txt     # no torch, no model downloads
python -m pytest tests -q
```

The suite builds a small synthetic PDF corpus at test time and covers:

- **End to end through the CLI**, one process per step as a user runs it:
  `ingest` → cited `ask` → out-of-scope refusal → sensitive-query block →
  oversized-input rejection → `eval --check-labels` → `eval` writing metrics.
- **Pipeline units**: header stripping, chunks never carrying text from another
  page, BM25 ranking, hashing-embedder stability across processes, query-screen
  rules, passage sanitising, number-level claim verification, input bounds,
  no shared-state mutation during `ask`.
- **HTTP service**: validation (422), API key (401), readiness (503 until
  loaded), load shedding (503 + `Retry-After`), timeout (504).

CI (`.github/workflows/ci.yml`) is configured to run the suite on Python 3.10,
3.11 and 3.12, plus the README path with the real models; its first run happens
on push. Verified locally on Windows with Python 3.14 and 3.11.

## Reproduce the deck

```bash
python tools/make_deck.py     # writes ../architecture_diagram.pptx from evaluation.csv
```
