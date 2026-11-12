# Task 2 — Retrieval-augmented answers over technical documentation

**Deliverables in this folder**

| Path | What it is |
|---|---|
| `architecture_diagram.pptx` | 5-slide architecture and evidence deck, generated from `prototype/evaluation.csv` |
| `notes.md` | This document: design, trade-offs, measured results, scaling |
| `prototype/rag_prototype.py` | The runnable system: download-models, ingest, ask, eval, calibrate, label check |
| `prototype/service.py` | The same pipeline as an HTTP service (models loaded once, validation, load shedding) |
| `prototype/tests/` | 50 offline tests: CLI end to end, pipeline units, HTTP contract |
| `prototype/tools/load_test.py` | Concurrency measurement used in §7 |
| `prototype/README.md` | Setup and run instructions |
| `prototype/eval_set.jsonl` | 37 labelled questions (30 answerable, 4 unanswerable, 3 sensitive) |
| `prototype/evaluation.csv` | Every metric quoted below, per configuration |
| `prototype/runs/*.csv` | Per-question retrieval scores, answers, support flags, calibration |
| `prototype/docs/*.pdf` | The 11-document corpus |

**Headline result.** On 30 answerable questions scored at page level, hybrid
retrieval with cross-encoder reranking reaches recall@5 = 1.00 and MRR@10 = 0.96,
against 0.77 / 0.61 for dense-only retrieval. The guarded answer path answers 27
of 30 questions, cites a correct page on 23 of 30, and produces zero answers with
an unsupported sentence. All seven out-of-scope and sensitive questions are
refused. Cost per query is $0 and 0.9 s median on CPU (1.3 s in an earlier
run; latency varies run to run, the accuracy metrics did not).

---

## 1. Problem and corpus

The task is a question-answering system over a document collection, where a
wrong answer is worse than no answer and every answer must be traceable to a
page. The corpus is a public Google Drive folder of 11 industrial PDFs
(195 pages, 8 MB): cyclone dust-collector manuals, installation and O&M
instructions, a B.Tech design report, US EPA-style technical guidance, a
mineral-processing tutorial, and a Directorate General of Shipping cyclone
disaster-management SOP.

Three properties of this corpus drove the design:

1. **One word, three meanings.** "Cyclone" is an industrial dust separator in
   most documents, a maritime storm in the DG Shipping SOP, and a
   mineral-processing classifier in the Weir tutorial. A retriever that only
   matches the term will mix them; the system needs a reranker and a refusal
   path, not just a vector index.
2. **Numbers, units and identifiers carry the answers.** "24 inches", "psig",
   "zone 22", "316-744-7151", "SITREP". These are exactly the tokens dense
   embeddings blur, which is why lexical retrieval is competitive here.
3. **Scanned pages.** 12 of 195 pages have no usable text layer. They are
   skipped and counted, not silently ignored (see §5.4).

## 2. Architecture

```
PDF ─► page text ─► boilerplate strip ─► page-anchored chunks
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
              dense index (MiniLM + FAISS)                 sparse index (BM25)
                        └──────────────► RRF ◄──────────────────────┘
                                          │
                                   cross-encoder rerank
                                          │
                              relevance gate  ── refuse (no LLM call)
                                          │ pass
                                    flan-t5-base draft
                                          │
                              sentence-level verification
                                          │
                    verified draft  vs  verbatim extract  ── pick better
                                          │
                                cited answer / explicit refusal
```

The deck (`architecture_diagram.pptx`) shows the same pipeline as five stages
with the parameters and the measured numbers attached to each stage.

**Stage 1 — ingestion.** PyMuPDF extracts text page by page. Text is NFKC
normalised, control and replacement glyphs are dropped, hyphenated line breaks
are rejoined, and lines repeated on 40% or more of a document's pages are
removed as headers and footers. Pages are chunked at 900 characters with 180
characters of sentence-aligned overlap, split on headings first, and never
across a page boundary — so every chunk has an exact page number. Result: 430
chunks, mean 701 characters. (An earlier version glued a short page tail onto
the previous page's chunk, so 30 tails were cited under the wrong page; fixing
that is what moved recall@5 from 0.967 to 1.000, see §9.) The 12 skipped
scanned pages leave 183 pages with text. Document titles are curated in `DOC_TITLES`,
because first-line extraction picks a URL or an address on 5 of these 11 files.

**Stage 2 — index.** Each chunk is embedded with `all-MiniLM-L6-v2` (384-d,
L2-normalised, FAISS flat inner product, exact search) and separately indexed
by BM25 (k1 = 1.5, b = 0.75, light stemming, stopword filter). `index/manifest.json`
records model ids, chunk parameters, corpus SHA-256 and library versions, so an
index can be reproduced or invalidated deterministically.

**Stage 3 — retrieval.** Dense top-20 and BM25 top-20 are fused with Reciprocal
Rank Fusion (k = 60), which needs no score calibration between two
incomparable scales. The fused top-20 is reranked by
`ms-marco-MiniLM-L-6-v2` cross-encoder, which scores query and passage jointly.
A gate on the top rerank score refuses before the LLM is called.

**Stage 4 — generation.** The top three passages go to `flan-t5-base` with a
prompt that instructs it to copy numbers verbatim, to answer in at most three
sentences, and to reply `NOT_ANSWERABLE` if the passages do not contain the
answer. The model is explicitly **not** asked to write citation markers.

**Stage 5 — attribution and guardrails.** See §4.

## 3. Retrieval strategy, and why measurement chose it

All retrieval numbers are page-level over the 30 answerable questions: a hit
means a chunk from the exact gold document *and* page is in the top *k*.

| Configuration | recall@1 | recall@5 | precision@5 | MRR@10 | nDCG@5 | p50 ms |
|---|---|---|---|---|---|---|
| Dense only (MiniLM) | 0.500 | 0.767 | 0.240 | 0.609 | 0.640 | 6 |
| BM25 only | 0.800 | 0.933 | 0.267 | 0.861 | 0.868 | 0.3 |
| Hybrid RRF | 0.767 | 0.900 | 0.267 | 0.823 | 0.833 | 0.3 |
| **Hybrid + rerank** | **0.933** | **1.000** | **0.300** | **0.961** | **0.965** | **293** |

Readings, in order of importance:

- **Lexical retrieval beats dense retrieval on this corpus** (0.933 vs 0.767
  recall@5). Technical questions are full of exact tokens that embeddings
  smooth away. This is a corpus property, not a claim that BM25 is generally
  better; it is why the design is hybrid rather than pure-vector.
- **Fusion recovers recall but not order.** Hybrid RRF does not beat BM25
  here: its recall@5 (0.900) and MRR (0.823) are both *below* BM25's (0.933,
  0.861), because fusing in a weaker dense list pulls some correct pages down.
  Recall and ranking order are separate problems.
- **Reranking fixes order.** The cross-encoder lifts recall@1 from 0.767 to
  0.933 and MRR from 0.823 to 0.961, and it is the only configuration at 1.00
  recall@5. It costs 0.3–0.56 s of CPU for 20 pairs across runs
  (15–28 ms per pair), nearly all of retrieval latency and the dominant cost of
  the whole pipeline on CPU. It is still the best accuracy-per-second change
  available here, because it is what makes the answer path safe to run at all.
- **It is not perfect, and the misses are honest.** The gold page is ranked
  first for 28 of 30 questions and in the top five for all 30. The previous
  single recall@5 miss (q29, a guaranteed dust removal rate) was caused by the
  page-boundary chunking defect, not by the reranker: the guarantee sat in a
  short page tail filed under the previous page.
- **Precision@5 is not comparable to 1.0.** Gold relevance is a page, and a
  page can contribute several chunks, so 0.300 means roughly 1.5 of the top 5
  chunks come from a gold page — consistent with one gold page per question.

The gate threshold is **calibrated, not chosen**. `python rag_prototype.py
calibrate` scans the labelled set and reports the separation between answerable
and out-of-scope questions for both scores:

```
dense   answerable p05 = 0.000  min = 0.000 | unanswerable max = 0.668 | cut = 0.55  TPR 0.73 TNR 0.75
rerank  answerable p05 = 2.621  min = 2.180 | unanswerable max = 2.837 | cut = 2.84  TPR 0.90 TNR 1.00
```

The dense score does not separate the classes at all (the best out-of-scope
question scores 0.668, above the fifth percentile of answerable questions), so
the dense cut of 0.28 is kept only as a cheap pre-filter; the calibrated dense
cut of 0.55 would refuse 27% of answerable questions for 25% of out-of-scope
ones. The real decision is the rerank cut of 2.84, just above the highest-scoring
out-of-scope question (2.837), which gives TNR = 1.00 at a cost of three
answerable questions (2.826, 2.621, 2.180) that fall below the line.
That trade is deliberate: on a maintenance-manual assistant, a false answer is
more expensive than a refusal. It is also the design's main known weakness — a
question whose evidence is retrieved at rank 1 can still be refused if the
cross-encoder scores the pair below 2.84, as q28 demonstrates in §5.4.

## 4. Attribution and guardrails

The design rule is that **the model never writes citations**. A 250M-parameter
seq2seq model cannot be trusted to attribute its own output, and citation
markers it emits are unfalsifiable decoration. Instead, attribution is computed:

1. **Sentence support.** Each sentence of the draft is tokenised; content tokens
   (stopwords removed, numbers kept) are compared with the candidate passages.
   A sentence is kept only if its support ratio against one passage is at least
   0.60 **and** every number it states appears verbatim in that passage.
2. **Drop, then fall back.** Unsupported sentences are dropped. If nothing
   survives, the system produces a verbatim extractive answer with its page, and
   if no extract is possible, an explicit refusal.
3. **Two candidates, one decision.** When both a verified LLM draft and an
   extractive answer exist, both are scored by the cross-encoder against the
   question and the better one ships. The LLM is not the default winner, and on
   this corpus it usually is not: 1 of 27 answers is LLM-generated.
4. **Relevance gate** (§3) refuses before generation when the corpus does not
   cover the question.
5. **Query screen** runs before retrieval and blocks credential requests,
   interlock or guard bypass, prompt injection, personal data and regulated
   advice, with a specific safe message per class.
6. **Untrusted context.** Instruction-like lines inside retrieved passages are
   stripped before they reach the prompt.

The guardrail ladder is measured by running the same 30 questions through three
configurations of the same pipeline:

| Configuration | fact coverage | hallucinated-answer rate | citation precision |
|---|---|---|---|
| Raw LLM draft, no guardrail | 0.033 | 0.125 | 0.000 |
| + sentence verification | 0.483 | 0.037 | 0.667 |
| + candidate selection (**shipped**) | **0.683** | **0.000** | **0.767** |

- **The raw baseline is worse than its fluency suggests.** `flan-t5-base`
  answers 22 of 30 questions, but 14 of those are degenerate fragments ("air",
  "The following instructions are for the following bearings:") with no
  scorable sentence at all. Of the 8 that are scorable, 7 contain none of the
  gold facts: across all 30 questions exactly 1 of 46 gold facts appears.
  Fact coverage is 3.3%, F1 0.06. A demo that stops at "the model answered"
  would report a working system.
- **Verification is what converts drafts into evidence.** Keeping only supported
  sentences raises fact coverage to 0.48, citation precision from 0 to 0.67, and
  drops the hallucinated-answer rate to 0.04.
- **Selection closes the gap.** Letting the verbatim extract compete lifts fact
  coverage to 0.68 and removes the last hallucinated answer. The system's
  usefulness on this corpus comes mainly from disciplined extraction, not from
  generation.

**Refusal behaviour.** All 7 non-answerable items are refused (4 unanswerable,
3 sensitive): refusal rate 1.00, false-answer rate 0.00. The assignment's own
example question — "What does a sudden draft drop indicate?" — is one of them:
the corpus contains boiler-draft vocabulary but no content that answers it, so
the correct behaviour is to refuse and say so.

## 5. Evaluation

### 5.1 Set design

37 items: 30 answerable, 4 unanswerable, 3 sensitive. Answerable items cover all
11 documents and all three senses of "cyclone". Each carries a gold document,
gold page(s), a gold answer and 1–3 gold facts (verbatim strings that must
appear in a correct answer). Unanswerable items are plausible but absent
(purchase price, plant-specific count, DCS alarm reset, draft-drop diagnosis).
Sensitive items test the policy screen rather than the corpus.

### 5.2 Label integrity

Gold labels are only useful if they are true. `python rag_prototype.py eval
--check-labels` re-reads the PDFs and fails if a gold answer is not on the page
it claims. It found one real defect during development: an item labelled
unanswerable ("What phone number does Kice give?") was in fact answered by
`Cyclone_Manual.pdf` p.5, and was reclassified as an answerable question. A
draft question about the warranty was dropped for the same reason the checker
exists: two documents state a two-year warranty, so no single gold page could be
correct, and a label that cannot be verified should not be scored.

### 5.3 Metric definitions

| Metric | Definition |
|---|---|
| recall@k | 1 if any of the top-k chunks comes from the gold document and page |
| precision@k | share of top-k chunks from gold pages (page-level, multiple chunks per page) |
| MRR@10 | reciprocal rank of the first gold chunk |
| nDCG@5 | binary-relevance nDCG at 5 |
| fact coverage | share of gold facts present in the answer, averaged over the 30 answerable questions |
| fully answered | share of questions with every gold fact present |
| citation precision | share of questions where the answer cites the gold page and carries its marker |
| support rate | share of answer sentences whose cited page supports ≥0.60 of their content tokens and all their numbers |
| hallucinated-answer rate | share of **scorable** answers (≥1 sentence of ≥3 tokens) containing an unsupported sentence |
| refusal rate (non-answerable) | share of the 4 unanswerable and 3 sensitive items that are refused or blocked |

### 5.4 Threats to validity

These are the honest limits of the numbers above.

- **Small set, single annotator.** 30 answerable questions is enough to separate
  configurations that differ by 15 points of recall@5, not enough for
  two-point differences. All labels were written and checked by one person.
- **The gate costs three correct answers.** q13 (2.826), q26 (2.621) and q28
  (2.180) are answerable, and all three are refused. In the worst case — q28 —
  the correct page is ranked first and the pair is still scored below the cut,
  so this is a false negative of the gate, not of retrieval. Lowering the cut to
  2.6 answers q13 and q26 but also answers the out-of-scope question u01
  (2.837), so the trade is two-for-one unless the reranker improves.
- **Page-level relevance.** Chunk-level relevance would be stricter; page-level
  matches what a user checks.
- **The hallucination metric has a selection effect.** It is computed over
  answers with at least one scorable sentence, because a two-word fragment
  cannot be checked for faithfulness. In the raw baseline only 8 of 22 answers
  were scorable, so its 0.125 understates the risk: judged over all 22 answered
  questions, 68% were unsupported or unverifiable. In the shipped configuration
  all 27 answers are scorable, so the 0.000 is not an artefact of the filter.
- **The support check is lexical.** It detects unsupported claims, not
  semantically inverted ones ("increase" for "decrease") when the numbers match.
  It is a cheap, deterministic proxy for entailment, not a substitute for one.
- **12 of 195 pages have no text layer** (1 in `Cyclone_Manual`, 1 in `Parker`,
  3 in `654b45bb…`, 3 in `D13`, 4 in `Operating-and-maintenance…`). Content on
  those pages is unreachable. OCR is the fix; it is not in this prototype.
- **Corpus ambiguity is real, not synthetic.** Most documents use "cyclone" for
  a dust separator, the SOP uses it for a storm, the Weir tutorial for a mineral
  classifier. The eval keeps all three, and reranking plus the gate is what
  keeps them apart. A production deployment would filter by equipment or site
  before retrieval.

## 6. Trade-offs

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Vector store | FAISS flat, in-process | Managed vector DBs (Pinecone, Weaviate): operational surface and cost with no gain at 430 chunks; flat exact search is <5 ms |
| Embeddings | `all-MiniLM-L6-v2` (384-d) | Larger embeddings (`bge-base`, `e5-large`): better recall on prose, 3–10× CPU cost, and dense retrieval is not the binding constraint here |
| Fusion | Reciprocal Rank Fusion | Weighted score blending: needs score calibration between cosine and BM25, which is unstable across corpora; RRF is rank-only |
| Reranking | `ms-marco-MiniLM-L-6-v2` cross-encoder | No rerank (0.90 recall@5, MRR 0.82); LLM-based reranking (10–20× cost) |
| Generation | `flan-t5-base`, local, CPU | Hosted LLMs (better prose, per-query cost and a data-egress decision); 7B local models (better, needs a GPU not available here) |
| Chunking | 900 chars, page-anchored | Token-based chunking: cleaner sizes, but breaks page attribution and needs a tokenizer per model |
| Citations | Computed by verification | Model-written markers: unfalsifiable, and measured at 0.00 precision in the raw baseline |
| Framework | Plain Python, ~1.3k lines | LangChain / LlamaIndex: faster to start, but hides the retrieval and guardrail logic that this task is about, and adds a dependency surface that changes between versions |
| Answer policy | Refuse when the gate fails | Always answer with the best passage: would have produced false answers on all 4 unanswerable and 3 sensitive items |

The single biggest caveat: **the LLM contributes little here.** Retrieval and
extraction do the work; the generator's output survives verification in 1 of 27
cases. That is an honest finding about small seq2seq models on dense technical
prose, not a hidden failure, and it is why the ladder was built rather than
assumed.

## 7. Scaling

What breaks, in order, as load grows:

| Dimension | Today (11 docs, 1 user) | 10× corpus (~110 docs, 4k pages) | 100 concurrent users |
|---|---|---|---|
| Ingestion | 30 s, one core | ~8 min, parallel by document; upsert by corpus hash instead of rebuild | Offline job, no user impact |
| Index | 430 × 384 floats, flat | ~40k chunks, still flat: <100 MB RAM, <5 ms search; IVF-PQ past ~1M chunks | One index per worker process (mmap), not per request |
| Retrieval | 0.3–0.56 s, almost all rerank | Search stays flat; rerank is O(candidates × pair cost) — cap candidates at 20–30 | 4–8 autoscaled workers; rerank is the throughput ceiling |
| Generation | ~0.5–0.8 s on CPU (0.9–1.3 s end to end in the eval) | Same, or a shared GPU; batch identical questions | Queue with a token budget; cache by (question, index hash) |
| Latency target | 0.9–1.3 s median | 1.3 s if rerank stays capped | Degrade by shortening candidate lists before degrading accuracy |
| Cost | $0 | One small CPU or GPU node; storage dominated by PDFs | Serverless CPU workers + one GPU pool, autoscaled to zero |

Three controls matter more than raw capacity: the relevance gate already keeps
the LLM off the critical path for out-of-scope queries; answer caching by
question and index hash removes repeated generation; and separating retrieval
from generation lets each scale on its own bottleneck — retrieval horizontally,
generation vertically.

**Measured, not assumed: one process.** `prototype/service.py` serves the same
pipeline over HTTP with the models loaded once, one answer in flight per
process, and load shedding (503 + `Retry-After`) after a 2 s wait for a slot.
`tools/load_test.py` against it on this laptop CPU, real models:

| Load | Served | Shed (503) | p50 / p95 of served | Throughput |
|---|---|---|---|---|
| 1 client, the 37 eval questions in turn | 37 | 0 | 1.3 s / 6.1 s | 0.57 answers/s |
| 100 concurrent clients, 200 requests | 4 | 196 (p95 3.5 s to the 503) | 3.4 s / 5.2 s | 0.53 answers/s |

The single-client run reproduces the eval exactly (27 answered, 7 refused,
3 blocked). The 100-client run shows the design working as intended rather
than the target being met: one CPU process delivers about 0.55 answers per
second and refuses the excess within seconds instead of queueing it for
minutes. Serving 100 concurrent users at, say, one question per user per
minute (~1.7 answers/s) therefore needs about four such processes, which
is the 4–8 worker figure in the table above; a sustained 100 questions per
second needs the GPU generation pool. The single-client p95 of 6.1 s
(mean 1.75 s) is a tail of a few slow answers; its cause was not isolated.

## 8. What I would do next

1. **Swap the generator, keep everything else.** The measurement says generation
   is the weak link: a 7B instruction-tuned model (Mistral, Qwen, Llama) behind
   the same verification and selection would raise LLM share without
   weakening the guardrail. The eval already measures whatever `--llm` points
   at, so this is a one-flag comparison.
2. **Improve the reranker before loosening the gate.** Three answerable questions
   are refused, one of them retrieved at rank 1. A better cross-encoder
   (`bge-reranker-base`, or a fine-tune on ~200 labelled pairs) would let the
   cut drop below 2.2 without admitting the out-of-scope question at 2.837.
3. **Add OCR** (Tesseract) for the 12 text-less pages, then re-run `--check-labels`.
4. **Grow the eval set** to ~100 questions with a second annotator and
   inter-annotator agreement, and report bootstrap confidence intervals so
   small differences stop looking meaningful.
5. **Replace the lexical support check with an NLI model** (`deberta-v3-small`),
   keeping the current check as a fast pre-filter.

## 9. End-to-end hardening

The pipeline was run end to end from a clean clone on a second Python version
(3.11, with transformers 5.22, torch 2.15 and sentence-transformers 6.3 as
resolved fresh from `requirements.txt`), and covered by 50 offline tests that
drive the CLI one process per step, exactly as the README does. That found
defects the metrics alone did not show:

| Defect | Effect | Fix |
|---|---|---|
| Short page tails were glued onto the previous page's chunk | 30 page tails, inside 45 of the 400 old chunks, cited under the wrong page; q29's answer had no chunk on its own page | Tails stay on their own page. recall@5 0.967 → 1.000, citation precision 0.733 → 0.767 |
| The offline hashing embedder used Python's salted `hash()` | An index from `ingest` never matched queries from `ask` (different process, different hash) | Stable `crc32`, idf refit at load |
| The relevance gate switched off for hybrid retrieval without the reranker | `--no-rerank` answered anything | Falls back to the dense pre-filter |
| `ask()` wrote passage numbers into shared chunk records | Concurrent requests in a service could swap citation numbers | Numbers are per request |
| README model download used a bare `snapshot_download` | ~3 GB of unused Flax/TF/`.bin` weights | `download-models` fetches only what is loaded |
| No input bounds | Unbounded questions reach the models | 3–1000 characters, control characters stripped |

Every fix has a test that failed before it and passes after. The gate
threshold was re-calibrated after the chunking fix and did not move (2.84).
The full evaluation was then re-run on both environments; all 54 non-latency
metrics are identical across them.

The HTTP service (`prototype/service.py`) adds the serving concerns only:
request validation, an optional API key, readiness, bounded in-flight work,
timeouts and JSON request logs without question text. Retrieval mode is not a
request field, because sparse-only retrieval skips the reranker and with it
the calibrated gate. CI (`.github/workflows/ci.yml`) is configured to run the
offline tests on Python 3.10–3.12 and the README path with the real models on a
clean Linux runner; its first run happens on push. Verified so far: Windows,
Python 3.14 and 3.11.

## 10. Reproduce

From `prototype/` (Python 3.10+, ~2 GB disk, CPU only, no API keys):

```bash
pip install -r requirements.txt
python rag_prototype.py download-models         # embedder, reranker, generator (~1.2 GB)
python rag_prototype.py ingest                  # build the index (~30 s)
python rag_prototype.py ask "What causes vibration and noise from the cyclone fan?"
python rag_prototype.py calibrate               # re-derive the gate threshold
python rag_prototype.py eval --check-labels     # verify the gold labels
python rag_prototype.py eval                    # full ablation + guardrail ladder (3–5 min)
python tools/make_deck.py                       # rebuild ../architecture_diagram.pptx
pip install -r requirements-dev.txt && python -m pytest tests -q   # 50 offline tests
uvicorn service:app --port 8000                 # HTTP service; tools/load_test.py measures it
```

Every number in this document and in the deck is read from
`prototype/evaluation.csv`, `prototype/runs/` or `prototype/index/manifest.json`.
Nothing is hand-copied.
