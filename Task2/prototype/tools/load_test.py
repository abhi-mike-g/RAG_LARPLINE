#!/usr/bin/env python3
"""Fire concurrent questions at the running service and report what happened.

    python tools/load_test.py --url http://127.0.0.1:8000 --users 100 --requests 200

Reports status-code counts and latency percentiles for successful answers.
503 responses are load shedding (the service refused rather than queued),
not crashes; the report keeps them separate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parents[1]


def questions() -> list[str]:
    path = HERE / "eval_set.jsonl"
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [i["question"] for i in items]


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = min(len(values) - 1, max(0, round(p / 100 * (len(values) - 1))))
    return values[k]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--users", type=int, default=100, help="concurrent clients")
    ap.add_argument("--requests", type=int, default=200, help="total requests")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    qs = questions()
    headers = {"x-api-key": args.api_key} if args.api_key else {}
    codes: Counter[int | str] = Counter()
    ok_ms: list[float] = []
    shed_ms: list[float] = []
    statuses: Counter[str] = Counter()
    sem = asyncio.Semaphore(args.users)

    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout, headers=headers,
                                 limits=httpx.Limits(max_connections=args.users)) as client:
        async def one(i: int) -> None:
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post("/v1/ask", json={"question": qs[i % len(qs)]})
                except httpx.HTTPError as exc:
                    codes[type(exc).__name__] += 1
                    return
                ms = (time.perf_counter() - t0) * 1000
                codes[r.status_code] += 1
                if r.status_code == 200:
                    ok_ms.append(ms)
                    statuses[r.json()["status"]] += 1
                elif r.status_code == 503:
                    shed_ms.append(ms)

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(args.requests)))
        wall = time.perf_counter() - t0

    report = {
        "users": args.users, "requests": args.requests, "wall_s": round(wall, 1),
        "codes": {str(k): v for k, v in sorted(codes.items(), key=str)},
        "answer_status": dict(statuses),
        "ok_p50_ms": round(pct(ok_ms, 50), 1), "ok_p95_ms": round(pct(ok_ms, 95), 1),
        "ok_mean_ms": round(statistics.fmean(ok_ms), 1) if ok_ms else None,
        "shed_p95_ms": round(pct(shed_ms, 95), 1),
        "throughput_ok_per_s": round(len(ok_ms) / wall, 2),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
