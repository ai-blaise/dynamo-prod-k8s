#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Streaming OpenAI benchmark for isolated MORI-IO transport variants.

The script is intentionally dependency-free so it can run from the B200 host or
inside a temporary pod. It measures request latency, TTFT, TPOT/ITL after first
token, tokens/sec/user, and abort cleanup triggerability for one reachable
OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class Result:
    request_index: int
    ok: bool
    status: int | None
    error: str | None
    latency_ms: float
    ttft_ms: float | None
    output_chunks: int
    output_chars: int
    tpot_ms: float | None
    tokens_per_sec_user_after_ft: float | None
    aborted_after_first_chunk: bool = False
    prompt_label: str | None = None
    prompt_tokens_target: int | None = None


def _parse_sse(raw: bytes) -> list[dict[str, Any]]:
    events = []
    for line in raw.decode('utf-8', errors='replace').splitlines():
        line = line.strip()
        if not line or not line.startswith('data:'):
            continue
        payload = line[5:].strip()
        if payload == '[DONE]':
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            events.append({'_decode_error': payload})
    return events


def synthetic_prompt(token_target: int) -> str:
    if token_target <= 0:
        raise ValueError('--prompt-tokens must be positive')
    # The OpenAI endpoint owns true tokenization. This generator intentionally
    # uses simple whitespace-separated words so long-context benchmark variants
    # are reproducible without importing tokenizer/model packages on the host.
    seed = (
        'moriio layersplit cp2 decode cp1 kvarn k2v2 dense mla latent kv '
        'indexer fp4 hisa dsa warpdecode smc request pinning cleanup abort '
        'nixl ucx mooncake transfer metadata steady state throughput '
    ).split()
    words = (seed * ((token_target // len(seed)) + 1))[:token_target]
    return ' '.join(words)


def load_prompt(args: argparse.Namespace) -> tuple[str, str, int | None]:
    if args.prompt_file:
        text = Path(args.prompt_file).read_text()
        return text, f'file:{args.prompt_file}', None
    if args.prompt_tokens:
        return synthetic_prompt(args.prompt_tokens), f'synthetic-{args.prompt_tokens}', args.prompt_tokens
    return args.prompt, 'inline', None


class NvidiaSmiSampler:
    def __init__(self, enabled: bool, interval_sec: float) -> None:
        self.enabled = enabled
        self.interval_sec = max(interval_sec, 0.25)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_sec * 2))

    def _run(self) -> None:
        query = 'timestamp,index,utilization.gpu,memory.used,memory.total,power.draw'
        while not self._stop.is_set():
            started = time.time()
            cp = subprocess.run(
                ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
                check=False,
                text=True,
                capture_output=True,
            )
            self.samples.append({
                'time': started,
                'returncode': cp.returncode,
                'stdout': cp.stdout.strip(),
                'stderr': cp.stderr.strip()[:1000],
            })
            self._stop.wait(self.interval_sec)


def one_request(url: str, model: str, prompt: str, max_tokens: int, request_index: int, abort_after_first: bool) -> Result:
    body = {
        'model': model,
        'stream': True,
        'max_tokens': max_tokens,
        'temperature': 0,
        'messages': [{'role': 'user', 'content': prompt}],
    }
    req = urllib.request.Request(
        url.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(body).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    start = time.perf_counter()
    first = None
    last = start
    chunks = 0
    chars = 0
    status = None
    aborted = False
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            status = resp.status
            while True:
                line = resp.readline()
                now = time.perf_counter()
                if not line:
                    break
                for event in _parse_sse(line):
                    delta = ''
                    try:
                        delta = event.get('choices', [{}])[0].get('delta', {}).get('content') or ''
                    except Exception:
                        delta = ''
                    if delta:
                        if first is None:
                            first = now
                        last = now
                        chunks += 1
                        chars += len(delta)
                        if abort_after_first:
                            aborted = True
                            raise RuntimeError('intentional abort after first streamed chunk')
    except RuntimeError as exc:
        if not abort_after_first:
            raise
        end = time.perf_counter()
        ttft = None if first is None else (first - start) * 1000
        return Result(request_index, True, status, str(exc), (end - start) * 1000, ttft, chunks, chars, None, None, aborted)
    except urllib.error.HTTPError as exc:
        end = time.perf_counter()
        return Result(request_index, False, exc.code, exc.read().decode('utf-8', errors='replace')[:2000], (end - start) * 1000, None, chunks, chars, None, None)
    except Exception as exc:
        end = time.perf_counter()
        return Result(request_index, False, status, repr(exc), (end - start) * 1000, None, chunks, chars, None, None)
    end = time.perf_counter()
    ttft = None if first is None else (first - start) * 1000
    span_after_ft = max(last - first, 0.0) if first is not None else 0.0
    intervals = max(chunks - 1, 0)
    tpot = (span_after_ft / intervals * 1000) if intervals else None
    tps = (intervals / span_after_ft) if span_after_ft > 0 and intervals else None
    return Result(request_index, True, status, None, (end - start) * 1000, ttft, chunks, chars, tpot, tps)


def summarize(results: list[Result], wall_time_ms: float) -> dict[str, Any]:
    ok = [r for r in results if r.ok and not r.aborted_after_first_chunk]
    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        values = sorted(values)
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
        return values[idx]
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms is not None]
    tps = [r.tokens_per_sec_user_after_ft for r in ok if r.tokens_per_sec_user_after_ft is not None]
    return {
        'requests': len(results),
        'wall_time_ms': wall_time_ms,
        'aggregate_output_chunks': sum(r.output_chunks for r in ok),
        'success': len(ok),
        'failed': len([r for r in results if not r.ok]),
        'aborted': len([r for r in results if r.aborted_after_first_chunk]),
        'ttft_ms_avg': statistics.mean(ttfts) if ttfts else None,
        'ttft_ms_p50': pct(ttfts, 0.50),
        'ttft_ms_p95': pct(ttfts, 0.95),
        'tpot_ms_avg': statistics.mean(tpots) if tpots else None,
        'tpot_ms_p50': pct(tpots, 0.50),
        'tpot_ms_p95': pct(tpots, 0.95),
        'tokens_per_sec_user_after_ft_avg': statistics.mean(tps) if tps else None,
        'tokens_per_sec_user_after_ft_p50': pct(tps, 0.50),
        'tokens_per_sec_user_after_ft_p95': pct(tps, 0.95),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True, help='Base URL, for example http://10.43.x.y:8000')
    ap.add_argument('--model', required=True)
    ap.add_argument('--prompt', default='Write a concise checklist for safe KV cache handoff validation.')
    ap.add_argument('--prompt-file', help='Read the prompt body from a file for exact long-context replay.')
    ap.add_argument('--prompt-tokens', type=int, help='Generate a deterministic synthetic prompt with this many whitespace token surrogates.')
    ap.add_argument('--requests', type=int, default=4)
    ap.add_argument('--concurrency', type=int, default=1)
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--abort-one', action='store_true', help='Close one stream after the first chunk to exercise cleanup logs.')
    ap.add_argument('--sample-nvidia-smi', action='store_true', help='Sample GPU util/memory/power while the request batch runs.')
    ap.add_argument('--sample-interval-sec', type=float, default=1.0)
    ap.add_argument('--out', default='-')
    args = ap.parse_args()
    prompt, prompt_label, prompt_tokens_target = load_prompt(args)

    results: list[Result] = []
    lock = threading.Lock()
    next_idx = 0

    def worker() -> None:
        nonlocal next_idx
        while True:
            with lock:
                if next_idx >= args.requests:
                    return
                idx = next_idx
                next_idx += 1
            res = one_request(args.url, args.model, prompt, args.max_tokens, idx, args.abort_one and idx == args.requests - 1)
            res.prompt_label = prompt_label
            res.prompt_tokens_target = prompt_tokens_target
            with lock:
                results.append(res)

    sampler = NvidiaSmiSampler(args.sample_nvidia_smi, args.sample_interval_sec)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.concurrency)]
    bench_start = time.perf_counter()
    sampler.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    sampler.stop()
    wall_time_ms = (time.perf_counter() - bench_start) * 1000

    payload = {
        'config': {
            'url': args.url,
            'model': args.model,
            'requests': args.requests,
            'concurrency': args.concurrency,
            'max_tokens': args.max_tokens,
            'prompt_label': prompt_label,
            'prompt_tokens_target': prompt_tokens_target,
            'prompt_chars': len(prompt),
            'abort_one': args.abort_one,
            'sample_nvidia_smi': args.sample_nvidia_smi,
        },
        'summary': summarize(results, wall_time_ms),
        'nvidia_smi_samples': sampler.samples,
        'results': [asdict(r) for r in sorted(results, key=lambda r: r.request_index)],
    }
    text = json.dumps(payload, indent=2)
    if args.out == '-':
        print(text)
    else:
        Path(args.out).write_text(text)
    return 1 if payload['summary']['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
