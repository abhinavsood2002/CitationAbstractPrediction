"""Shared S2 datasets-API plumbing: rate-limited GETs + parallel shard download.

Public surface: ``download_dataset`` and ``get_latest_release_id``. The CLI in
``download.py`` calls in with the dataset name pinned.

A dataset ships as many gzipped JSONL shards per release. This module:

* Rate-limits S2 API calls to ≤ 1 req/s globally (the key's documented
  ceiling), thread-safe under any worker count.
* Retries 429s with exponential backoff, honouring ``Retry-After`` if sent.
* Downloads shards in parallel (default 8 threads) — the rate limit only
  applies to ``api.semanticscholar.org``, not S3 shard URLs.
* Is idempotent — a shard already on disk is skipped; partial shards live at
  ``{shard}.gz.part`` and are renamed only on success.
* Refreshes expired pre-signed S3 URLs mid-run, coordinated across workers
  so a failure burst causes one refresh, not N.

Set ``S2_API_KEY`` in the environment (free key at
https://www.semanticscholar.org/product/api) or pass ``api_key`` explicitly.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

API_BASE = "https://api.semanticscholar.org/datasets/v1"
MIN_S2_INTERVAL = 1.1  # seconds between S2 API calls — pad slightly over 1/s

_last_s2_call = 0.0
_s2_lock = threading.Lock()


def _s2_get(url: str, headers: dict | None = None, max_retries: int = 6):
    """GET an S2 API endpoint with rate-limit + exponential backoff on 429.

    Thread-safe: holds ``_s2_lock`` across the request so concurrent callers
    can't both observe the rate-limit window as expired and fire together.
    S2 API calls are rare (URL list + occasional refreshes), so serializing
    them is exactly what we want.
    """
    global _last_s2_call
    for attempt in range(max_retries):
        with _s2_lock:
            gap = time.time() - _last_s2_call
            if gap < MIN_S2_INTERVAL:
                time.sleep(MIN_S2_INTERVAL - gap)
            resp = requests.get(url, headers=headers)
            _last_s2_call = time.time()
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        wait = int(resp.headers.get("Retry-After", 2 ** attempt))
        print(f"  429 — sleeping {wait}s (attempt {attempt + 1}/{max_retries})")
        time.sleep(wait)
    resp.raise_for_status()


def get_latest_release_id() -> str:
    """Return the ID of the most recent Semantic Scholar datasets release."""
    return _s2_get(f"{API_BASE}/release/latest").json()["release_id"]


def _fetch_shard_urls(release_id: str, dataset: str, api_key: str) -> dict[str, str]:
    """{shard_stem: presigned_url} for every shard of a dataset release."""
    resp = _s2_get(
        f"{API_BASE}/release/{release_id}/dataset/{dataset}/",
        headers={"x-api-key": api_key},
    )
    return {Path(urlparse(u).path).stem: u for u in resp.json()["files"]}


def _download_shard(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream one shard to ``dest``, renaming the .part file only on success."""
    tmp = dest.parent / (dest.name + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def download_dataset(
    local_path: str | Path,
    dataset: str = "s2orc_v2",
    release_id: str | None = None,
    api_key: str | None = None,
    max_retries_per_shard: int = 3,
    workers: int = 8,
) -> list[Path]:
    """Download every shard of ``dataset`` into ``local_path``.

    ``workers`` threads pull shards in parallel from S3. On shard failure
    (most commonly an expired pre-signed URL), the first failing worker
    re-queries S2 for fresh URLs once; concurrent failers see the bumped
    version and reuse the refreshed dict instead of refetching N times.
    Returns the list of local shard paths (already-present + downloaded).
    """
    api_key = api_key or os.environ.get("S2_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set S2_API_KEY or pass api_key=...")

    release_id = release_id or get_latest_release_id()
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    urls = _fetch_shard_urls(release_id, dataset, api_key)
    url_lock = threading.Lock()
    url_version = [0]  # boxed so closures can mutate

    todo = [stem for stem in urls if not (local_path / f"{stem}.gz").exists()]
    done = [local_path / f"{stem}.gz" for stem in urls
            if (local_path / f"{stem}.gz").exists()]
    print(f"{dataset}@{release_id}: {len(urls)} shards total, "
          f"{len(done)} already on disk, {len(todo)} to fetch with {workers} workers")

    def _download_one(stem: str) -> Path:
        dest = local_path / f"{stem}.gz"
        seen_version = url_version[0]
        url = urls[stem]
        for attempt in range(max_retries_per_shard):
            try:
                _download_shard(url, dest)
                return dest
            except (requests.RequestException, OSError) as e:
                with url_lock:
                    if url_version[0] == seen_version:
                        # First failer in this burst: refresh URLs once.
                        print(f"\n  {stem}: {e} — refreshing URLs "
                              f"(attempt {attempt + 1}/{max_retries_per_shard})")
                        fresh = _fetch_shard_urls(release_id, dataset, api_key)
                        urls.clear()
                        urls.update(fresh)
                        url_version[0] += 1
                    seen_version = url_version[0]
                    if stem not in urls:
                        raise RuntimeError(
                            f"{stem} missing from refreshed URL list") from e
                    url = urls[stem]
        raise RuntimeError(f"Failed {stem} after {max_retries_per_shard} attempts")

    failures: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_download_one, s): s for s in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="shards"):
            stem = futures[fut]
            try:
                done.append(fut.result())
            except BaseException as e:
                print(f"\n  FAILED {stem}: {e}")
                failures.append((stem, e))

    if failures:
        names = ", ".join(s for s, _ in failures[:5])
        more = "" if len(failures) <= 5 else f" (+{len(failures) - 5} more)"
        raise RuntimeError(
            f"{len(failures)}/{len(todo)} shards failed: {names}{more}. "
            f"Re-run to retry — completed shards will be skipped.")

    return done
