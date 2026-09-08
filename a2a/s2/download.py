"""Download one Semantic Scholar dataset (``papers``, ``abstracts``, ``citations``).

Each dataset ships as gzipped JSONL shards. Downloads are rate-limited to the
key's ceiling (1 req/s to the S2 API), parallel over S3 shard URLs, and
idempotent: shards already on disk are skipped, so a killed run resumes.

    S2_API_KEY=... python -m a2a.s2.download papers    /data/s2/papers
    S2_API_KEY=... python -m a2a.s2.download abstracts /data/s2/abstracts
    S2_API_KEY=... python -m a2a.s2.download citations /data/s2/citations --workers 16

Sizes (March 2026 release): papers ~36 GB (60 shards), abstracts ~23 GB
(30 shards), citations ~340 GB (358 shards).
"""

import argparse

from .api import download_dataset

DATASETS = ("papers", "abstracts", "citations")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("local_path", help="Destination directory for shards")
    ap.add_argument("--release", default=None, help="Release ID (default: latest)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel download threads (default 8)")
    args = ap.parse_args()
    download_dataset(args.local_path, dataset=args.dataset,
                     release_id=args.release, workers=args.workers)


if __name__ == "__main__":
    main()
