"""Build hash-bucketed parquet from raw S2 shards (one pass per dataset).

* ``abstracts``: bucket on ``corpusid % 64``   -> (corpusid, abstract)
* ``citers``:    bucket on ``citedcorpusid % 64`` -> one citation edge per row
                 (rows with NULL citedcorpusid, ~11%, are dropped)
* ``papers``:    bucket on ``corpusid % 64``   -> metadata incl. ``acl_id``
                 (the ACL Anthology id from ``externalids.ACL``), ``doi``,
                 ``arxiv``, ``venue``, ``year``, ``s2fieldsofstudy``, counts
* ``acl_ids``:   side table (corpusid, acl_id, doi, arxiv, venue) for papers
                 layers built before ``acl_id`` was part of the schema

Each source shard ``X.gz`` produces up to 64 ``bucket=<N>/X.parquet`` files plus
a ``_done/X.ok`` marker; re-running skips finished shards. ``compact`` merges
each bucket into one file sorted on its lookup key so DuckDB point lookups
open one file and prune row groups by zone map.

    python -m a2a.s2.build_parquet papers    --source /data/s2/papers    --dest /data/derived/papers
    python -m a2a.s2.build_parquet abstracts --source /data/s2/abstracts --dest /data/derived/abstracts
    python -m a2a.s2.build_parquet citers    --source /data/s2/citations --dest /data/derived/citers
    python -m a2a.s2.build_parquet compact papers --dest /data/derived/papers
    python -m a2a.s2.build_parquet acl_ids   --source /data/s2/papers    --dest /data/derived/acl_ids
"""

import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import orjson

    def _loads(line):
        return orjson.loads(line)
except ImportError:  # pragma: no cover
    def _loads(line):
        return json.loads(line)

N_BUCKETS = 64

ABSTRACTS_SCHEMA = pa.schema([
    pa.field("corpusid", pa.int64()),
    pa.field("abstract", pa.string()),
])

CITERS_SCHEMA = pa.schema([
    pa.field("citationid", pa.int64()),
    pa.field("citingcorpusid", pa.int64()),
    pa.field("citedcorpusid", pa.int64()),
    pa.field("isinfluential", pa.bool_()),
    pa.field("contexts", pa.list_(pa.string())),
    pa.field("intents", pa.list_(pa.list_(pa.string()))),
])

PAPERS_SCHEMA = pa.schema([
    pa.field("corpusid", pa.int64()),
    pa.field("acl_id", pa.string()),      # externalids.ACL (ACL Anthology id) or null
    pa.field("doi", pa.string()),
    pa.field("arxiv", pa.string()),
    pa.field("title", pa.string()),
    pa.field("authors", pa.list_(pa.string())),
    pa.field("year", pa.int32()),
    pa.field("publicationdate", pa.string()),  # YYYY-MM-DD; null when S2 has only year
    pa.field("venue", pa.string()),
    pa.field("publicationvenueid", pa.string()),
    pa.field("journal_name", pa.string()),
    pa.field("referencecount", pa.int32()),
    pa.field("citationcount", pa.int32()),
    pa.field("influentialcitationcount", pa.int32()),
    pa.field("isopenaccess", pa.bool_()),
    pa.field("s2fieldsofstudy", pa.list_(pa.string())),
    pa.field("publicationtypes", pa.list_(pa.string())),
])

ACL_IDS_SCHEMA = pa.schema([
    pa.field("corpusid", pa.int64()),
    pa.field("acl_id", pa.string()),
    pa.field("doi", pa.string()),
    pa.field("arxiv", pa.string()),
    pa.field("venue", pa.string()),
    pa.field("publicationvenueid", pa.string()),
])


def _write_atomic(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def _shard_done(dest: Path, stem: str) -> bool:
    return (dest / "_done" / f"{stem}.ok").exists()


def _mark_done(dest: Path, stem: str) -> None:
    marker = dest / "_done" / f"{stem}.ok"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def _flush_buckets(buckets: dict, schema: pa.Schema, dest: Path, stem: str) -> None:
    for b, cols in buckets.items():
        if not next(iter(cols.values())):
            continue
        out_dir = dest / f"bucket={b}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_atomic(pa.Table.from_pydict(cols, schema=schema), out_dir / f"{stem}.parquet")


def _paper_row(rec: dict) -> dict:
    ext = rec.get("externalids") or {}
    authors = rec.get("authors") or []
    fos = rec.get("s2fieldsofstudy") or []
    j = rec.get("journal") or {}
    return {
        "corpusid": rec["corpusid"],
        "acl_id": ext.get("ACL") or None,
        "doi": ext.get("DOI") or None,
        "arxiv": ext.get("ArXiv") or None,
        "title": rec.get("title"),
        "authors": [a["name"] for a in authors if a and a.get("name")] or None,
        "year": rec.get("year"),
        "publicationdate": rec.get("publicationdate") or None,
        "venue": rec.get("venue") or None,
        "publicationvenueid": rec.get("publicationvenueid") or None,
        "journal_name": (j.get("name") if isinstance(j, dict) else None) or None,
        "referencecount": rec.get("referencecount"),
        "citationcount": rec.get("citationcount"),
        "influentialcitationcount": rec.get("influentialcitationcount"),
        "isopenaccess": rec.get("isopenaccess"),
        "s2fieldsofstudy": sorted({f["category"] for f in fos if f and f.get("category")}) or None,
        "publicationtypes": rec.get("publicationtypes"),
    }


def _process_abstracts(shard: Path, dest: Path) -> None:
    buckets = {b: {"corpusid": [], "abstract": []} for b in range(N_BUCKETS)}
    with gzip.open(shard, "rb") as f:
        for line in f:
            rec = _loads(line)
            cid = rec.get("corpusid")
            if cid is None:
                continue
            cols = buckets[cid % N_BUCKETS]
            cols["corpusid"].append(cid)
            cols["abstract"].append(rec.get("abstract"))
    _flush_buckets(buckets, ABSTRACTS_SCHEMA, dest, shard.stem)
    _mark_done(dest, shard.stem)


def _process_papers(shard: Path, dest: Path) -> None:
    buckets = {b: {f.name: [] for f in PAPERS_SCHEMA} for b in range(N_BUCKETS)}
    with gzip.open(shard, "rb") as f:
        for line in f:
            rec = _loads(line)
            if rec.get("corpusid") is None:
                continue
            row = _paper_row(rec)
            cols = buckets[row["corpusid"] % N_BUCKETS]
            for k, v in row.items():
                cols[k].append(v)
    _flush_buckets(buckets, PAPERS_SCHEMA, dest, shard.stem)
    _mark_done(dest, shard.stem)


def _process_citers(shard: Path, dest: Path) -> None:
    buckets = {b: {f.name: [] for f in CITERS_SCHEMA} for b in range(N_BUCKETS)}
    with gzip.open(shard, "rb") as f:
        for line in f:
            rec = _loads(line)
            cited = rec.get("citedcorpusid")
            if cited is None:
                continue
            cols = buckets[cited % N_BUCKETS]
            cols["citationid"].append(rec.get("citationid"))
            cols["citingcorpusid"].append(rec.get("citingcorpusid"))
            cols["citedcorpusid"].append(cited)
            cols["isinfluential"].append(rec.get("isinfluential"))
            cols["contexts"].append(rec.get("contexts"))
            cols["intents"].append(rec.get("intents"))
    _flush_buckets(buckets, CITERS_SCHEMA, dest, shard.stem)
    _mark_done(dest, shard.stem)


def _process_acl_ids(shard: Path, dest: Path) -> None:
    """Unbucketed side table: only papers with an ACL Anthology id (~0.05%)."""
    cols = {f.name: [] for f in ACL_IDS_SCHEMA}
    with gzip.open(shard, "rb") as f:
        for line in f:
            rec = _loads(line)
            ext = rec.get("externalids") or {}
            if not ext.get("ACL") or rec.get("corpusid") is None:
                continue
            cols["corpusid"].append(rec["corpusid"])
            cols["acl_id"].append(ext["ACL"])
            cols["doi"].append(ext.get("DOI"))
            cols["arxiv"].append(ext.get("ArXiv"))
            cols["venue"].append(rec.get("venue") or None)
            cols["publicationvenueid"].append(rec.get("publicationvenueid"))
    _write_atomic(pa.Table.from_pydict(cols, schema=ACL_IDS_SCHEMA), dest / f"{shard.stem}.parquet")
    _mark_done(dest, shard.stem)


PROCESSORS = {
    "abstracts": _process_abstracts,
    "citers": _process_citers,
    "papers": _process_papers,
    "acl_ids": _process_acl_ids,
}
SORT_KEY = {"abstracts": "corpusid", "citers": "citedcorpusid", "papers": "corpusid"}


def _compact_bucket(bucket_dir: Path, sort_col: str) -> None:
    import duckdb

    files = sorted(p for p in bucket_dir.glob("*.parquet")
                   if not p.name.endswith(".tmp") and p.name != "compacted.parquet")
    if not files:
        return
    out_tmp = bucket_dir / "compacted.parquet.tmp"
    files_arr = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect(":memory:")
    con.execute("SET threads=2")
    con.execute(f"COPY (SELECT * FROM read_parquet({files_arr}) ORDER BY {sort_col}) "
                f"TO '{out_tmp}' (FORMAT PARQUET, COMPRESSION 'zstd')")
    con.close()
    for f in files:
        f.unlink()
    os.replace(out_tmp, bucket_dir / "compacted.parquet")


def compact(kind: str, dest: Path, workers: int = 4) -> None:
    """Merge each bucket's per-shard files into one sorted parquet (re-runnable)."""
    if kind == "acl_ids":
        import duckdb
        con = duckdb.connect(":memory:")
        con.execute(f"COPY (SELECT * FROM read_parquet('{dest}/*.parquet') "
                    f"WHERE filename NOT LIKE '%acl_papers.parquet' ORDER BY corpusid) "
                    f"TO '{dest}/acl_papers.parquet.tmp' (FORMAT PARQUET, COMPRESSION 'zstd')")
        os.replace(dest / "acl_papers.parquet.tmp", dest / "acl_papers.parquet")
        return
    sort_col = SORT_KEY[kind]
    bucket_dirs = sorted(d for d in dest.glob("bucket=*") if d.is_dir())
    print(f"compacting {kind}: {len(bucket_dirs)} buckets, sort by {sort_col}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_compact_bucket, d, sort_col): d for d in bucket_dirs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="compact"):
            fut.result()


def _worker(kind: str, shard: str, dest: str) -> None:
    PROCESSORS[kind](Path(shard), Path(dest))


def build(kind: str, source: Path, dest: Path, workers: int = 8,
          max_shards: int | None = None) -> None:
    shards = sorted(source.glob("*.gz"))[:max_shards]
    todo = [s for s in shards if not _shard_done(dest, s.stem)]
    print(f"{kind}: {len(shards)} shards, {len(shards) - len(todo)} done, "
          f"{len(todo)} to process with {workers} processes")
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, kind, str(s), str(dest)): s for s in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=kind):
            try:
                fut.result()
            except BaseException as e:  # noqa: BLE001
                print(f"\n  FAILED {futs[fut].name}: {e}")
                failures.append(futs[fut].name)
    if failures:
        raise RuntimeError(f"{len(failures)}/{len(todo)} shards failed; re-run to retry")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for kind in sorted(PROCESSORS):
        p = sub.add_parser(kind, help=f"build derived/{kind} from raw .gz shards")
        p.add_argument("--source", required=True, type=Path)
        p.add_argument("--dest", required=True, type=Path)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--max-shards", type=int, default=None)
    pc = sub.add_parser("compact", help="merge per-shard files into one sorted parquet per bucket")
    pc.add_argument("kind", choices=sorted(PROCESSORS))
    pc.add_argument("--dest", required=True, type=Path)
    pc.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "compact":
        compact(args.kind, args.dest, args.workers)
    else:
        args.dest.mkdir(parents=True, exist_ok=True)
        build(args.cmd, args.source, args.dest, args.workers, args.max_shards)
        if args.cmd == "acl_ids":
            compact("acl_ids", args.dest)


if __name__ == "__main__":
    main()
