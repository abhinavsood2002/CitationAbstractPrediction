"""ACL-seed corpus: ACL Anthology papers as seeds, *any* citing paper as target.

Two layers, both built with DuckDB over the hash-bucketed parquet produced by
``a2a.s2.build_parquet``:

1. **Corpus** (``build_seeds`` / ``build_edges`` / ``build_citers``): every
   S2 paper with an ACL Anthology id and an abstract; every citation edge
   pointing at one of them (deduplicated per pair); metadata + abstract for
   every citing paper, from any venue or field. Written as three parquet
   artefacts under ``<out_root>/acl_corpus/``.
2. **Benchmark** (``make_benchmark``): qualifying (seed, citer) pairs under
   fixed filters, every eligible seed with every qualifying citer (optional
   hash-prefix caps), and a hash-based dev/test split. Written as seed-grouped
   JSONL plus an id manifest that reproduces the cut exactly.

Filters (see DECISIONS.md, entries 2026-09-08 "Benchmark filters" and
2026-09-09 "Final cut"):

* seed is in the top ``seed_citation_top_frac`` of *all* ACL Anthology papers
  by S2 ``citationcount`` but not in the top ``seed_citation_cap_frac`` (the
  most-cited papers hold too many pairs); ``acl_citation_quantile`` resolves
  both integer bounds
* seed year is not NULL (``seed_year_min`` / ``seed_year_max`` are optional)
* seed abstract and citer abstract both ``abstract_min_chars`` to
  ``abstract_max_chars`` characters (wide bounds: drops placeholders, stubs
  and pasted full texts only)
* citer year >= seed year; citer is not the seed
* no author-name overlap between seed and citer (self-citations are
  trivially anticipatable and excluded)
* seeds with at least ``min_citers`` qualifying citers (default 1: a seed
  with nothing to cover cannot be scored)
* optional floor on citer citation count (default 0: keep every citer, since
  the point of the ACL-seed corpus is *diverse* citers)
* no sampling by default: every eligible seed and every qualifying citer is
  kept (``n_seeds`` / ``citers_per_seed`` are optional caps)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pyarrow as pa

DERIVED_ROOT = Path(os.environ.get("A2A_DERIVED_ROOT",
                                   "/mnt/scratch/abhinav/quality_diversity/derived"))
N_BUCKETS = 64
DEV_MOD = 10  # 1-in-10 seeds -> dev pool (stable BLAKE2b hash of the seed id)


# ------------------------------------------------------------------ helpers

def connect(memory_limit: str = "40GB", threads: int = 16,
            temp_dir: str | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{temp_dir}'")
    return con


def stable_hash(*parts) -> int:
    h = hashlib.blake2b("\t".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def in_dev_pool(seed_id) -> bool:
    return stable_hash(seed_id) % DEV_MOD == 0


def _copy_atomic(con, select_sql: str, out: Path) -> int:
    tmp = out.with_suffix(out.suffix + ".tmp")
    con.execute(f"COPY ({select_sql}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION zstd)")
    os.replace(tmp, out)
    return con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]


def _acl_source(derived: Path, acl_ids: Path | None) -> str:
    """SQL relation (corpusid, acl_id) of ACL Anthology papers."""
    if acl_ids is not None:
        return f"(SELECT corpusid, acl_id FROM read_parquet('{acl_ids}'))"
    cols = [r[0] for r in duckdb.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{derived}/papers/bucket=0/*.parquet')").fetchall()]
    if "acl_id" not in cols:
        raise RuntimeError("papers parquet has no acl_id column; pass acl_ids="
                           "<derived>/acl_ids/acl_papers.parquet (build_parquet acl_ids)")
    return (f"(SELECT corpusid, acl_id FROM read_parquet('{derived}/papers/bucket=*/*.parquet') "
            f"WHERE acl_id IS NOT NULL)")


PAPER_COLS = """p.title, p.authors, p.year, p.publicationdate, p.venue, p.journal_name,
    p.referencecount, p.citationcount, p.influentialcitationcount,
    p.s2fieldsofstudy, p.publicationtypes"""


# ------------------------------------------------------------------- corpus

def build_seeds(con, out: Path, derived: Path = DERIVED_ROOT,
                acl_ids: Path | None = None) -> int:
    """All ACL Anthology papers with an abstract -> seeds.parquet."""
    sql = f"""
        SELECT p.corpusid, a.acl_id, ab.abstract, {PAPER_COLS}
        FROM {_acl_source(derived, acl_ids)} a
        JOIN read_parquet('{derived}/papers/bucket=*/*.parquet') p USING (corpusid)
        JOIN read_parquet('{derived}/abstracts/bucket=*/*.parquet') ab USING (corpusid)
        WHERE ab.abstract IS NOT NULL
        ORDER BY p.corpusid
    """
    return _copy_atomic(con, sql, out)


def build_edges(con, seeds: Path, out_dir: Path, derived: Path = DERIVED_ROOT,
                buckets: list[int] | None = None, resume: bool = True) -> int:
    """Every citation edge whose cited paper is a seed, one row per pair.

    The citers parquet is bucketed on ``citedcorpusid % 64`` so each bucket is
    an independent scan; a killed run resumes at the first bucket without an
    output file. The S2 release ships most (cited, citing) pairs twice (a
    partial re-export; see corpus_statistics/raw_data_stats.md), the copies
    sometimes differing only in NULL vs populated contexts; we keep one row
    with the non-NULL contexts/intents and OR the influential flag.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"CREATE OR REPLACE TABLE seed_ids AS "
                f"SELECT corpusid FROM read_parquet('{seeds}')")
    total = 0
    for b in buckets or range(N_BUCKETS):
        out = out_dir / f"bucket_{b:02d}.parquet"
        if resume and out.exists():
            continue
        t0 = time.time()
        sql = f"""
            SELECT citedcorpusid, citingcorpusid,
                   bool_or(isinfluential) AS isinfluential,
                   first(contexts ORDER BY (contexts IS NULL), citationid) AS contexts,
                   first(intents  ORDER BY (intents  IS NULL), citationid) AS intents
            FROM read_parquet('{derived}/citers/bucket={b}/*.parquet')
            WHERE citedcorpusid IN (SELECT corpusid FROM seed_ids)
              AND citingcorpusid IS NOT NULL
            GROUP BY citedcorpusid, citingcorpusid
        """
        n = _copy_atomic(con, sql, out)
        total += n
        print(f"edges bucket {b:2d}/63: {n:>9,} rows in {time.time() - t0:5.1f}s", flush=True)
    return total


def build_citers(con, edges_dir: Path, out: Path, derived: Path = DERIVED_ROOT,
                 acl_ids: Path | None = None) -> int:
    """Metadata + abstract for every distinct citing paper -> citers.parquet.

    Abstract is LEFT-joined (nullable) so the corpus records how many citers
    have no abstract on file; ``acl_id`` marks citers that are themselves ACL
    Anthology papers.
    """
    sql = f"""
        WITH ids AS (SELECT DISTINCT citingcorpusid AS corpusid
                     FROM read_parquet('{edges_dir}/bucket_*.parquet'))
        SELECT p.corpusid, a.acl_id, ab.abstract, {PAPER_COLS}
        FROM ids
        JOIN read_parquet('{derived}/papers/bucket=*/*.parquet') p USING (corpusid)
        LEFT JOIN read_parquet('{derived}/abstracts/bucket=*/*.parquet') ab USING (corpusid)
        LEFT JOIN {_acl_source(derived, acl_ids)} a USING (corpusid)
        ORDER BY p.corpusid
    """
    return _copy_atomic(con, sql, out)


# ---------------------------------------------------------------- benchmark

@dataclass
class BenchmarkParams:
    """Benchmark filters; ``None`` means "no bound". Defaults are the final cut
    (DECISIONS.md 2026-09-09). ``min_seed_citations`` / ``max_seed_citations`` are
    normally resolved from the two fractions by ``acl_citation_quantile`` and stored
    in the manifest."""
    seed_year_min: int | None = None       # seed year must be non-NULL; bounds are optional
    seed_year_max: int | None = None
    abstract_min_chars: int = 200          # seed and citer; drops placeholders and stubs
    abstract_max_chars: int = 5000         # drops pasted full texts / thesis summaries
    seed_citation_top_frac: float = 0.2    # seeds = top 20% of ACL papers by citationcount
    seed_citation_cap_frac: float = 0.01   # ... minus the top 1% (too many pairs per seed)
    min_seed_citations: int | None = None  # resolved floor (inclusive); None = no floor
    max_seed_citations: int | None = None  # resolved cap (inclusive); None = no cap
    min_citers: int = 1                    # qualifying citers needed to be scorable
    min_citer_citations: int = 0
    n_seeds: int | None = None             # None = every eligible seed
    citers_per_seed: int | None = None     # None = every qualifying citer
    dev_mod: int = DEV_MOD


def acl_citation_quantile(con, q: float, derived: Path = DERIVED_ROOT,
                          acl_ids: Path | None = None) -> int:
    """Discrete ``q`` quantile of S2 ``citationcount`` over *all* ACL Anthology papers.

    The population is every paper with an ACL id in the release, with or without an
    abstract, so the bounds do not move with abstract coverage. The floor is
    ``q = 1 - seed_citation_top_frac`` and the cap ``q = 1 - seed_citation_cap_frac``;
    both are applied inclusively (a paper at exactly the floor or the cap is kept).
    """
    if not 0 < q < 1:
        raise ValueError(f"quantile must be in (0, 1), got {q}")
    row = con.execute(f"""
        SELECT quantile_disc(p.citationcount, {q})
        FROM {_acl_source(derived, acl_ids)} a
        JOIN read_parquet('{derived}/papers/bucket=*/*.parquet') p USING (corpusid)
        WHERE p.citationcount IS NOT NULL
    """).fetchone()
    return int(row[0])


def qualifying_pairs_sql(seeds: Path, edges_dir: Path, citers: Path,
                         p: BenchmarkParams, seed_ids_table: str | None = None) -> str:
    year = ["s.year IS NOT NULL"]
    if p.seed_year_min is not None:
        year.append(f"s.year >= {p.seed_year_min}")
    if p.seed_year_max is not None:
        year.append(f"s.year <= {p.seed_year_max}")
    floor = (f"AND coalesce(s.citationcount, 0) >= {p.min_seed_citations}"
             if p.min_seed_citations is not None else "")
    cap = (f"AND coalesce(s.citationcount, 0) <= {p.max_seed_citations}"
           if p.max_seed_citations is not None else "")
    restrict = f"AND s.corpusid IN (SELECT corpusid FROM {seed_ids_table})" if seed_ids_table else ""
    return f"""
        SELECT s.corpusid AS seed_id, c.corpusid AS citer_id
        FROM read_parquet('{edges_dir}/bucket_*.parquet') e
        JOIN read_parquet('{seeds}') s ON s.corpusid = e.citedcorpusid
        JOIN read_parquet('{citers}') c ON c.corpusid = e.citingcorpusid
        WHERE {' AND '.join(year)}
          AND length(s.abstract) BETWEEN {p.abstract_min_chars} AND {p.abstract_max_chars}
          AND c.abstract IS NOT NULL
          AND length(c.abstract) BETWEEN {p.abstract_min_chars} AND {p.abstract_max_chars}
          AND c.year IS NOT NULL AND c.year >= s.year
          AND c.corpusid <> s.corpusid
          AND NOT coalesce(list_has_any(c.authors, s.authors), FALSE)
          AND coalesce(c.citationcount, 0) >= {p.min_citer_citations}
          {floor}
          {cap}
          {restrict}
    """


def _flat_intents(intents) -> list[str]:
    out = []
    for lst in intents or []:
        for x in lst or []:
            if x and x not in out:
                out.append(x)
    return out


def make_benchmark(con, seeds: Path, edges_dir: Path, citers: Path, out_jsonl: Path,
                   out_ids: Path, p: BenchmarkParams) -> dict:
    """Select seeds and citers deterministically; write JSONL + id manifest.

    Seeds and citers are ordered by ``stable_hash`` so the JSONL is reproducible
    and optional caps (``n_seeds``, ``citers_per_seed``) are hash prefixes.
    ``p.min_seed_citations`` / ``p.max_seed_citations`` must already be resolved
    (see ``acl_citation_quantile``).
    """
    t0 = time.time()
    pairs_sql = qualifying_pairs_sql(seeds, edges_dir, citers, p)
    con.execute(f"CREATE OR REPLACE TABLE qpairs AS {pairs_sql}")
    counts = con.execute("SELECT seed_id, count(*) FROM qpairs GROUP BY 1").fetchall()
    eligible = sorted((sid for sid, n in counts if n >= p.min_citers),
                      key=lambda s: stable_hash("seed", s))
    chosen = eligible if p.n_seeds is None else eligible[:p.n_seeds]
    n_qual = dict(counts)
    print(f"{len(counts):,} seeds with >=1 qualifying citer; {len(eligible):,} eligible "
          f"(>= {p.min_citers}); choosing {len(chosen):,} [{time.time() - t0:.0f}s]", flush=True)

    con.register("chosen_arrow", pa.table({"corpusid": pa.array(chosen, pa.int64())}))
    con.execute("CREATE OR REPLACE TABLE chosen AS SELECT * FROM chosen_arrow")
    rows = con.execute("SELECT seed_id, citer_id FROM qpairs "
                       "WHERE seed_id IN (SELECT corpusid FROM chosen)").fetchall()
    by_seed: dict[int, list[int]] = {}
    for sid, cid in rows:
        by_seed.setdefault(sid, []).append(cid)
    sampled = {sid: sorted(cids, key=lambda c: stable_hash("citer", sid, c))[:p.citers_per_seed]
               for sid, cids in by_seed.items()}

    con.register("pick_arrow", pa.table({
        "seed_id": pa.array([s for s, cs in sampled.items() for _ in cs], pa.int64()),
        "citer_id": pa.array([c for cs in sampled.values() for c in cs], pa.int64())}))
    con.execute("CREATE OR REPLACE TABLE pick AS SELECT * FROM pick_arrow")
    seed_rows = con.execute(f"""
        SELECT s.corpusid, s.acl_id, s.title, s.abstract, s.authors, s.year, s.venue,
               s.citationcount, s.s2fieldsofstudy
        FROM read_parquet('{seeds}') s WHERE s.corpusid IN (SELECT corpusid FROM chosen)
    """).fetchall()
    citer_rows = con.execute(f"""
        SELECT k.seed_id, c.corpusid, c.acl_id, c.title, c.abstract, c.year, c.venue,
               c.citationcount, c.s2fieldsofstudy, e.isinfluential, e.intents,
               coalesce(len(e.contexts), 0)
        FROM pick k
        JOIN read_parquet('{citers}') c ON c.corpusid = k.citer_id
        JOIN read_parquet('{edges_dir}/bucket_*.parquet') e
             ON e.citedcorpusid = k.seed_id AND e.citingcorpusid = k.citer_id
    """).fetchall()
    citers_of: dict[int, dict[int, dict]] = {}
    for (sid, cid, acl, title, abstract, year, venue, cc, fos, infl, intents, nctx) in citer_rows:
        citers_of.setdefault(sid, {})[cid] = {
            "id": str(cid), "acl_id": acl, "title": title, "abstract": abstract,
            "year": year, "venue": venue, "citationcount": cc, "fields": fos or [],
            "isinfluential": bool(infl), "intents": _flat_intents(intents),
            "n_contexts": int(nctx),
        }

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"params": asdict(p), "seeds": {}}
    n_pairs = 0
    with open(out_jsonl, "w") as f:
        for (sid, acl, title, abstract, authors, year, venue, cc, fos) in sorted(seed_rows):
            cids = sampled[sid]
            split = "dev" if stable_hash(sid) % p.dev_mod == 0 else "test"
            rec = {
                "seed_id": str(sid), "acl_id": acl, "split": split,
                "seed_title": title, "seed_abstract": abstract,
                "seed_authors": authors or [], "seed_year": year, "seed_venue": venue,
                "seed_citationcount": cc, "seed_fields": fos or [],
                "n_citers_qualifying": int(n_qual[sid]),
                "citers": [citers_of[sid][c] for c in cids],
            }
            n_pairs += len(rec["citers"])
            f.write(json.dumps(rec) + "\n")
            manifest["seeds"][str(sid)] = {"split": split, "citers": [str(c) for c in cids]}
    with open(out_ids, "w") as f:
        json.dump(manifest, f)
    stats = {
        "n_seeds_any_qualifying": len(counts), "n_seeds_eligible": len(eligible),
        "n_seeds": len(chosen), "n_pairs": n_pairs,
        "n_dev": sum(1 for s in chosen if stable_hash(s) % p.dev_mod == 0),
        "elapsed_s": round(time.time() - t0, 1), "params": asdict(p),
    }
    print(json.dumps(stats, indent=1), flush=True)
    return stats


# ---------------------------------------------------------------------- I/O

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    shutil.move(tmp, path)
