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
   hash-prefix caps). Written as seed-grouped JSONL plus an id manifest that
   reproduces the cut exactly.

Filters (see DECISIONS.md, entries 2026-09-08 "Benchmark filters" and
2026-09-19 "Final scope"):

* seed is in the top ``seed_infl_top_frac`` of *all* ACL Anthology papers
  **published the same year** by S2 ``influentialcitationcount``, but not in
  the top ``seed_infl_cap_frac`` (the canonical layer: tools, datasets and
  metrics whose citers are uses of an artefact); ``acl_influential_thresholds``
  resolves the per-year integer bounds
* seed year is not NULL (``seed_year_min`` / ``seed_year_max`` are optional)
* only edges S2 flags ``isinfluential`` (``influential_only``): incidental
  citations are only loosely conditioned on the seed's content
* edges whose only citation-intent label is ``background`` are dropped
  (``drop_background_only``); unlabelled edges are kept, because the intent
  classifier has not been run over most 2025+ citers
* seed abstract and citer abstract both ``abstract_min_chars`` to
  ``abstract_max_chars`` characters (wide bounds: drops placeholders, stubs
  and pasted full texts only)
* citer year >= seed year; citer is not the seed
* no author-name overlap between seed and citer (self-citations are
  trivially anticipatable and excluded)
* citer abstract detected as English (``english_only``; langdetect, seeded):
  seeds are ACL papers with English abstracts, and a handful of Japanese,
  Portuguese and Indonesian citer abstracts (0.1%) cannot be matched by an
  English prediction
* seeds with at least ``min_citers`` qualifying citers (default 10: per-seed
  coverage then has resolution <= 0.1)
* optional floor on citer citation count (default 0: keep every citer, since
  the point of the ACL-seed corpus is *diverse* citers)
* no sampling by default: every eligible seed and every qualifying citer is
  kept (``n_seeds`` / ``citers_per_seed`` are optional caps)

There is no dev/test split: the benchmark is evaluation-only and tau* (the
random-pool null percentile) does not depend on any system, so it is
calibrated on the whole set (DECISIONS.md 2026-09-19).
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
    """Benchmark filters; ``None`` means "no bound". Defaults are the final scope
    (DECISIONS.md 2026-09-19). ``seed_infl_thresholds`` is normally resolved from the two
    fractions by ``acl_influential_thresholds`` and stored in the manifest."""
    seed_year_min: int | None = None       # seed year must be non-NULL; bounds are optional
    seed_year_max: int | None = None
    abstract_min_chars: int = 200          # seed and citer; drops placeholders and stubs
    abstract_max_chars: int = 5000         # drops pasted full texts / thesis summaries
    seed_infl_top_frac: float = 0.10       # seeds = top 10% of ACL papers of the same year
    seed_infl_cap_frac: float = 0.01       #   by S2 influentialcitationcount, minus the top 1%
    seed_infl_thresholds: dict | None = None  # resolved {year: [floor, cap]}; None = no bound
    influential_only: bool = True          # keep only edges S2 flags as influential
    drop_background_only: bool = True      # drop edges whose only intent label is background
    english_only: bool = True              # drop citers whose abstract is not detected as English
    min_citers: int = 10                   # qualifying citers needed to be scorable
    min_citer_citations: int = 0
    n_seeds: int | None = None             # None = every eligible seed
    citers_per_seed: int | None = None     # None = every qualifying citer


def acl_influential_thresholds(con, top_frac: float, cap_frac: float,
                               derived: Path = DERIVED_ROOT,
                               acl_ids: Path | None = None) -> dict[int, list]:
    """Per publication year, the S2 ``influentialcitationcount`` floor (inclusive) and
    cap (exclusive) selecting the top ``top_frac`` of *all* ACL Anthology papers of that
    year minus the top ``cap_frac``: ``floor = quantile_disc(1 - top_frac)``,
    ``cap = quantile_disc(1 - cap_frac)``, so a paper at exactly the floor is kept and one
    at exactly the cap is in the excluded top slice.

    The population is every paper with an ACL id and a year in the release, with or
    without an abstract, so the thresholds do not move with abstract coverage. Within-year
    percentiles follow the bibliometric convention (Leiden PP top 10% / top 1%): raw counts
    are not comparable across years. ``top_frac = 0`` disables the floor (0) and
    ``cap_frac = 0`` disables the cap (None).
    """
    if not (0 <= top_frac < 1 and 0 <= cap_frac < 1):
        raise ValueError(f"fractions must be in [0, 1), got {top_frac}, {cap_frac}")
    if top_frac and cap_frac and cap_frac >= top_frac:
        raise ValueError(f"cap fraction {cap_frac} must be below top fraction {top_frac}")
    icc = "coalesce(p.influentialcitationcount, 0)"
    sel_floor = f"quantile_disc({icc}, {1 - top_frac})" if top_frac else "0"
    sel_cap = f"quantile_disc({icc}, {1 - cap_frac})" if cap_frac else "NULL"
    rows = con.execute(f"""
        SELECT p.year, {sel_floor}, {sel_cap}
        FROM {_acl_source(derived, acl_ids)} a
        JOIN read_parquet('{derived}/papers/bucket=*/*.parquet') p USING (corpusid)
        WHERE p.year IS NOT NULL
        GROUP BY p.year ORDER BY p.year
    """).fetchall()
    return {int(y): [int(f), None if c is None else int(c)] for y, f, c in rows}


# citation-intent labels of an edge as a deduplicated flat list (S2 stores one list per
# citation context; inner lists and elements can be NULL)
_INTENTS_SQL = ("list_distinct(list_filter(flatten(coalesce(e.intents, [])), "
                "x -> x IS NOT NULL AND x <> ''))")


def qualifying_pairs_sql(seeds: Path, edges_dir: Path, citers: Path,
                         p: BenchmarkParams, seed_ids_table: str | None = None) -> str:
    year = ["s.year IS NOT NULL"]
    if p.seed_year_min is not None:
        year.append(f"s.year >= {p.seed_year_min}")
    if p.seed_year_max is not None:
        year.append(f"s.year <= {p.seed_year_max}")
    thr_join = thr_where = ""
    if p.seed_infl_thresholds is not None:
        if not p.seed_infl_thresholds:
            raise ValueError("seed_infl_thresholds is empty; pass None for no seed bound")
        vals = ", ".join(f"({int(y)}, {int(f)}, {'NULL' if c is None else int(c)})"
                         for y, (f, c) in p.seed_infl_thresholds.items())
        thr_join = f"JOIN (VALUES {vals}) t(year, floor, cap) ON t.year = s.year"
        thr_where = ("AND coalesce(s.influentialcitationcount, 0) >= t.floor "
                     "AND (t.cap IS NULL OR coalesce(s.influentialcitationcount, 0) < t.cap)")
    infl = "AND coalesce(e.isinfluential, FALSE)" if p.influential_only else ""
    bg = f"AND NOT ({_INTENTS_SQL} = ['background'])" if p.drop_background_only else ""
    restrict = f"AND s.corpusid IN (SELECT corpusid FROM {seed_ids_table})" if seed_ids_table else ""
    return f"""
        SELECT s.corpusid AS seed_id, c.corpusid AS citer_id
        FROM read_parquet('{edges_dir}/bucket_*.parquet') e
        JOIN read_parquet('{seeds}') s ON s.corpusid = e.citedcorpusid
        JOIN read_parquet('{citers}') c ON c.corpusid = e.citingcorpusid
        {thr_join}
        WHERE {' AND '.join(year)}
          AND length(s.abstract) BETWEEN {p.abstract_min_chars} AND {p.abstract_max_chars}
          AND c.abstract IS NOT NULL
          AND length(c.abstract) BETWEEN {p.abstract_min_chars} AND {p.abstract_max_chars}
          AND c.year IS NOT NULL AND c.year >= s.year
          AND c.corpusid <> s.corpusid
          AND NOT coalesce(list_has_any(c.authors, s.authors), FALSE)
          AND coalesce(c.citationcount, 0) >= {p.min_citer_citations}
          {thr_where}
          {infl}
          {bg}
          {restrict}
    """


def _flat_intents(intents) -> list[str]:
    out = []
    for lst in intents or []:
        for x in lst or []:
            if x and x not in out:
                out.append(x)
    return out


def is_english(text: str) -> bool:
    """langdetect verdict, seeded so the benchmark cut is reproducible."""
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    try:
        return detect(text) == "en"
    except Exception:
        return False


def _quantiles(values: list[int]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(round(q * (len(v) - 1))))]  # noqa: E731
    return {"p10": pick(0.1), "p25": pick(0.25), "p50": pick(0.5), "p75": pick(0.75),
            "p90": pick(0.9), "max": v[-1], "mean": round(sum(v) / len(v), 2)}


def make_benchmark(con, seeds: Path, edges_dir: Path, citers: Path, out_jsonl: Path,
                   out_ids: Path, p: BenchmarkParams) -> dict:
    """Select seeds and citers deterministically; write JSONL + id manifest.

    Seeds and citers are ordered by ``stable_hash`` so the JSONL is reproducible
    and optional caps (``n_seeds``, ``citers_per_seed``) are hash prefixes.
    ``p.seed_infl_thresholds`` must already be resolved (see
    ``acl_influential_thresholds``) or be None for no seed bound.
    """
    t0 = time.time()
    pairs_sql = qualifying_pairs_sql(seeds, edges_dir, citers, p)
    con.execute(f"CREATE OR REPLACE TABLE qpairs AS {pairs_sql}")
    counts = con.execute("SELECT seed_id, count(*) FROM qpairs GROUP BY 1").fetchall()
    if p.english_only:
        # only citers of seeds that clear the floor need a verdict: removing pairs can drop a
        # seed below it but never lift one above it
        con.execute(f"""CREATE OR REPLACE TABLE lang_cand AS
            SELECT DISTINCT c.corpusid, c.abstract FROM qpairs q
            JOIN read_parquet('{citers}') c ON c.corpusid = q.citer_id
            WHERE q.seed_id IN (SELECT seed_id FROM qpairs GROUP BY 1
                                HAVING count(*) >= {p.min_citers})""")
        non_en = [cid for cid, text in con.execute("SELECT * FROM lang_cand").fetchall()
                  if not is_english(text)]
        con.register("non_en_arrow", pa.table({"corpusid": pa.array(non_en, pa.int64())}))
        n_before = con.execute("SELECT count(*) FROM qpairs").fetchone()[0]
        con.execute("DELETE FROM qpairs WHERE citer_id IN (SELECT corpusid FROM non_en_arrow)")
        n_after = con.execute("SELECT count(*) FROM qpairs").fetchone()[0]
        print(f"english_only: {len(non_en):,} non-English citers, {n_before - n_after:,} pairs "
              f"removed [{time.time() - t0:.0f}s]", flush=True)
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
               s.citationcount, s.influentialcitationcount, s.s2fieldsofstudy
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
    tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    with open(tmp, "w") as f:
        for (sid, acl, title, abstract, authors, year, venue, cc, icc, fos) in sorted(seed_rows):
            cids = sampled[sid]
            rec = {
                "seed_id": str(sid), "acl_id": acl,
                "seed_title": title, "seed_abstract": abstract,
                "seed_authors": authors or [], "seed_year": year, "seed_venue": venue,
                "seed_citationcount": cc, "seed_influentialcitationcount": icc,
                "seed_fields": fos or [],
                "n_citers_qualifying": int(n_qual[sid]),
                "citers": [citers_of[sid][c] for c in cids],
            }
            n_pairs += len(rec["citers"])
            f.write(json.dumps(rec) + "\n")
            manifest["seeds"][str(sid)] = {"citers": [str(c) for c in cids]}
    os.replace(tmp, out_jsonl)
    with open(out_ids, "w") as f:
        json.dump(manifest, f)
    stats = {
        "n_seeds_any_qualifying": len(counts), "n_seeds_eligible": len(eligible),
        "n_seeds": len(chosen), "n_pairs": n_pairs,
        "citers_per_seed": _quantiles([len(sampled[s]) for s in chosen]),
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
