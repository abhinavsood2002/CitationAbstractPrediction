import json

import duckdb
import pytest

from a2a import corpus


def test_stable_hash_orders_deterministically():
    a = sorted(range(50), key=lambda s: corpus.stable_hash("seed", s))
    b = sorted(range(50), key=lambda s: corpus.stable_hash("seed", s))
    assert a == b and a != list(range(50))


def test_flat_intents_dedups_and_flattens():
    assert corpus._flat_intents([["background"], ["methodology", "background"], None]) == \
        ["background", "methodology"]


THR = {2015: [5, 60]}   # within-year band for the toy corpus: 5 <= influential < 60


def _toy_corpus(tmp_path, con):
    """Five seeds (2015), six citers, edges with varied influence / intent labels.

    Seed 1 is in band and has three good citers (10, 14, 15) plus one non-influential
    (16), one background-only (17), one author-overlap (11), one predating (12), one
    short-abstract (13) and a self edge. Seed 2 has a short abstract, 3 is below the
    influential floor, 4 has no year, 5 is at/above the cap.
    """
    long = "x" * 500
    con.execute("CREATE TABLE s AS SELECT * FROM (VALUES "
                "(1, 'P15-1', 't1', ?, ['A. Smith','B. Jones'], 2015, 'ACL', 40, 9, ['Computer Science']), "
                "(2, 'P15-2', 't2', 'short', ['C. Lee'], 2015, 'ACL', 40, 9, NULL), "
                "(3, 'P15-3', 't3', ?, ['D. Low'], 2015, 'ACL', 10, 4, NULL), "      # below floor
                "(4, 'P15-4', 't4', ?, ['E. Null'], NULL, 'ACL', 40, 9, NULL), "    # no year
                "(5, 'P15-5', 't5', ?, ['F. Famous'], 2015, 'ACL', 9000, 60, NULL)) " # at cap
                "t(corpusid, acl_id, title, abstract, authors, year, venue, citationcount, "
                "influentialcitationcount, s2fieldsofstudy)", [long] * 4)
    con.execute("CREATE TABLE c AS SELECT * FROM (VALUES "
                "(10, 2016, ['Z. Zed'], ?, 5), "     # ok
                "(11, 2016, ['A. Smith'], ?, 5), "   # author overlap with seed 1
                "(12, 2014, ['Y. Why'], ?, 5), "     # predates seed
                "(13, 2016, ['X. Ex'], 'tiny', 5), " # short abstract
                "(14, 2017, ['W. Wu'], ?, 5), "      # ok, unlabelled intents
                "(15, 2017, ['V. Vee'], ?, 5), "     # ok, background + methodology
                "(16, 2017, ['U. Uh'], ?, 5), "      # not influential
                "(17, 2017, ['T. Tee'], ?, 5), "     # background only
                "(1,  2016, ['Q. Q'], ?, 5)) "       # self
                "t(corpusid, year, authors, abstract, citationcount)", [long] * 8)
    con.execute("CREATE TABLE c2 AS SELECT corpusid, NULL::VARCHAR acl_id, 'ct' title, abstract, "
                "authors, year, 'V' venue, citationcount, NULL::VARCHAR[] s2fieldsofstudy FROM c")
    con.execute("CREATE TABLE e AS SELECT * FROM (VALUES "
                "(1, 10, TRUE, ['ctx'], [['methodology']]), "
                "(1, 11, TRUE, NULL, NULL), (1, 12, TRUE, NULL, NULL), (1, 13, TRUE, NULL, NULL), "
                "(1, 1, TRUE, NULL, NULL), "
                "(1, 14, TRUE, ['ctx'], NULL), "
                "(1, 15, TRUE, ['a', 'b'], [['background'], ['methodology', NULL]]), "
                "(1, 16, FALSE, ['ctx'], [['methodology']]), "
                "(1, 17, TRUE, ['a', 'b'], [['background'], ['background'], NULL]), "
                "(2, 10, TRUE, NULL, NULL), (3, 10, TRUE, NULL, NULL), (4, 10, TRUE, NULL, NULL), "
                "(5, 10, TRUE, NULL, NULL)) "
                "t(citedcorpusid, citingcorpusid, isinfluential, contexts, intents)")
    con.execute("CREATE TABLE e2 AS SELECT citedcorpusid, citingcorpusid, isinfluential, "
                "contexts::VARCHAR[] contexts, intents::VARCHAR[][] intents FROM e")
    con.execute(f"COPY s TO '{tmp_path}/s.parquet' (FORMAT PARQUET)")
    con.execute(f"COPY c2 TO '{tmp_path}/c.parquet' (FORMAT PARQUET)")
    (tmp_path / "edges").mkdir()
    con.execute(f"COPY e2 TO '{tmp_path}/edges/bucket_00.parquet' (FORMAT PARQUET)")
    return tmp_path / "s.parquet", tmp_path / "edges", tmp_path / "c.parquet"


def test_qualifying_pairs_filters(tmp_path):
    """Author overlap, year order, abstract length, self-citation, NULL seed year, the
    within-year influential band, the influential flag and background-only intents are
    excluded; every one of the last three is switchable."""
    con = duckdb.connect()
    s, e, c = _toy_corpus(tmp_path, con)

    def pairs(**kw):
        p = corpus.BenchmarkParams(**{"seed_infl_thresholds": THR, **kw})
        return sorted(con.execute(corpus.qualifying_pairs_sql(s, e, c, p)).fetchall())

    assert pairs() == [(1, 10), (1, 14), (1, 15)]
    assert pairs(drop_background_only=False) == [(1, 10), (1, 14), (1, 15), (1, 17)]
    assert pairs(influential_only=False) == [(1, 10), (1, 14), (1, 15), (1, 16)]
    assert pairs(seed_infl_thresholds={2015: [5, None]}) == \
        [(1, 10), (1, 14), (1, 15), (5, 10)]                                   # cap off
    assert pairs(seed_infl_thresholds={2015: [0, 60]}) == \
        [(1, 10), (1, 14), (1, 15), (3, 10)]                                   # floor off
    assert pairs(seed_infl_thresholds=None) == \
        [(1, 10), (1, 14), (1, 15), (3, 10), (5, 10)]                          # no seed bound
    assert pairs(seed_infl_thresholds={2016: [0, None]}) == []                 # no row for 2015
    assert pairs(seed_year_min=2016) == []
    with pytest.raises(ValueError):
        pairs(seed_infl_thresholds={})


def test_make_benchmark_keeps_every_eligible_seed_and_citer(tmp_path):
    con = duckdb.connect()
    s, e, c = _toy_corpus(tmp_path, con)
    p = corpus.BenchmarkParams(seed_infl_thresholds={2015: [0, 60]}, min_citers=1,
                               english_only=False)   # toy abstracts are "xxx..."
    out = tmp_path / "bench.jsonl"
    stats = corpus.make_benchmark(con, s, e, c, out, tmp_path / "ids.json", p)
    recs = {r["seed_id"]: r for r in corpus.load_jsonl(out)}
    assert set(recs) == {"1", "3"}
    assert all(len(r["citers"]) == r["n_citers_qualifying"] for r in recs.values())
    assert sorted(c["id"] for c in recs["1"]["citers"]) == ["10", "14", "15"]
    assert recs["1"]["seed_influentialcitationcount"] == 9 and "split" not in recs["1"]
    by_id = {c["id"]: c for c in recs["1"]["citers"]}
    assert by_id["15"]["intents"] == ["background", "methodology"] and by_id["14"]["intents"] == []
    assert stats["n_seeds"] == 2 and stats["n_pairs"] == 4
    assert stats["citers_per_seed"]["max"] == 3 and stats["citers_per_seed"]["p50"] in (1, 3)
    assert stats["params"]["n_seeds"] is None and stats["params"]["citers_per_seed"] is None
    manifest = json.loads((tmp_path / "ids.json").read_text())
    assert manifest["seeds"]["3"] == {"citers": ["10"]}
    assert manifest["params"]["seed_infl_thresholds"] == {"2015": [0, 60]}
    # the citer floor drops seed 3; a cap of one seed is a hash prefix of the uncapped order
    assert corpus.make_benchmark(con, s, e, c, out, tmp_path / "ids2.json",
                                 corpus.BenchmarkParams(seed_infl_thresholds={2015: [0, 60]},
                                                        min_citers=2, english_only=False))["n_seeds"] == 1
    stats1 = corpus.make_benchmark(con, s, e, c, out, tmp_path / "ids1.json",
                                   corpus.BenchmarkParams(seed_infl_thresholds={2015: [0, 60]},
                                                          min_citers=1, n_seeds=1, english_only=False))
    first = sorted([1, 3], key=lambda x: corpus.stable_hash("seed", x))[0]
    assert stats1["n_seeds"] == 1 and corpus.load_jsonl(out)[0]["seed_id"] == str(first)


def test_english_only_drops_non_english_citers(tmp_path):
    """A seed whose only citer abstract is not English loses that citer (and the seed)."""
    con = duckdb.connect()
    s, e, c = _toy_corpus(tmp_path, con)
    en = ("We study whether neural language models can anticipate the papers that will cite "
          "a given abstract, and propose a coverage metric over real citing abstracts. ") * 4
    ja = "本研究では、引用予測のためのニューラル言語モデルの能力を検証し、被覆率に基づく評価指標を提案する。" * 6
    con.execute(f"CREATE TABLE cc AS SELECT * FROM read_parquet('{c}')")
    con.execute("UPDATE cc SET abstract = ? WHERE corpusid = 10", [ja])
    con.execute("UPDATE cc SET abstract = ? WHERE corpusid IN (14, 15)", [en])
    con.execute(f"COPY cc TO '{c}' (FORMAT PARQUET)")
    assert corpus.is_english(en) and not corpus.is_english(ja)
    p = corpus.BenchmarkParams(seed_infl_thresholds={2015: [0, 60]}, min_citers=1)
    stats = corpus.make_benchmark(con, s, e, c, tmp_path / "b.jsonl", tmp_path / "i.json", p)
    recs = {r["seed_id"]: r for r in corpus.load_jsonl(tmp_path / "b.jsonl")}
    assert set(recs) == {"1"} and sorted(x["id"] for x in recs["1"]["citers"]) == ["14", "15"]
    assert recs["1"]["n_citers_qualifying"] == 2 and stats["n_pairs"] == 2


def test_acl_influential_thresholds_are_per_year_over_all_acl_papers(tmp_path):
    con = duckdb.connect()
    (tmp_path / "papers" / "bucket=0").mkdir(parents=True)
    # 2015: 10 ACL papers with influential counts 1..10; 2016: 20 papers with 1..20;
    # one non-ACL paper with 1000 must not count
    con.execute(f"""COPY (
        SELECT i AS corpusid, 'P' || i AS acl_id, 2015 AS year, i AS influentialcitationcount
        FROM range(1, 11) t(i)
        UNION ALL
        SELECT 100 + i, 'Q' || i, 2016, i FROM range(1, 21) t(i)
        UNION ALL SELECT 999, NULL, 2015, 1000
        UNION ALL SELECT 998, 'X', NULL, 7)
        TO '{tmp_path}/papers/bucket=0/x.parquet' (FORMAT PARQUET)""")
    thr = corpus.acl_influential_thresholds(con, 0.2, 0.1, tmp_path)
    assert thr == {2015: [8, 9], 2016: [16, 18]}        # top 20% floor, top 10% cap, per year
    assert corpus.acl_influential_thresholds(con, 0.2, 0, tmp_path)[2015] == [8, None]
    assert corpus.acl_influential_thresholds(con, 0, 0.1, tmp_path)[2016] == [0, 18]
    with pytest.raises(ValueError):
        corpus.acl_influential_thresholds(con, 0.1, 0.2, tmp_path)   # cap above floor
    with pytest.raises(ValueError):
        corpus.acl_influential_thresholds(con, 1.0, 0.1, tmp_path)


def test_jsonl_roundtrip(tmp_path):
    recs = [{"a": 1}, {"b": [1, 2]}]
    corpus.write_jsonl(recs, tmp_path / "x.jsonl")
    assert corpus.load_jsonl(tmp_path / "x.jsonl") == recs
    assert json.loads((tmp_path / "x.jsonl").read_text().splitlines()[0]) == {"a": 1}
