import json

import duckdb
import pytest

from a2a import corpus
from a2a.generate import parse_list
from a2a.llm import strip_reasoning


def test_dev_pool_is_stable_and_about_ten_percent():
    dev = sum(corpus.in_dev_pool(i) for i in range(20_000))
    assert 1800 < dev < 2200
    assert corpus.in_dev_pool(12345) == corpus.in_dev_pool("12345")


def test_stable_hash_orders_deterministically():
    a = sorted(range(50), key=lambda s: corpus.stable_hash("seed", s))
    b = sorted(range(50), key=lambda s: corpus.stable_hash("seed", s))
    assert a == b and a != list(range(50))


def test_flat_intents_dedups_and_flattens():
    assert corpus._flat_intents([["background"], ["methodology", "background"], None]) == \
        ["background", "methodology"]


def _toy_corpus(tmp_path, con):
    """Two seeds, five citers, six edges; seed 1 has one qualifying citer (10)."""
    long = "x" * 500
    con.execute("CREATE TABLE s AS SELECT * FROM (VALUES "
                "(1, 'P15-1', 't1', ?, ['A. Smith','B. Jones'], 2015, 'ACL', 40, ['Computer Science']), "
                "(2, 'P15-2', 't2', 'short', ['C. Lee'], 2015, 'ACL', 40, NULL), "
                "(3, 'P15-3', 't3', ?, ['D. Low'], 2015, 'ACL', 10, NULL), "     # below floor
                "(4, 'P15-4', 't4', ?, ['E. Null'], NULL, 'ACL', 40, NULL), "    # no year
                "(5, 'P15-5', 't5', ?, ['F. Famous'], 2015, 'ACL', 9000, NULL)) " # above cap
                "t(corpusid, acl_id, title, abstract, authors, year, venue, citationcount, "
                "s2fieldsofstudy)", [long] * 4)
    con.execute("CREATE TABLE c AS SELECT * FROM (VALUES "
                "(10, 2016, ['Z. Zed'], ?, 5), "     # ok
                "(11, 2016, ['A. Smith'], ?, 5), "   # author overlap with seed 1
                "(12, 2014, ['Y. Why'], ?, 5), "     # predates seed
                "(13, 2016, ['X. Ex'], 'tiny', 5), " # short abstract
                "(1,  2016, ['Q. Q'], ?, 5)) "       # self
                "t(corpusid, year, authors, abstract, citationcount)", [long] * 4)
    con.execute("CREATE TABLE c2 AS SELECT corpusid, NULL::VARCHAR acl_id, 'ct' title, abstract, "
                "authors, year, 'V' venue, citationcount, NULL::VARCHAR[] s2fieldsofstudy FROM c")
    con.execute("CREATE TABLE e AS SELECT citedcorpusid, citingcorpusid, FALSE isinfluential, "
                "NULL::VARCHAR[] contexts, NULL::VARCHAR[][] intents FROM (VALUES "
                "(1,10),(1,11),(1,12),(1,13),(1,1),(2,10),(3,10),(4,10),(5,10)) "
                "t(citedcorpusid, citingcorpusid)")
    con.execute(f"COPY s TO '{tmp_path}/s.parquet' (FORMAT PARQUET)")
    con.execute(f"COPY c2 TO '{tmp_path}/c.parquet' (FORMAT PARQUET)")
    (tmp_path / "edges").mkdir()
    con.execute(f"COPY e TO '{tmp_path}/edges/bucket_00.parquet' (FORMAT PARQUET)")
    return tmp_path / "s.parquet", tmp_path / "edges", tmp_path / "c.parquet"


def test_qualifying_pairs_filters(tmp_path):
    """Author overlap, year order, abstract length, self-citation, citation floor / cap
    and NULL seed year are excluded; every seed bound is optional."""
    con = duckdb.connect()
    s, e, c = _toy_corpus(tmp_path, con)

    def pairs(**kw):
        p = corpus.BenchmarkParams(**kw)
        return sorted(con.execute(corpus.qualifying_pairs_sql(s, e, c, p)).fetchall())

    assert pairs(min_seed_citations=35, max_seed_citations=500) == [(1, 10)]
    assert pairs(min_seed_citations=35) == [(1, 10), (5, 10)]                 # cap off
    assert pairs(max_seed_citations=500) == [(1, 10), (3, 10)]                # floor off
    assert pairs(min_seed_citations=None, max_seed_citations=None, seed_year_min=2016) == []


def test_make_benchmark_keeps_every_eligible_seed_and_citer(tmp_path):
    con = duckdb.connect()
    s, e, c = _toy_corpus(tmp_path, con)
    p = corpus.BenchmarkParams(max_seed_citations=500)
    out = tmp_path / "bench.jsonl"
    stats = corpus.make_benchmark(con, s, e, c, out, tmp_path / "ids.json", p)
    recs = corpus.load_jsonl(out)
    assert {r["seed_id"] for r in recs} == {"1", "3"}
    assert all(len(r["citers"]) == r["n_citers_qualifying"] == 1 for r in recs)
    assert stats["n_seeds"] == 2 and stats["n_pairs"] == 2
    assert stats["params"]["n_seeds"] is None and stats["params"]["citers_per_seed"] is None
    manifest = json.loads((tmp_path / "ids.json").read_text())
    assert manifest["seeds"]["1"]["citers"] == ["10"]
    assert manifest["seeds"]["1"]["split"] in ("dev", "test")
    # a cap of one seed is a hash prefix of the uncapped order
    stats1 = corpus.make_benchmark(con, s, e, c, out, tmp_path / "ids1.json",
                                   corpus.BenchmarkParams(max_seed_citations=500, n_seeds=1))
    first = sorted([1, 3], key=lambda x: corpus.stable_hash("seed", x))[0]
    assert stats1["n_seeds"] == 1 and corpus.load_jsonl(out)[0]["seed_id"] == str(first)


def test_acl_citation_quantile_is_over_all_acl_papers(tmp_path):
    con = duckdb.connect()
    (tmp_path / "papers" / "bucket=0").mkdir(parents=True)
    # 10 ACL papers with citation counts 1..10 and one non-ACL paper with 1000 citations
    con.execute(f"""COPY (SELECT i AS corpusid, 'P' || i AS acl_id, i AS citationcount
                    FROM range(1, 11) t(i) UNION ALL SELECT 99, NULL, 1000)
                    TO '{tmp_path}/papers/bucket=0/x.parquet' (FORMAT PARQUET)""")
    assert corpus.acl_citation_quantile(con, 0.8, tmp_path) == 8    # top-20% floor
    assert corpus.acl_citation_quantile(con, 0.5, tmp_path) == 5
    assert corpus.acl_citation_quantile(con, 0.99, tmp_path) == 10  # top-1% cap
    with pytest.raises(ValueError):
        corpus.acl_citation_quantile(con, 1.0, tmp_path)


def test_parse_list_tolerates_wrapping_text():
    text = 'Sure:\n[{"abstract": " one "}, {"abstract": "two"}, {"x": 1}]\nthanks'
    assert parse_list(text) == ["one", "two"]
    assert parse_list("no json here") == []


@pytest.mark.parametrize("raw,expected", [
    ("analysis thinking... assistantfinal The answer.", "The answer."),
    ("<think>hmm</think>\nThe answer.", "The answer."),
    ("<think>never closed", ""),
    ("plain text", "plain text"),
])
def test_strip_reasoning(raw, expected):
    assert strip_reasoning(raw) == expected


def test_jsonl_roundtrip(tmp_path):
    recs = [{"a": 1}, {"b": [1, 2]}]
    corpus.write_jsonl(recs, tmp_path / "x.jsonl")
    assert corpus.load_jsonl(tmp_path / "x.jsonl") == recs
    assert json.loads((tmp_path / "x.jsonl").read_text().splitlines()[0]) == {"a": 1}


def test_parse_list_falls_back_to_bare_objects():
    text = ('{"abstract": "first one", "year": 2025}\n\n'
            '{"title": "t", "abstract": "second one"}\nnot json {broken')
    assert parse_list(text) == ["first one", "second one"]


def test_clean_drops_title_line_and_label():
    from a2a.generate import _clean
    assert _clean("Title: Some paper\n\nAbstract: The body.") == "The body."
    assert _clean("**Abstract:** The body.") == "The body."
