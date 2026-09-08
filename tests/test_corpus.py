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


def test_qualifying_pairs_filters(tmp_path):
    """Author overlap, year order, abstract length and self-citation are excluded."""
    con = duckdb.connect()
    long = "x" * 500
    con.execute("CREATE TABLE s AS SELECT * FROM (VALUES "
                "(1, 2015, ['A. Smith','B. Jones'], ?), (2, 2015, ['C. Lee'], 'short')) "
                "t(corpusid, year, authors, abstract)", [long])
    con.execute("CREATE TABLE c AS SELECT * FROM (VALUES "
                "(10, 2016, ['Z. Zed'], ?, 5), "     # ok
                "(11, 2016, ['A. Smith'], ?, 5), "   # author overlap
                "(12, 2014, ['Y. Why'], ?, 5), "     # predates seed
                "(13, 2016, ['X. Ex'], 'tiny', 5), " # short abstract
                "(1,  2016, ['Q. Q'], ?, 5)) "       # self
                "t(corpusid, year, authors, abstract, citationcount)", [long] * 4)
    con.execute("CREATE TABLE e AS SELECT * FROM (VALUES (1,10),(1,11),(1,12),(1,13),(1,1),(2,10)) "
                "t(citedcorpusid, citingcorpusid)")
    for name in ("s", "c", "e"):
        con.execute(f"COPY {name} TO '{tmp_path}/{name}.parquet' (FORMAT PARQUET)")
    (tmp_path / "edges").mkdir()
    (tmp_path / "e.parquet").rename(tmp_path / "edges" / "bucket_00.parquet")
    p = corpus.BenchmarkParams(seed_year_min=2010, seed_year_max=2020, min_citer_citations=0)
    sql = corpus.qualifying_pairs_sql(tmp_path / "s.parquet", tmp_path / "edges",
                                      tmp_path / "c.parquet", p)
    assert con.execute(sql).fetchall() == [(1, 10)]


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
