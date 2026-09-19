import pytest

from a2a.generate import CONDITIONS, N_GENERATIONS, _clean, extract, parse_list
from a2a.llm import GENERATORS, strip_reasoning
from a2a.rerank import target_citers


def test_conditions_are_the_planned_run():
    assert set(CONDITIONS) == {"gemma_t10", "gemma_t05", "gemma_t15", "gemma_t10_vs", "gptoss"}
    assert {c.generator for c in CONDITIONS.values()} <= set(GENERATORS)
    assert [CONDITIONS[c].temperature for c in ("gemma_t10", "gemma_t05", "gemma_t15")] == [1.0, 0.5, 1.5]
    vs = CONDITIONS["gemma_t10_vs"]
    assert (vs.kind, vs.per_call, vs.temperature) == ("verbalized", 3, 1.0)
    assert N_GENERATIONS % vs.per_call == 0 and N_GENERATIONS // vs.per_call == 17
    assert '"probability"' in vs.system and "exactly 3 items" in vs.system
    assert CONDITIONS["gptoss"].temperature is None      # the generator's recommended settings


def test_parse_list_tolerates_wrapping_text():
    text = 'Sure:\n[{"abstract": " one "}, {"abstract": "two"}, {"x": 1}]\nthanks'
    assert parse_list(text) == ["one", "two"]
    assert parse_list("no json here") == []


def test_parse_list_falls_back_to_bare_objects():
    text = ('{"abstract": "first one", "year": 2025}\n\n'
            '{"title": "t", "abstract": "second one"}\nnot json {broken')
    assert parse_list(text) == ["first one", "second one"]


@pytest.mark.parametrize("raw,expected", [
    ("analysis thinking... assistantfinal The answer.", "The answer."),
    ("<think>hmm</think>\nThe answer.", "The answer."),
    ("<think>never closed", ""),
    ("plain text", "plain text"),
])
def test_strip_reasoning(raw, expected):
    assert strip_reasoning(raw) == expected


def test_clean_drops_title_line_and_label():
    assert _clean("Title: Some paper\n\nAbstract: The body.") == "The body."
    assert _clean("**Abstract:** The body.") == "The body."
    assert _clean("**Abstract**  \nThe body.") == "The body."
    assert _clean("**Abstract – Proposal for X**  \nThe body.") == "The body."
    assert _clean("**Abstract** – We propose **a thing**.") == "We propose a thing."
    assert _clean("Abstract meaning representation is hard.") == "Abstract meaning representation is hard."


def test_extract_strips_the_trace_and_keeps_only_abstracts():
    assert extract("analysis plan it assistantfinal **Abstract:** We propose X.", "independent") == ["We propose X."]
    assert extract("analysis never reached the answer", "independent") == []
    raw = '[{"abstract": "We study A.", "probability": 0.2}, {"abstract": "", "probability": 0.1}]'
    assert extract(raw, "verbalized") == ["We study A."]


def _seed(sid, citer_ids):
    return {"seed_id": sid, "citers": [{"id": c} for c in citer_ids]}


def test_shuffled_targets_are_a_derangement_without_shared_citers():
    seeds = [_seed("s1", ["a", "b"]), _seed("s2", ["b", "c"]), _seed("s3", ["d"]), _seed("s4", ["a"])]
    own = target_citers(seeds, "own")
    shuf = target_citers(seeds, "shuffled")
    assert own["s2"] == seeds[1]["citers"]
    donors = {}
    for s in seeds:
        got = {c["id"] for c in shuf[s["seed_id"]]}
        assert not got & {c["id"] for c in s["citers"]}          # never one of its own citers
        donor = [d for d in seeds if got <= {c["id"] for c in d["citers"]} and d is not s]
        assert donor
        donors[s["seed_id"]] = donor
    assert shuf == target_citers(list(reversed(seeds)), "shuffled")  # order-independent
