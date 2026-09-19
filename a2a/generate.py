"""Generation conditions for the abstract-to-abstract task (vLLM).

Every condition produces ``N_GENERATIONS`` = 51 abstracts per seed (51 = 95th
percentile of citers per seed). Each abstract gives the background, objective
and method of a plausible future paper influenced by the seed, 100-150 words,
no results: the targets are citing abstracts with their result sentences
removed, and fabricated numbers buy embedding similarity for free.

Two kinds of condition:

* ``independent``: one prompt sampled 51 times; temperature is the only knob.
* ``verbalized``: Verbalized Sampling (Zhang et al. 2025): each call asks for
  ``per_call`` abstracts with their probabilities as JSON; 17 calls x 3.

A call can fail (reasoning never closed, JSON unparseable), so a seed short of
51 is topped up with further calls, up to ``max_rounds``.

A model runs data-parallel: one process per GPU, process i of n taking
``seeds[i::n]`` and writing its own shard files, so no file has two writers.
Outputs, all appended and fsync'd per seed so a killed run resumes cleanly:
``<out_dir>/<condition>/shard_<i>_of_<n>.jsonl``  ``{"seed_id", "condition", "model", "predictions"}``
``<out_dir>/<condition>/shard_<i>_of_<n>.jsonl.processed``  finished seed ids
``<out_dir>/<condition>/traces/shard_<i>_of_<n>.jsonl``  ``{"seed_id", "raw"}``: the
unstripped text of every call (reasoning trace, verbalized probabilities).
A seed listed in any shard's ``.processed`` is skipped, so a run can be resumed
with a different number of processes.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .llm import GENERATORS, REASONING_TOKENS, strip_reasoning

N_GENERATIONS = 51

# Each bullet answers something measured on the 50-seed pilot against the real (filtered)
# citing abstracts: Gemma opened 99% of abstracts with "While ..." (real: 3%) and wrote 74%
# in proposal tense (real: 7%); gpt-oss said "the seed" in 13% and put a number in 32%;
# both claimed improvements about twice as often as the targets, which have had their
# result sentences removed.
_TASK = """\
You are given the title, year and abstract of a published paper (the SEED). \
{ask} that was influenced by the seed and cites it, published in the years \
after it. Real citing papers relate to a seed in many ways: they extend its \
method, use it as one component of a different system, apply it to another \
task, language or domain, build on its data or evaluation, or address one of \
its limitations.

{each}:
- reads as the abstract of a finished, published paper: present tense ("We \
propose ...", "This paper presents ..."), never a plan ("we will", "aims to", \
"seeks to");
- gives the background, the objective and the method of that paper and \
nothing else, in one paragraph of 100-150 words (about 120);
- opens the way real abstracts do, with the problem, the task or the \
contribution itself, not with a "While ..." critique of earlier work;
- is concrete: it names the specific task, data and techniques, as a real \
abstract does;
- stands on its own: it may name methods, tasks or datasets that come from the \
seed, but never refers to "the seed", "the original paper" or its title;
- contains NO results: no findings, no claims of improvement or effectiveness \
("outperforms", "improves", "achieves", "demonstrates"), no numbers or \
invented quantities of any kind, and no closing sentence about conclusions, \
implications, contributions, or code and data availability."""

INDEPENDENT_SYSTEM = _TASK.format(
    ask="Write the abstract of ONE plausible research paper",
    each="The abstract") + "\n\nReturn ONLY the abstract text."

VERBALIZED_SYSTEM = _TASK.format(
    ask="Write the abstracts of {k} plausible research papers, each one a paper",
    each="Each abstract") + """

Sample the {k} papers at random from the full distribution of papers the seed \
could influence, and give each one's probability under that distribution.
Return STRICT JSON: [{{"abstract": "...", "probability": 0.0}}, ...] with \
exactly {k} items, no markdown fences."""


@dataclass(frozen=True)
class Condition:
    generator: str                   # key of llm.GENERATORS
    kind: str = "independent"        # "independent" | "verbalized"
    temperature: float | None = None  # None = the generator's recommended value
    per_call: int = 1
    max_tokens: int = 400
    reasoning_effort: str | None = None  # gpt-oss only; None = the generator's setting (low)

    @property
    def system(self) -> str:
        if self.kind == "independent":
            return INDEPENDENT_SYSTEM
        return VERBALIZED_SYSTEM.format(k=self.per_call)


# top_p and top_k are never overridden: every condition, of both generators, samples with
# llm.TOP_P and llm.TOP_K.
CONDITIONS: dict[str, Condition] = {
    "gemma_t10": Condition("gemma", temperature=1.0),          # baseline
    "gemma_t05": Condition("gemma", temperature=0.5),
    "gemma_t15": Condition("gemma", temperature=1.5),
    "gemma_t10_vs": Condition("gemma", "verbalized", temperature=1.0, per_call=3,
                              max_tokens=1200),
    "gptoss": Condition("gpt-oss"),                            # reasoning effort low
    "gptoss_high": Condition("gpt-oss", reasoning_effort="high"),
}


def user_message(seed: dict, max_chars: int = 3000) -> str:
    return (f"SEED PAPER TITLE: {seed.get('seed_title') or ''}\n"
            f"SEED PAPER YEAR: {seed.get('seed_year') or 'unknown'}\n"
            f"SEED PAPER ABSTRACT: {seed['seed_abstract'][:max_chars]}")


_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)


def _scan_objects(text: str) -> list:
    """Every parseable JSON object in ``text`` (models sometimes emit objects
    separated by blank lines instead of one array)."""
    dec, out, i = json.JSONDecoder(), [], 0
    while (j := text.find("{", i)) >= 0:
        try:
            obj, end = dec.raw_decode(text, j)
            out.append(obj)
            i = end
        except json.JSONDecodeError:
            i = j + 1
    return out


def parse_list(text: str) -> list[str]:
    """Extract abstracts from a JSON list; tolerate stray text around it and
    fall back to scanning for bare JSON objects."""
    items = None
    m = _LIST_RE.search(text)
    if m:
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            items = None
    if not isinstance(items, list):
        items = _scan_objects(text)
    return [it["abstract"].strip() for it in items
            if isinstance(it, dict) and isinstance(it.get("abstract"), str)
            and it["abstract"].strip()]


def _clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\**title\**\s*:[^\n]*\n+", "", text, flags=re.I)  # drop a leading title line
    text = re.sub(r"^\*\*[^\n]*\*\*[ \t]*\n\s*", "", text)  # all-bold first line = header (gpt-oss)
    # label with a separator ("Abstract:") or on a line of its own ("**Abstract**\n", gpt-oss)
    text = re.sub(r"^\**\s*(abstract|proposal)\s*\**[ \t]*([:\-–—]\s*\**\s*|\n\s*)", "", text, flags=re.I)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # inline markdown bold (gpt-oss)
    return text.strip().strip('"').strip()


def extract(raw: str, kind: str) -> list[str]:
    """Abstracts in one call's raw output, reasoning stripped."""
    text = strip_reasoning(raw)
    items = [text] if kind == "independent" else parse_list(text)
    return [c for c in map(_clean, items) if c]


def _load_done(cond_dir: Path) -> set[str]:
    """Seed ids finished by any shard of this condition."""
    done: set[str] = set()
    for p in cond_dir.glob("*.processed"):
        with open(p) as f:
            done |= {line.strip() for line in f if line.strip()}
    return done


def _append(f, line: str) -> None:
    f.write(line + "\n")
    f.flush()
    os.fsync(f.fileno())


def generate_condition(llm, name: str, seeds: list[dict], out_dir: Path,
                       shard: tuple[int, int] = (0, 1), n: int = N_GENERATIONS,
                       batch_size: int | None = None, max_rounds: int = 4) -> None:
    """Run one condition over ``seeds[i::n_shards]``; ``seeds`` is the full list."""
    from vllm import SamplingParams

    cond = CONDITIONS[name]
    gen = GENERATORS[cond.generator]
    batch_size = batch_size or gen.batch_size
    template_kwargs, reasoning_tokens = gen.chat_template_kwargs, gen.reasoning_tokens
    if cond.reasoning_effort:
        template_kwargs = {**template_kwargs, "reasoning_effort": cond.reasoning_effort}
        reasoning_tokens = REASONING_TOKENS[cond.reasoning_effort]
    stem = f"shard_{shard[0]}_of_{shard[1]}.jsonl"
    out_path = out_dir / name / stem
    proc_path = out_dir / name / f"{stem}.processed"
    trace_path = out_dir / name / "traces" / stem
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_dir / name)
    todo = [s for s in seeds[shard[0]::shard[1]] if s["seed_id"] not in done]
    print(f"[{name}] {len(todo)} seeds to run ({len(done)} done)", flush=True)
    if not todo:
        return

    def params(n_calls: int):
        return SamplingParams(
            temperature=gen.temperature if cond.temperature is None else cond.temperature,
            top_p=gen.top_p, top_k=gen.top_k,
            max_tokens=cond.max_tokens + reasoning_tokens, n=n_calls)

    t0, n_done, n_tok, n_short = time.time(), 0, 0, 0
    with open(out_path, "a") as out_f, open(proc_path, "a") as proc_f, \
            open(trace_path, "a") as trace_f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            preds = {s["seed_id"]: [] for s in batch}
            raws = {s["seed_id"]: [] for s in batch}
            for _ in range(max_rounds):
                short = [s for s in batch if len(preds[s["seed_id"]]) < n]
                if not short:
                    break
                msgs = [[{"role": "system", "content": cond.system},
                         {"role": "user", "content": user_message(s)}] for s in short]
                sps = [params(-(-(n - len(preds[s["seed_id"]])) // cond.per_call)) for s in short]
                outs = llm.chat(msgs, sps, chat_template_kwargs=template_kwargs,
                                use_tqdm=False)
                for s, out in zip(short, outs):
                    for o in out.outputs:
                        n_tok += len(o.token_ids)
                        raws[s["seed_id"]].append(o.text)
                        preds[s["seed_id"]].extend(extract(o.text, cond.kind))
            for s in batch:
                sid = s["seed_id"]
                if len(preds[sid]) < n:
                    n_short += 1
                    print(f"[{name}] {sid}: {len(preds[sid])}/{n} usable after "
                          f"{len(raws[sid])} calls", flush=True)
                _append(trace_f, json.dumps({"seed_id": sid, "raw": raws[sid]}))
                _append(out_f, json.dumps({"seed_id": sid, "condition": name, "model": gen.hf_id,
                                           "predictions": preds[sid][:n]}))
                _append(proc_f, sid)
            n_done += len(batch)
            rate = n_done / (time.time() - t0 + 1e-9)
            print(f"[{name}] {n_done}/{len(todo)} seeds ({rate * 3600:.0f} seeds/h)", flush=True)
    dt = time.time() - t0
    print(f"[{name}] finished {n_done} seeds in {dt:.0f}s ({dt / n_done:.1f} s/seed, "
          f"{n_tok} output tokens, {n_tok / dt:.0f} tok/s, {n_short} seeds short of {n})",
          flush=True)


def load_generations(gen_dir: Path) -> dict[str, dict[str, list[str]]]:
    """{condition: {seed_id: predictions}} from every shard of every ``<condition>/`` in
    ``gen_dir``, conditions in ``CONDITIONS`` order.

    Last write wins per seed, so a resumed run that re-emitted a seed is safe.
    """
    out: dict[str, dict[str, list[str]]] = {}
    dirs = sorted((d for d in Path(gen_dir).iterdir() if d.is_dir()),
                  key=lambda d: (list(CONDITIONS).index(d.name) if d.name in CONDITIONS
                                 else len(CONDITIONS), d.name))
    for d in dirs:
        recs: dict[str, list[str]] = {}
        for p in sorted(d.glob("shard_*.jsonl")):
            with open(p) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        recs[r["seed_id"]] = [t for t in r["predictions"] if t]
        if recs:
            out[d.name] = recs
    return out
