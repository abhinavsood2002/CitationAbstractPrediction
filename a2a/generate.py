"""Generation arms for the abstract-to-abstract task (vLLM).

Given a seed abstract, an arm produces ``n`` candidate abstracts of future
papers that would cite the seed. Two kinds of arm:

* ``independent``: one chat call sampled ``n`` times (temperature is the only
  knob). The prompt asks for a *proposal-style* abstract with no invented
  numerical results: fabricated numbers buy embedding similarity for free
  (a leakage artefact documented in earlier work) and a paper that has not
  been written has no results yet.
* ``list``: the model is asked for ``per_call`` *substantially different*
  proposals in one call (verbalized sampling, Zhang et al. 2025), repeated
  ``ceil(n / per_call)`` times. This is the explicit-diversity arm.

Outputs are appended to ``<out_dir>/<arm>.jsonl`` as
``{"seed_id", "arm", "model", "predictions": [str]}`` with a ``.processed``
sidecar listing finished seed ids; both are fsync'd per seed so a killed run
resumes without loss or duplication.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .llm import strip_reasoning

N_DEFAULT = 50

PROPOSAL_SYSTEM = """\
You write research proposals. Given the abstract of a published paper (the \
SEED), write the abstract of ONE plausible FUTURE research paper that would \
cite the seed. Write it as a PROPOSAL-style abstract: motivation, gap, \
proposed approach, and what will be evaluated. Do NOT invent numerical \
results, percentages, or completed findings — the paper has not been done \
yet. One paragraph, 120-220 words. Return ONLY the abstract text."""

DIVERSE_SYSTEM = """\
You write research proposals. Given the abstract of a published paper (the \
SEED), generate {k} SUBSTANTIALLY DIFFERENT plausible future research papers \
that would cite the seed — different subfields, methods, and applications, \
including papers from OTHER disciplines that might use or build on the seed. \
For each, give a proposal-style abstract (100-180 words; motivation, gap, \
approach; NO invented numerical results).
Return STRICT JSON: [{{"abstract": "..."}}, ...] with exactly {k} items, no \
markdown fences."""


@dataclass(frozen=True)
class Arm:
    kind: str            # "independent" | "list"
    system: str
    temperature: float
    top_p: float = 0.95
    per_call: int = 1    # list arms: proposals per call
    max_tokens: int = 512


ARMS: dict[str, Arm] = {
    "proposal_t07": Arm("independent", PROPOSAL_SYSTEM, 0.7),
    "proposal_t10": Arm("independent", PROPOSAL_SYSTEM, 1.0),
    "proposal_t13": Arm("independent", PROPOSAL_SYSTEM, 1.3),
    "diverse_list_t10": Arm("list", DIVERSE_SYSTEM.format(k=10), 1.0, per_call=10,
                            max_tokens=4096),
}


def user_message(seed: dict, max_chars: int = 3000) -> str:
    return (f"SEED PAPER TITLE: {seed.get('seed_title') or ''}\n"
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
    text = strip_reasoning(text).strip()
    text = re.sub(r"^\**title\**\s*:[^\n]*\n+", "", text, flags=re.I)  # drop a leading title line
    text = re.sub(r"^\**\s*(abstract|proposal)\s*\**\s*[:\-]\s*\**\s*", "", text, flags=re.I)
    return text.strip().strip('"').strip()


def _load_done(proc_path: Path) -> set[str]:
    if not proc_path.exists():
        return set()
    with open(proc_path) as f:
        return {line.strip() for line in f if line.strip()}


def generate_arm(llm, arm_name: str, seeds: list[dict], out_dir: Path, model: str,
                 n: int = N_DEFAULT, batch_size: int = 16,
                 chat_template_kwargs: dict | None = None) -> None:
    from vllm import SamplingParams

    arm = ARMS[arm_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{arm_name}.jsonl"
    proc_path = out_dir / f"{arm_name}.jsonl.processed"
    done = _load_done(proc_path)
    todo = [s for s in seeds if s["seed_id"] not in done]
    print(f"[{arm_name}] {len(todo)} seeds to run ({len(done)} done)", flush=True)
    if not todo:
        return

    if arm.kind == "independent":
        n_calls = n
    else:
        n_calls = -(-n // arm.per_call)
    sp = SamplingParams(temperature=arm.temperature, top_p=arm.top_p,
                        max_tokens=arm.max_tokens, n=n_calls)
    t0, n_done = time.time(), 0
    with open(out_path, "a") as out_f, open(proc_path, "a") as proc_f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            msgs = [[{"role": "system", "content": arm.system},
                     {"role": "user", "content": user_message(s)}] for s in batch]
            outs = llm.chat(msgs, sp, chat_template_kwargs=chat_template_kwargs or {},
                            use_tqdm=False)
            for s, out in zip(batch, outs):
                preds: list[str] = []
                for o in out.outputs:
                    text = strip_reasoning(o.text)
                    if arm.kind == "independent":
                        p = _clean(text)
                        if p:
                            preds.append(p)
                    else:
                        preds.extend(parse_list(text))
                preds = preds[:n]
                if len(preds) < n:
                    print(f"[{arm_name}] {s['seed_id']}: {len(preds)}/{n} usable", flush=True)
                rec = {"seed_id": s["seed_id"], "arm": arm_name, "model": model,
                       "predictions": preds}
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush(); os.fsync(out_f.fileno())
                proc_f.write(s["seed_id"] + "\n")
                proc_f.flush(); os.fsync(proc_f.fileno())
            n_done += len(batch)
            rate = n_done / (time.time() - t0 + 1e-9)
            print(f"[{arm_name}] {n_done}/{len(todo)} seeds ({rate * 3600:.0f} seeds/h)", flush=True)


def load_generations(gen_dir: Path) -> dict[str, dict[str, list[str]]]:
    """{arm: {seed_id: predictions}} for every ``<arm>.jsonl`` in ``gen_dir``.

    Last write wins per seed, so a resumed run that re-emitted a seed is safe.
    """
    arms: dict[str, dict[str, list[str]]] = {}
    for p in sorted(Path(gen_dir).glob("*.jsonl")):
        recs: dict[str, list[str]] = {}
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    recs[r["seed_id"]] = [t for t in r["predictions"] if t]
        arms[p.stem] = recs
    return arms
