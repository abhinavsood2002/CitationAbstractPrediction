"""Sentence splitting and zero-shot rhetorical-role labelling, shared by evaluate.py and
remove_results.py. The LLM sees a whole abstract as numbered sentences and returns one label each."""
import re
import sys
from pathlib import Path

import pysbd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a2a.llm import chat_template_kwargs_for, strip_reasoning  # noqa: E402

LABELS = ["BACKGROUND", "OBJECTIVE", "METHOD", "RESULT", "OTHER"]
MAX_SENTENCES = 120  # longer "abstracts" are full texts or debris; left unlabelled (all kept)

PROMPT = """You are annotating the rhetorical role of each sentence in a scientific abstract.

Labels:
- BACKGROUND: context, prior work, motivation, the problem or gap, limitations of existing work, \
definitions of the task or its terms.
- OBJECTIVE: what this paper sets out to do: its aim, research question or main contribution \
("we propose X", "this paper studies Y", "we address two problems").
- METHOD: how the work is done: the approach and its components, further proposed techniques, \
data, experimental setup, what was evaluated or compared. A sentence that only says experiments \
or analyses were carried out, without saying what they showed, is METHOD.
- RESULT: anything the paper found, showed or concluded: performance numbers, comparisons with \
baselines, rankings, what experiments or analyses demonstrate, confirm, reveal or suggest, \
interpretations of the findings, and claims about what the work provides or how well it works \
("our findings suggest ...", "we provide insights into ...", "the method is effective"). A \
sentence that describes a method and also states a finding is RESULT.
- OTHER: text that carries none of the above: future-work and "this will facilitate further \
research" statements, code or data availability, disclaimers, headings, list stubs, broken or \
leftover markup.

Title: {title}

Sentences:
{sentences}

Read the whole abstract first; a sentence's role depends on its context. Give exactly one label \
per sentence, one per line, in the form "<index>: <LABEL>". Output nothing else."""

_LINE_RE = re.compile(r"^\s*\[?(\d+)\]?\s*[:.\-]\s*\**([A-Za-z]+)", re.MULTILINE)
_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SEGMENTER.segment(" ".join(text.split())) if s.strip()]


def parse(text: str, n: int) -> list[str | None]:
    """One label per sentence; None where the model gave no valid label for that index."""
    got = {int(i): lab.upper() for i, lab in _LINE_RE.findall(text)}
    return [got[i] if got.get(i) in LABELS else None for i in range(n)]


def load_llm(model: str, tp: int, max_model_len: int = 8192, gpu_mem_util: float = 0.85):
    from vllm import LLM
    return LLM(model=model, tensor_parallel_size=tp, max_model_len=max_model_len,
               gpu_memory_utilization=gpu_mem_util, dtype="auto", trust_remote_code=True)


def label_abstracts(llm, model: str, items: list[tuple[str, list[str]]]) -> list[list[str | None]]:
    """items: (title, sentences) per abstract -> labels per abstract (greedy, thinking off)."""
    from vllm import SamplingParams
    msgs = [[{"role": "user", "content": PROMPT.format(
        title=title, sentences="\n".join(f"{i}: {s}" for i, s in enumerate(sents)))}]
        for title, sents in items]
    sps = [SamplingParams(temperature=0.0, max_tokens=12 * len(sents) + 32, seed=0)
           for _, sents in items]
    outs = llm.chat(msgs, sps, chat_template_kwargs=chat_template_kwargs_for(model), use_tqdm=False)
    return [parse(strip_reasoning(o.outputs[0].text), len(sents))
            for o, (_, sents) in zip(outs, items)]
