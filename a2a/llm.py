"""Model defaults for the run: the two generators, the reranker, and how reasoning
channels are switched off and stripped. Everything model-specific lives here."""

import re
from dataclasses import dataclass, field


# Truncation is shared by every generator and every condition, so the models differ in
# nothing but their weights and a condition in nothing but its temperature. The values are
# Gemma 4's generation_config.json; gpt-oss's own recommendation is top_p 1.0 with no top_k,
# and untruncated sampling alone made its sets more diverse than Gemma's on the pilot.
TOP_P = 0.95
TOP_K = 64


@dataclass(frozen=True)
class Generator:
    hf_id: str
    tp: int
    temperature: float              # the model's recommended value; a condition may override it
    top_p: float = TOP_P
    top_k: int = TOP_K
    chat_template_kwargs: dict = field(default_factory=dict)
    reasoning_tokens: int = 0       # added to max_tokens so the trace cannot eat the answer
    # vLLM keeps its small defaults on A100s: 256 concurrent sequences per process
    max_num_seqs: int = 256
    batch_size: int = 16            # seeds per llm.chat call (x 51 samples each)


# gpt-oss reasoning effort -> reasoning_tokens. A condition may name an effort; the
# generator's own setting is "low". A trace that never closes is discarded and topped up.
REASONING_TOKENS = {"low": 1024, "high": 6144}

GENERATORS: dict[str, Generator] = {
    # thinking off. Weights take 48.5 GiB, so the 20 GiB KV cache is near full at 256.
    "gemma": Generator("google/gemma-4-26B-A4B-it", tp=1, temperature=1.0,
                       chat_template_kwargs={"enable_thinking": False}),
    # Reasoning cannot be disabled, so it runs at the lowest effort by default. Weights take
    # 13.7 GiB and at 256 sequences the 56 GiB KV cache (1.2M tokens) was 7% used, so it runs
    # 1,024 sequences, with 64 seeds (3,264 samples) per call to keep that queue full.
    "gpt-oss": Generator("openai/gpt-oss-20b", tp=1, temperature=1.0,
                         chat_template_kwargs={"reasoning_effort": "low"},
                         reasoning_tokens=REASONING_TOKENS["low"],
                         max_num_seqs=1024, batch_size=64),
}

# ---------------------------------------------------------------- reranker

RERANKER = "Qwen/Qwen3-Reranker-0.6B"
# vLLM loads the original checkpoint as a 2-way classifier over the "no"/"yes" logits,
# so the score is P(yes).
RERANKER_HF_OVERRIDES = {"architectures": ["Qwen3ForSequenceClassification"],
                         "classifier_from_token": ["no", "yes"],
                         "is_original_qwen3_reranker": True}
RERANKER_MAX_LEN = 2048
MATCH_THRESHOLD = 0.5

# The instruction is what "match" means. Query = a real citing abstract, document = a
# generated abstract.
RERANK_INSTRUCTION = (
    "Given a research paper abstract as the query, retrieve descriptions of research that "
    "address the same problem with the same kind of method. Sharing only a broad topic or "
    "field is not relevant.")

_RERANK_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on '
                  'the Query and the Instruct provided. Note that the answer can only be "yes" or '
                  '"no".<|im_end|>\n<|im_start|>user\n')
_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def rerank_query(text: str) -> str:
    return f"{_RERANK_PREFIX}<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {text}\n"


def rerank_document(text: str) -> str:
    return f"<Document>: {text}{_RERANK_SUFFIX}"


# ------------------------------------------------------- reasoning channels

def chat_template_kwargs_for(model_name: str) -> dict:
    """Chat-template kwargs that turn reasoning off (gpt-oss: lowest effort) for any model
    family; Llama and other plain instruct models take no flag."""
    name = model_name.lower()
    if "gpt-oss" in name:
        return {"reasoning_effort": "low"}
    if "gemma" in name:
        return {"enable_thinking": False}
    return {}


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove a reasoning channel that leaked into the sampled text.

    * gpt-oss (Harmony): ``analysis<...>assistantfinal<answer>`` -> after the
      literal ``assistantfinal``; an output that never reached the final
      channel is discarded (empty string).
    * Gemma-4: ``<|channel>thought ...<channel|><answer>`` -> after the last
      ``<channel|>``.
    * DeepSeek-style ``<think>...</think>`` blocks are removed.
    """
    if "assistantfinal" in text:
        return text.split("assistantfinal", 1)[1].lstrip()
    if "<|channel>thought" in text:
        parts = text.rsplit("<channel|>", 1)
        return parts[1].lstrip() if len(parts) == 2 else ""
    if text.lstrip().startswith("analysis"):
        return ""
    text = _THINK_RE.sub("", text)
    if "<think>" in text:  # unterminated think block: no answer produced
        return ""
    return text
