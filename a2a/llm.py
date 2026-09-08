"""Small vLLM helpers shared by generation scripts: chat-template flags per model
family and stripping of reasoning channels from sampled text."""

import re

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def chat_template_kwargs_for(model_name: str) -> dict:
    """Chat-template kwargs that turn *off* reasoning for generation.

    Generation is sampling, not problem solving: a long hidden reasoning trace
    only eats the token budget. gpt-oss cannot disable reasoning, so it is set
    to ``low``; Qwen3 and Gemma honour ``enable_thinking``; Llama and other
    plain instruct models take no flag.
    """
    name = model_name.lower()
    if "gpt-oss" in name:
        return {"reasoning_effort": "low"}
    if "qwen3" in name or "gemma" in name:
        return {"enable_thinking": False}
    return {}


def strip_reasoning(text: str) -> str:
    """Remove a reasoning channel that leaked into the sampled text.

    * gpt-oss (Harmony): ``analysis<...>assistantfinal<answer>`` -> after the
      literal ``assistantfinal``; an output that never reached the final
      channel is discarded (empty string).
    * Gemma-4: ``<|channel>thought ...<channel|><answer>`` -> after the last
      ``<channel|>``.
    * Qwen3 / DeepSeek: ``<think>...</think>`` blocks are removed.
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
