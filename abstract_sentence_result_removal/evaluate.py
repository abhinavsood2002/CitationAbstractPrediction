"""Score the CSAbstruct DistilBERT classifier and the zero-shot LLM against both annotators in
review.md, and write agreement.md (numbers only, no abstract text).

    CUDA_VISIBLE_DEVICES=0,1 python abstract_sentence_result_removal/evaluate.py
"""
import argparse
import os
import re
from pathlib import Path

import torch
from sklearn.metrics import cohen_kappa_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from labelling import LABELS, label_abstracts, load_llm

HERE = Path(__file__).resolve().parent
DISTILBERT = "typeof/distilbert_base_uncased_csabstruct"
# The checkpoint's config.json id2label (BACKGROUND, OBJECTIVE, METHOD, RESULT, OTHER) is wrong: its
# output indices follow the allenai/csabstruct ClassLabel order. Checked on the CSAbstruct test
# split (1,349 sentences): 75.2% accuracy with this order, a clean permutation with the config's.
DISTILBERT_ORDER = ["BACKGROUND", "METHOD", "OBJECTIVE", "OTHER", "RESULT"]
REMOVED = {"RESULT", "OTHER"}  # what remove_results.py deletes


def read_review(path: Path):
    """-> [(title, [sentence]), ...], claude labels, user labels (flat, in file order)."""
    abstracts, claude, user = [], [], []
    for line in open(path):
        if line.startswith("## "):
            abstracts.append((line[3:].split(". ", 1)[1].strip(), []))
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        if len(cells) == 4 and cells[0].isdigit():
            claude.append(cells[1].strip("*"))
            user.append(cells[2].strip("*"))
            abstracts[-1][1].append(cells[3].replace("\\|", "|"))
    assert set(claude) | set(user) <= set(LABELS)
    return abstracts, claude, user


def distilbert_labels(sentences: list[str]) -> list[str]:
    tok = AutoTokenizer.from_pretrained(DISTILBERT)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT).eval()
    with torch.no_grad():
        enc = tok(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
        return [DISTILBERT_ORDER[i] for i in model(**enc).logits.argmax(1).tolist()]


def row(name: str, ref: list[str], pred: list[str]) -> str:
    r, p = [x in REMOVED for x in ref], [x in REMOVED for x in pred]
    tp = sum(a and b for a, b in zip(r, p))
    return (f"| {name} | {cohen_kappa_score(ref, pred):.2f} | {tp / max(sum(p), 1):.2f} "
            f"| {tp / max(sum(r), 1):.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--tp", type=int, default=2)
    args = ap.parse_args()

    abstracts, claude, user = read_review(HERE / "review.md")
    flat = [s for _, sents in abstracts for s in sents]
    bert = distilbert_labels(flat)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # torch is already loaded here
    llm = load_llm(args.model, args.tp)
    llm_lab = [lab or "UNPARSED" for labs in label_abstracts(llm, args.model, abstracts) for lab in labs]

    short = args.model.split("/")[-1]
    out = ["# Agreement with the human annotator", "",
           f"{len(abstracts)} abstracts, {len(flat)} sentences (`review.md`). kappa is 5-way; P/R are for "
           "the delete decision (RESULT or OTHER).", "",
           "| labeller | kappa | P | R |", "|---|---|---|---|",
           row("Claude (second annotator)", user, claude),
           row("DistilBERT-CSAbstruct", user, bert),
           row(short, user, llm_lab), "",
           f"{llm_lab.count('UNPARSED')} unparsed LLM labels. The LLM prompt was revised once after a "
           "first run on these abstracts, so its row is in-sample."]
    tmp = HERE / "agreement.md.tmp"
    tmp.write_text("\n".join(out) + "\n")
    os.replace(tmp, HERE / "agreement.md")
    print("\n".join(out), "\n\nLLM vs user disagreements:")
    for s, u, c, g in zip(flat, user, claude, llm_lab):
        if g != u:
            print(f"  llm={g} user={u} claude={c} :: {s[:140]}")


if __name__ == "__main__":
    main()
