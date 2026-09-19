"""Give generated abstracts the treatment remove_results.py gave the citer abstracts: label every
sentence with the same LLM and prompt (labelling.py) and delete the RESULT and OTHER sentences.

    python abstract_sentence_result_removal/remove_results_generations.py \
        --conditions gemma_t10,gptoss,gptoss_medium

Reads <gen-dir>/<condition>/shard_*.jsonl and writes <out-dir>/<condition>/shard_0_of_1.jsonl in
the same schema, so scripts/rerank.py and scripts/score.py take --gen-dir <out-dir> unchanged
(<out-dir> defaults to <gen-dir>/results_other_citer_removed, which a2a.generate.load_generations
skips when it reads <gen-dir> because it holds no shard files of its own). <gen-dir> is only read.

With no --shard this launches one worker per --tp GPUs (logs in results/logs/), waits, then merges.
Workers append text-free label records to <label-dir>/labels_shard_<i>.jsonl, fsync'd per chunk,
and skip (condition, seed) units already there, so a killed run resumes. The merge re-splits each
generation with the same pysbd splitter and writes every output file atomically, plus stats.json.

As for the citers, a kept generation is its kept sentences joined by a space, sentences with no
parsed label are kept, and a generation left with no sentence is dropped: it is written as "" so
that positions still line up with <gen-dir>, and load_generations skips it, so that seed is scored
on fewer than 51 generations. Each record also carries `labels` (one list per generation).
Generations have no title, so the prompt's title line reads NO_TITLE.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from labelling import MAX_SENTENCES, label_abstracts, load_llm, split_sentences  # noqa: E402

REMOVED = {"RESULT", "OTHER"}
NO_TITLE = "(untitled)"
CHUNK = 24  # (condition, seed) units per LLM call: 24 x 51 abstracts, about remove_results.CHUNK


def read_generations(gen_dir: Path, conditions: list[str]) -> dict[tuple[str, str], dict]:
    """{(condition, seed_id): record}, last write wins per seed; predictions are left as written."""
    recs = {}
    for cond in conditions:
        paths = sorted((gen_dir / cond).glob("shard_*.jsonl"))
        if not paths:
            sys.exit(f"no generations under {gen_dir / cond}")
        for path in paths:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        recs[cond, r["seed_id"]] = r
    return recs


def select(recs: dict, max_seeds: int | None) -> list[tuple[str, str]]:
    """Units in a fixed order; --max-seeds keeps seeds spread evenly over the sorted ids."""
    seeds = sorted({sid for _, sid in recs})
    if max_seeds and max_seeds < len(seeds):
        seeds = seeds[::len(seeds) // max_seeds][:max_seeds]
    keep = set(seeds)
    return sorted(u for u in recs if u[1] in keep)


def split_all(texts: list[str], procs: int) -> list[list[str]]:
    if procs <= 1 or len(texts) < 2000:
        return [split_sentences(t) for t in texts]
    with Pool(procs) as pool:
        return pool.map(split_sentences, texts, chunksize=256)


def read_labels(label_dir: Path) -> dict[tuple[str, str], list]:
    """All shard files -> {(condition, seed_id): labels per generation}. A line cut short by a kill
    is dropped from its file first, so the next append starts on a clean line."""
    labels = {}
    for path in sorted(label_dir.glob("labels_shard_*.jsonl")):
        lines = path.read_text().splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines.pop()
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(lines))
            os.replace(tmp, path)
        for line in lines:
            r = json.loads(line)
            labels[r["condition"], r["seed_id"]] = r["labels"]
    return labels


def work(args):
    recs = read_generations(args.gen_dir, args.conditions)
    units = select(recs, args.max_seeds)[args.shard::args.n_shards]
    done = set(read_labels(args.label_dir))  # any shard file, so another shard count still resumes
    todo = [u for u in units if u not in done]
    print(f"[shard {args.shard}] {len(units)} units, {len(units) - len(todo)} done, "
          f"{len(todo)} to do", flush=True)
    if not todo:
        return
    # split everything first (CPU, forked before CUDA exists), so the GPU never waits for pysbd
    flat = [t for u in todo for t in recs[u]["predictions"]]
    sents_flat = split_all(flat, args.split_procs)
    sents, k = {}, 0
    for u in todo:
        n = len(recs[u]["predictions"])
        sents[u] = sents_flat[k:k + n]
        k += n
    llm = load_llm(args.model, args.tp)
    out = args.label_dir / f"labels_shard_{args.shard}.jsonl"
    with open(out, "a") as f:
        for k in range(0, len(todo), CHUNK):
            chunk = todo[k:k + CHUNK]
            where = [(u, j) for u in chunk for j, s in enumerate(sents[u]) if 0 < len(s) <= MAX_SENTENCES]
            labs = label_abstracts(llm, args.model, [(NO_TITLE, sents[u][j]) for u, j in where])
            by = dict(zip(where, labs))
            for u in chunk:
                rec = {"condition": u[0], "seed_id": u[1],
                       "labels": [by.get((u, j), [None] * len(s)) for j, s in enumerate(sents[u])]}
                f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(f"[shard {args.shard}] {k + len(chunk)}/{len(todo)}", flush=True)


def merge(args):
    recs = read_generations(args.gen_dir, args.conditions)
    units = select(recs, args.max_seeds)
    labels = read_labels(args.label_dir)
    missing = [u for u in units if u not in labels]
    if missing:
        sys.exit(f"{len(missing)} units have no labels in {args.label_dir}; rerun without --merge-only")
    sents_flat = split_all([t for u in units for t in recs[u]["predictions"]], args.split_procs)
    stats = {c: Counter() for c in args.conditions}
    lines = {c: [] for c in args.conditions}
    k = 0
    for u in units:
        rec, st, kept_texts = recs[u], stats[u[0]], []
        assert len(labels[u]) == len(rec["predictions"]), (u, len(labels[u]), len(rec["predictions"]))
        for text, labs in zip(rec["predictions"], labels[u]):
            sents = sents_flat[k]
            k += 1
            assert len(labs) == len(sents), (u, len(labs), len(sents))
            kept = [s for s, lab in zip(sents, labs) if lab not in REMOVED]
            kept_texts.append(" ".join(kept))
            st["generations"] += 1
            st["sentences"] += len(sents)
            st["sentences_removed"] += len(sents) - len(kept)
            st["words"] += len(text.split())
            st["words_kept"] += sum(len(s.split()) for s in kept)
            st["generations_unlabelled_too_long"] += len(sents) > MAX_SENTENCES
            st["generations_emptied"] += not kept
            st["generations_one_sentence_left"] += len(kept) == 1
            st["generations_unchanged"] += len(kept) == len(sents)
            st["generations_last_sentence_removed"] += bool(sents) and labs[-1] in REMOVED
            for lab in labs:
                st[f"label_{lab or 'UNPARSED'}"] += 1
        st["seeds"] += 1
        st["seeds_short_of_original"] += sum(bool(t) for t in kept_texts) < sum(bool(t) for t in rec["predictions"])
        lines[u[0]].append(json.dumps({**rec, "predictions": kept_texts, "labels": labels[u]},
                                      ensure_ascii=False) + "\n")
    for cond in args.conditions:
        path = args.out_dir / cond / "shard_0_of_1.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(lines[cond]))
        os.replace(tmp, path)
    info = {"source": str(args.gen_dir), "model": args.model, "removed_labels": sorted(REMOVED),
            "splitter": "pysbd 0.3.4", "title": NO_TITLE,
            "conditions": {c: dict(sorted(s.items())) for c, s in stats.items()}}
    stats_tmp = args.out_dir / "stats.json.tmp"
    stats_tmp.write_text(json.dumps(info, indent=2) + "\n")
    os.replace(stats_tmp, args.out_dir / "stats.json")
    print(json.dumps(info, indent=2))


def launch(args):
    gpus = args.gpus.split(",")
    groups = [gpus[i:i + args.tp] for i in range(0, len(gpus) - args.tp + 1, args.tp)]
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
    procs = []
    for i, g in enumerate(groups):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(g), "VLLM_PORT": str(29600 + 20 * i)}
        log = open(ROOT / "results" / "logs" / f"result_removal_gen_shard_{i}{args.tag}.log", "w")
        cmd = [sys.executable, __file__, "--shard", str(i), "--n-shards", str(len(groups)),
               "--model", args.model, "--tp", str(args.tp), "--gen-dir", str(args.gen_dir),
               "--out-dir", str(args.out_dir), "--label-dir", str(args.label_dir),
               "--conditions", ",".join(args.conditions), "--split-procs", str(args.split_procs)]
        if args.max_seeds:
            cmd += ["--max-seeds", str(args.max_seeds)]
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f"worker exit codes {codes}; see results/logs/result_removal_gen_shard_*.log, then rerun")
    merge(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", type=Path, default=ROOT / "results" / "gen")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <gen-dir>/results_other_citer_removed")
    ap.add_argument("--label-dir", type=Path, default=ROOT / "results" / "result_removal" / "generations")
    ap.add_argument("--conditions", required=True, help="comma-separated condition directories")
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--split-procs", type=int, default=16, help="CPU processes for pysbd, per worker")
    ap.add_argument("--max-seeds", type=int, default=None, help="pilot: this many seeds, evenly spread")
    ap.add_argument("--tag", default="", help="suffix for the worker log names")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--n-shards", type=int)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    args.conditions = args.conditions.split(",")
    args.out_dir = args.out_dir or args.gen_dir / "results_other_citer_removed"
    args.label_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge(args)
    elif args.shard is not None:
        work(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
