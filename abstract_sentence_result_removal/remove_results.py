"""Label every citer abstract's sentences with the LLM and write a copy of the benchmark whose
citer abstracts have RESULT and OTHER sentences removed.

    python abstract_sentence_result_removal/remove_results.py --gpus 0,1,2,3,4,5,6,7

With no --shard this launches one worker per --tp GPUs (logs in results/logs/), waits, then merges.
Workers append text-free label records to results/result_removal/labels_shard_<i>.jsonl, fsync'd
per chunk, and skip citers already there, so a killed run resumes. The merge re-splits each
abstract with the same pysbd splitter and writes --out atomically, plus a *_stats.json next to it;
--data is only read, and an existing --out is never replaced without --force.

Each citer in --out keeps the benchmark schema, with `abstract` = the kept sentences joined,
`abstract_full` = the original text, and `sentences` = [{text, label, removed}] for every
sentence, so what was deleted can be audited. Sentences with no parsed label (label null), and
abstracts over labelling.MAX_SENTENCES, are kept as they are. Citers whose abstract is emptied
are dropped, and then any seed left with fewer than MIN_CITERS citers.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from labelling import MAX_SENTENCES, label_abstracts, load_llm, split_sentences  # noqa: E402

REMOVED = {"RESULT", "OTHER"}
MIN_CITERS = 10  # BenchmarkParams floor on qualifying citers per seed
LABEL_DIR = ROOT / "results" / "result_removal"
CHUNK = 1024


def unique_citers(data: str) -> dict:
    citers = {}
    with open(data) as f:
        for line in f:
            for c in json.loads(line)["citers"]:
                citers.setdefault(c["id"], c)
    return citers


def read_labels() -> dict:
    """All shard files -> {citer_id: labels}. A line cut short by a kill is dropped from its file
    first, so the next append starts on a clean line."""
    labels = {}
    for path in sorted(LABEL_DIR.glob("labels_shard_*.jsonl")):
        lines = path.read_text().splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines.pop()
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(lines))
            os.replace(tmp, path)
        for line in lines:
            r = json.loads(line)
            labels[r["citer_id"]] = r["labels"]
    return labels


def check_out(args) -> tuple[Path, Path]:
    out = Path(args.out)
    stats_path = out.with_name(out.stem + "_stats.json")
    if Path(args.data).resolve() == out.resolve():
        sys.exit("--out must differ from --data")
    for path in (out, stats_path):
        if path.exists() and not args.force:
            sys.exit(f"{path} exists; pass --force to replace it")
    return out, stats_path


def work(args):
    citers = unique_citers(args.data)
    ids = sorted(citers)[args.shard::args.n_shards]
    out = LABEL_DIR / f"labels_shard_{args.shard}.jsonl"
    done = set(read_labels())  # any shard file, so a rerun with another shard count still resumes
    todo = [i for i in ids if i not in done]
    print(f"[shard {args.shard}] {len(ids)} citers, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    llm = load_llm(args.model, args.tp)
    with open(out, "a") as f:
        for k in range(0, len(todo), CHUNK):
            chunk = todo[k:k + CHUNK]
            sents = [split_sentences(citers[i]["abstract"]) for i in chunk]
            ok = [j for j, s in enumerate(sents) if 0 < len(s) <= MAX_SENTENCES]
            labs = label_abstracts(llm, args.model, [(citers[chunk[j]]["title"], sents[j]) for j in ok])
            by_j = dict(zip(ok, labs))
            for j, i in enumerate(chunk):
                rec = {"citer_id": i, "labels": by_j.get(j, [None] * len(sents[j]))}
                f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(f"[shard {args.shard}] {k + len(chunk)}/{len(todo)}", flush=True)


def merge(args):
    out, stats_path = check_out(args)
    labels = read_labels()
    missing = set(unique_citers(args.data)) - set(labels)
    if missing:
        sys.exit(f"{len(missing)} citers have no labels in {LABEL_DIR}; rerun without --merge-only")
    stats = Counter()
    seen = set()
    split_cache = {}  # a citer shared by several seeds is split once
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(args.data) as fin, open(tmp, "w") as fout:
        for line in fin:
            seed = json.loads(line)
            n_nonempty = 0
            for c in seed["citers"]:
                if c["id"] not in split_cache:
                    split_cache[c["id"]] = split_sentences(c["abstract"])
                sents = split_cache[c["id"]]
                labs = labels[c["id"]]
                assert len(labs) == len(sents), (c["id"], len(labs), len(sents))
                kept = [s for s, lab in zip(sents, labs) if lab not in REMOVED]
                c["abstract_full"] = c["abstract"]
                c["abstract"] = " ".join(kept)
                c["sentences"] = [{"text": s, "label": lab, "removed": lab in REMOVED}
                                  for s, lab in zip(sents, labs)]
                n_nonempty += bool(kept)
                if c["id"] not in seen:
                    seen.add(c["id"])
                    stats["citers"] += 1
                    stats["sentences"] += len(sents)
                    stats["sentences_removed"] += len(sents) - len(kept)
                    stats["citers_unlabelled_too_long"] += len(sents) > MAX_SENTENCES
                    stats["citers_emptied"] += not kept
                    stats["citers_one_sentence_left"] += len(kept) == 1
                    stats["citers_unchanged"] += len(kept) == len(sents)
                    for lab in labs:
                        stats[f"label_{lab or 'UNPARSED'}"] += 1
            # an empty target cannot be matched: drop it, then re-apply the benchmark's citer floor
            stats["pairs_dropped_emptied"] += len(seed["citers"]) - n_nonempty
            seed["citers"] = [c for c in seed["citers"] if c["abstract"]]
            if len(seed["citers"]) < MIN_CITERS:
                stats["seeds_dropped_under_floor"] += 1
                stats["pairs_dropped_with_seed"] += len(seed["citers"])
                continue
            seed["n_citers_qualifying"] = len(seed["citers"])
            stats["seeds"] += 1
            stats["pairs"] += len(seed["citers"])
            fout.write(json.dumps(seed, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    info = {"source": args.data, "model": args.model, "removed_labels": sorted(REMOVED),
            "splitter": "pysbd 0.3.4", **dict(sorted(stats.items()))}
    stats_tmp = stats_path.with_suffix(".json.tmp")
    stats_tmp.write_text(json.dumps(info, indent=2) + "\n")
    os.replace(stats_tmp, stats_path)
    print(json.dumps(info, indent=2))


def launch(args):
    check_out(args)  # fail before the GPU work, not after it
    gpus = args.gpus.split(",")
    groups = [gpus[i:i + args.tp] for i in range(0, len(gpus) - args.tp + 1, args.tp)]
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
    procs = []
    for i, g in enumerate(groups):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(g), "VLLM_PORT": str(29600 + 20 * i)}
        log = open(ROOT / "results" / "logs" / f"result_removal_shard_{i}.log", "w")
        cmd = [sys.executable, __file__, "--shard", str(i), "--n-shards", str(len(groups)),
               "--model", args.model, "--tp", str(args.tp), "--data", args.data]
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f"worker exit codes {codes}; see results/logs/result_removal_shard_*.log, then rerun")
    merge(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "acl_a2a.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "acl_a2a_noresults.jsonl"))
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--n-shards", type=int)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    args = ap.parse_args()
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge(args)
    elif args.shard is not None:
        work(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
