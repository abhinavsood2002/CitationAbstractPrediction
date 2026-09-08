# Decisions

Dated rationale for choices that the code does not explain by itself.
Newest entries at the bottom.

## 2026-09-08 — Scope: a task paper, not a systems paper

The repository exists to establish the *abstract-to-abstract* task: given a
paper's abstract, produce a set of abstracts anticipating the papers that
will cite it, scored by coverage of the realised citing literature. Three
empirical sections, one question each:

1. Characterise the target sets (how many futures, how far from the seed,
   how spread out).
2. Show that off-the-shelf LLMs do not recreate that spread (coverage vs
   null and reference systems; diversity of prediction sets vs real sets).
3. Show that coverage is not redundant with reference-free diversity or
   seed-overlap metrics (per-seed correlations, arm rankings, far-tail
   coverage).

Everything not needed for those three sections was left out of the port:
reframe classification, citation-function labelling, RESULT-stripping,
decoding-strategy comparisons.

## 2026-09-08 — Seeds are ACL Anthology papers; citers are anything

Seed set = every S2 paper with a non-null `externalids.ACL` (ACL Anthology
id). This is a reproducible, venue-agnostic definition that includes
workshops, LREC, SemEval and journals (92,659 papers in the March 2026
release). Venue-string matching was rejected: S2 venue names are noisy and
the same conference appears under several strings.

Citers are *not* filtered by venue or field. The earlier anticipation work
kept a citation-count floor (>= 10) on citers and required a citation
context; both bias towards well-cited, open-access citers. The point of the
ACL-seed corpus is to see the whole realised future of an NLP paper,
including uses in other disciplines, so the default floor is 0 and contexts
are recorded but not required.

## 2026-09-08 — Benchmark filters

* Seed and citer abstracts 400-3500 characters (drops stubs and pasted
  full texts; same bounds as the earlier work so numbers stay comparable).
* Citer year >= seed year; citer is not the seed.
* No author-name overlap between seed and citer. Self-citations are the
  easiest futures to anticipate and would inflate every system equally.
  Name matching is approximate (initials collide); the error is
  conservative (over-exclusion).
* Seeds published 2012-2022 with >= 20 qualifying citers. 2022 leaves three
  years of citation accumulation before the snapshot.
* Deterministic sampling: seeds ordered by BLAKE2b("seed", id), first
  1,000 taken; citers per seed ordered by BLAKE2b("citer", seed, citer),
  first 100 taken. Dev/test split by BLAKE2b(seed) mod 10 == 0 (~10%). The
  id manifest (`data/acl_a2a_ids.json`) is committed; the text is not
  (Semantic Scholar terms), and is rebuilt from the manifest.

## 2026-09-08 — Metric: coverage at a null-calibrated threshold

Coverage@tau = fraction of a seed's real citers whose best cosine
similarity to any prediction is >= tau. tau* is the 95th percentile of the
best-similarity distribution of the *random-pool* null (N random citers of
other seeds), calibrated once on the dev split and frozen for test. A grid
average (AUC) is reported in the JSON but not used for claims: it saturates
and lets chance systems look competitive.

Encoder: SciNCL (CLS) as primary because it is the scientific-document
encoder used in the earlier work; MiniLM as the robustness check because
SciNCL's objective (citation neighbours are close) is related to the task.

## 2026-09-08 — Reference systems, not just nulls

Two systems bound what coverage can mean without any model:

* `ref/retrieval_knn`: the N real abstracts (of other seeds' citers)
  closest to the seed. If an LLM does not beat it, the LLM adds nothing
  over topical proximity.
* `ref/split_half`: half of a seed's own citers predicting the other half.
  This is what N real futures of the same paper cover; the gap to it is the
  "diversity not recreated" claim in one number.

## 2026-09-08 — Generation arms

Proposal-style prompt (no invented results) at temperatures 0.7 / 1.0 / 1.3
plus an explicit-diversity list prompt (10 per call). The temperature sweep
exists for the contrast section: reference-free diversity rises with
temperature monotonically; coverage should not. N = 50 per seed;
coverage@N for N in {1,5,10,20,50} is reported so the budget is not a
hidden choice. Reasoning channels are switched off or set to low: this is
sampling, not problem solving.

Contamination: most citers predate every generator's training cutoff.
`post_cutoff` (citer year >= 2025) is a reported stratum, not a filter, so
the reader sees whether coverage drops on citers the model cannot have
read.
