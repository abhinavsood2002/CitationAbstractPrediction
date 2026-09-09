# Stage 1: raw Semantic Scholar shards -> bucketed parquet

Statistics of the data layer that everything else is built on: the three Semantic
Scholar (S2) dataset dumps as downloaded, and the hash-bucketed parquet that
`a2a.s2.build_parquet` derives from them. No content filtering happens at this
stage; the only rows dropped are citation rows with a NULL `citedcorpusid`.

Computed 2026-09-09 with DuckDB 1.5.5 over the on-disk layer (queries described
at the end). Downstream layers (ACL corpus, benchmark cut) are not covered here.

* S2 Datasets API release: **2026-03-10** (shard files stamped `20260313_*`).
* Raw shards: `/mnt/scratch/abhinav/quality_diversity/{papers,abstracts,citations}`
* Derived parquet: `/mnt/scratch/abhinav/quality_diversity/derived/{papers,abstracts,citers,acl_ids}`

## 1. Sizes: raw shards and derived parquet

| dataset | raw shards | raw size | raw JSONL lines | parquet rows | parquet size | rows dropped |
|---|---:|---:|---:|---:|---:|---|
| papers | 60 | 48 GB | 233,251,301 | 233,251,301 | 17 GB | none |
| abstracts | 30 | 23 GB | 42,203,850 | 42,203,850 | 22 GB | none |
| citations -> `citers` | 358 | 337 GB | 5,585,374,089 | 4,996,625,798 | 154 GB | 588,748,291 (10.54%) with NULL `citedcorpusid` |
| papers -> `acl_ids` (side table) | 60 (same shards) | | | 92,659 | 1.0 MB | everything without `externalids.ACL` |

Layout: 64 buckets each for `papers` / `abstracts` (on `corpusid % 64`) and `citers`
(on `citedcorpusid % 64`), each bucket compacted to one `compacted.parquet` sorted on
its key. `papers` bucket 0 holds 3.64 M rows in 30 row groups; `citers` buckets hold
77.1 M to 79.1 M rows each (bucket 0: 634 row groups). Every `_done` marker is present
(60 / 30 / 358), so all shards were processed.

**Schema note.** The on-disk `papers` parquet predates the `acl_id` / `doi` / `arxiv` /
`publicationvenueid` columns in `PAPERS_SCHEMA`; it has `corpusid, title, authors, year,
publicationdate, venue, journal_name, referencecount, citationcount,
influentialcitationcount, isopenaccess, s2fieldsofstudy, publicationtypes`. That is why
`build_acl_corpus.py` auto-detects `derived/acl_ids/acl_papers.parquet` and joins it in.

## 2. Papers (233,251,301 rows)

`corpusid` is unique (distinct count equals row count).

### Field completeness

| field | NULL rows | share |
|---|---:|---:|
| title | 1 | 0.0% |
| authors | 6,655,558 | 2.9% |
| year | 3,464,566 | 1.5% |
| publicationdate | 86,644,438 | 37.1% |
| venue | 145,100,371 | 62.2% |
| journal_name | 98,769,631 | 42.3% |
| s2fieldsofstudy | 84,068,158 | 36.0% |
| publicationtypes | 158,393,956 | 67.9% |
| citationcount / referencecount | 0 | 0% |
| isopenaccess = true | 31,017,743 | 13.3% |

### Year

| year | papers |
|---|---:|
| < 1950 | 3,880,187 |
| 1950-1999 | 53,188,181 |
| 2000 | 3,204,980 |
| 2001 | 3,400,913 |
| 2002 | 3,679,592 |
| 2003 | 4,003,134 |
| 2004 | 4,543,208 |
| 2005 | 4,903,318 |
| 2006 | 5,255,355 |
| 2007 | 5,672,023 |
| 2008 | 6,129,414 |
| 2009 | 6,602,588 |
| 2010 | 7,050,408 |
| 2011 | 7,517,256 |
| 2012 | 7,923,017 |
| 2013 | 8,304,200 |
| 2014 | 8,518,167 |
| 2015 | 8,746,583 |
| 2016 | 8,807,833 |
| 2017 | 8,382,969 |
| 2018 | 8,312,888 |
| 2019 | 8,478,544 |
| 2020 | 8,562,737 |
| 2021 | 7,260,292 |
| 2022 | 6,290,634 |
| 2023 | 6,342,965 |
| 2024 | 6,708,207 |
| 2025 | 7,010,707 |
| 2026 (partial) | 1,106,435 |
| NULL | 3,464,566 |

### Citation and reference counts

| statistic | citationcount | referencecount |
|---|---:|---:|
| p25 / p50 / p75 | 0 / 0 / 5 | 0 / 0 / 16 |
| p90 / p95 / p99 / p99.9 | 24 / 49 / 164 / 680 | 41 / - / 116 / - |
| max | 343,300 | |
| mean | 11.3 | |
| = 0 | 125,250,609 (53.7%) | 135,427,621 (58.1%) |
| >= 10 | 42,866,123 (18.4%) | |
| >= 100 | 4,793,460 (2.1%) | |

Authors per paper: median 2, p90 6, p99 13, max 10,000.

### Fields of study (`s2fieldsofstudy`, multi-label)

Papers by number of labels: 0: 84.07 M, 1: 45.96 M, 2: 68.25 M, 3: 30.32 M, 4: 4.36 M, 5+: 0.30 M.

| field | papers | field | papers |
|---|---:|---|---:|
| Medicine | 59,550,130 | Education | 7,570,200 |
| Engineering | 30,641,191 | History | 7,374,440 |
| Environmental Science | 22,413,511 | Mathematics | 7,003,816 |
| Biology | 20,833,311 | Economics | 6,575,650 |
| Computer Science | 19,472,277 | Agricultural and Food Sciences | 5,715,187 |
| Chemistry | 18,303,778 | Geography | 4,637,276 |
| Physics | 16,475,314 | Geology | 3,703,712 |
| Materials Science | 15,983,949 | Art | 3,586,298 |
| Psychology | 9,977,171 | Law | 3,206,981 |
| Sociology | 8,221,170 | Philosophy | 3,067,399 |
| Political Science | 8,020,624 | Linguistics | 2,040,869 |
| Business | 7,884,253 | Agricultural And Food Sciences (casing variant) | 83,284 |

The casing variant is present in the raw data and survives into the parquet; treat the
two spellings as one field when grouping.

### Publication types (multi-label, 67.9% NULL)

JournalArticle 60,192,164; Review 17,136,722; Conference 5,135,429; CaseReport 2,461,392;
Study 2,378,332; LettersAndComments 1,632,933; Editorial 773,572; ClinicalTrial 584,836;
Book 540,477; News 250,517; MetaAnalysis 117,969; Dataset 2,642.

### Most frequent venue strings (62.2% NULL)

Nature 425,219; arXiv.org 374,050; Social Science Research Network 333,722; PLoS ONE
323,296; Scientific Reports 302,363; bioRxiv 297,936; Science 277,470; British medical
journal 235,332; Journal of Biological Chemistry 197,428; Reactions weekly 175,615; PNAS
164,531; The Lancet 159,290.

## 3. Abstracts (42,203,850 rows)

* One row per `corpusid`, none NULL, every row has a matching `papers` row (0 orphans).
* **18.1% of papers have an abstract in S2.**
* Length in characters: p1 62, p5 319, p25 879, **p50 1,262**, p75 1,652, p95 2,405, p99 3,563; mean 1,311.
* Under the benchmark's 400-3,500 character window: 39,002,623 (92.4%) inside;
  2,748,015 (6.5%) shorter than 400; 453,212 (1.1%) longer than 3,500.

### Coverage by year and open-access status

Abstract availability depends heavily on year and on open access (S2 redistributes only
abstracts it is licensed to). Open-access papers: 17,215,953 of 31,017,743 have an abstract
(55.5%). Non-open-access: 24,987,897 of 202,233,558 (12.4%).

| year | papers | with abstract | share |
|---|---:|---:|---:|
| < 2000 | 57,068,368 | 1,185,559 | 2.1% |
| 2000 | 3,204,980 | 98,621 | 3.1% |
| 2001 | 3,400,913 | 109,855 | 3.2% |
| 2002 | 3,679,592 | 116,474 | 3.2% |
| 2003 | 4,003,134 | 131,419 | 3.3% |
| 2004 | 4,543,208 | 150,451 | 3.3% |
| 2005 | 4,903,318 | 175,226 | 3.6% |
| 2006 | 5,255,355 | 196,250 | 3.7% |
| 2007 | 5,672,023 | 227,952 | 4.0% |
| 2008 | 6,129,414 | 266,149 | 4.3% |
| 2009 | 6,602,588 | 317,350 | 4.8% |
| 2010 | 7,050,408 | 376,639 | 5.3% |
| 2011 | 7,517,256 | 459,044 | 6.1% |
| 2012 | 7,923,017 | 571,323 | 7.2% |
| 2013 | 8,304,200 | 672,286 | 8.1% |
| 2014 | 8,518,167 | 789,247 | 9.3% |
| 2015 | 8,746,583 | 910,405 | 10.4% |
| 2016 | 8,807,833 | 1,024,005 | 11.6% |
| 2017 | 8,382,969 | 1,194,347 | 14.2% |
| 2018 | 8,312,888 | 2,457,052 | 29.6% |
| 2019 | 8,478,544 | 3,735,256 | 44.1% |
| 2020 | 8,562,737 | 4,488,218 | 52.4% |
| 2021 | 7,260,292 | 4,303,475 | 59.3% |
| 2022 | 6,290,634 | 4,005,886 | 63.7% |
| 2023 | 6,342,965 | 4,234,202 | 66.8% |
| 2024 | 6,708,207 | 4,571,799 | 68.2% |
| 2025 | 7,010,707 | 4,638,563 | 66.2% |
| 2026 (partial) | 1,106,435 | 596,659 | 53.9% |
| NULL | 3,464,566 | 200,138 | 5.8% |

Implication for the citer side of the benchmark: a citer needs an abstract to qualify, so
pre-2018 and closed-access citers are under-represented relative to the true citing
literature.

## 4. Citations -> `citers` (4,996,625,798 rows)

### Raw export structure and duplicate rows

The 358 raw shards come in two export groups (Athena/Trino file-name prefixes):

| export group | shards | raw lines | NULL cited (dropped) | kept rows |
|---|---:|---:|---:|---:|
| `20260313_071717_00052_i6vhf` | 239 | 2,961,453,633 | 319,762,690 | **2,641,690,943** |
| `20260313_073822_00034_xmmad` | 119 | 2,623,920,456 | 268,985,601 | **2,354,934,855** |
| total | 358 | 5,585,374,089 | 588,748,291 (10.54%) | 4,996,625,798 |

`citingcorpusid` is never NULL in the raw data.

The parquet has **2,641,690,943 distinct `citationid`** and exactly as many distinct
`(citedcorpusid, citingcorpusid)` pairs, i.e. `citationid` identifies a pair. That equals
the kept rows of the first export group; the second group's kept rows equal the number of
extra copies. So the S2 release ships the full citation set once (239 shards) plus a
partial re-export of 89% of it (119 shards); **47.1% of parquet rows are redundant
copies**. This is in the raw dump, not introduced by our build: parquet rows equal raw
non-NULL lines exactly, and sampled raw shards contain each `citationid` essentially once (at most 9 repeats in 24 M lines).

| copies per pair | pairs | share |
|---|---:|---:|
| 1 | 286,757,328 | 10.9% |
| 2 | 2,354,932,375 | 89.1% |
| 3 | 1,240 | 0.0% |

Nature of the copies (bucket 0, 36,702,161 duplicated pairs): 80.1% are byte-identical rows;
19.9% differ only in `contexts` (one copy NULL, the other populated; `intents` and
`isinfluential` agree). `build_edges` in `a2a/corpus.py` collapses these with
`GROUP BY (cited, citing)` keeping the non-NULL contexts, so the ACL corpus is unaffected.
Deduplicating on `citationid` at compaction time would roughly halve the 154 GB.

### Graph statistics (over rows unless stated)

| statistic | value |
|---|---:|
| distinct cited papers | 108,076,064 (46.3% of papers) |
| distinct citing papers | 95,758,008 (41.1% of papers) |
| rows with citing = cited | 2,691 |
| `isinfluential` = true | 230,200,469 rows (4.6%); never NULL |
| `contexts` non-NULL | 1,229,294,177 rows (24.6%) |
| `intents` non-NULL | 1,699,747,812 rows (34.0%) |
| in-degree per cited paper (distinct citers) | p50 6, p75 20, p90 52, p99 268, p99.9 1,066, max 343,268 |

Context strings per row: 0: 3,767 M (75.4%); 1: 797 M; 2: 230 M; 3: 80 M; 4: 41 M; 5: 23 M;
6-9: 38 M; 10+: 20 M.

Intent labels (flattened over all rows, so duplicated pairs count twice): background
1,760,100,436; methodology 398,195,320; result 95,563,199.

## 5. ACL Anthology ids (`acl_ids`, 92,659 rows)

Papers with a non-NULL `externalids.ACL`, extracted from the same 60 papers shards.

* 92,659 distinct `corpusid`; 92,657 distinct `acl_id` (two ACL ids each map to two corpus ids).
* All 92,659 have a `papers` row. With DOI: 54,391 (58.7%). With arXiv id: 20,089 (21.7%).
* **With an abstract: 54,006 (58.3%)**; with an abstract of 400-3,500 chars: 51,608 (55.7%).
* Open access: 36,646, of which 36,017 (98.3%) have an abstract. Closed: 56,013, of which
  17,989 (32.1%) have an abstract.
* Citation count: p25 2, p50 9, p75 27, p90 75, p95 141, p99 497, max 111,193, mean 43.3.
  Zero citations: 10,619 (11.5%). At least 20: 28,955 (31.2%).
* Fields of study: Computer Science 80,817; Linguistics 41,091; Medicine 3,300; Psychology
  2,046; Mathematics 1,981; Engineering 1,324; Education 1,243 (rest < 1,000 each).

### Year and abstract availability

| year (S2) | ACL papers | with abstract | share |
|---|---:|---:|---:|
| < 2000 | 8,567 | 4,476 | 52.2% |
| 2000 | 1,177 | 532 | 45.2% |
| 2001 | 755 | 320 | 42.4% |
| 2002 | 1,352 | 564 | 41.7% |
| 2003 | 1,117 | 644 | 57.7% |
| 2004 | 1,839 | 610 | 33.2% |
| 2005 | 1,271 | 518 | 40.8% |
| 2006 | 2,035 | 916 | 45.0% |
| 2007 | 1,529 | 632 | 41.3% |
| 2008 | 2,144 | 719 | 33.5% |
| 2009 | 2,149 | 1,416 | 65.9% |
| **2010** | 2,966 | **61** | **2.1%** |
| **2011** | 2,183 | **55** | **2.5%** |
| **2012** | 3,267 | **62** | **1.9%** |
| **2013** | 2,726 | **342** | **12.5%** |
| 2014 | 3,513 | 1,929 | 54.9% |
| 2015 | 2,864 | 1,988 | 69.4% |
| 2016 | 4,140 | 2,212 | 53.4% |
| 2017 | 3,377 | 2,425 | 71.8% |
| 2018 | 4,683 | 3,102 | 66.2% |
| 2019 | 4,996 | 4,395 | 88.0% |
| 2020 | 6,989 | 4,962 | 71.0% |
| 2021 | 6,540 | 5,413 | 82.8% |
| 2022 | 7,341 | 5,074 | 69.1% |
| 2023 | 5,710 | 4,414 | 77.3% |
| 2024 | 7,266 | 6,174 | 85.0% |
| 2025 | 45 | 45 | 100.0% |
| 2026 | 5 | 5 | 100.0% |
| NULL | 113 | 1 | 0.9% |

Two gaps that matter for seed selection:

1. **2010-2013 hole.** S2 has abstracts for only about 2% of ACL Anthology papers from
   2010-2012 and 12.5% from 2013. Cross-checked on the year encoded in the ACL id itself
   (`P10-`, `D11-`, ...): 2010: 54 / 2,640; 2011: 57 / 1,840; 2012: 55 / 3,020;
   2013: 329 / 2,523. Seed years are unrestricted in the final cut (DECISIONS.md 2026-09-09), but
   2010-2013 seeds are nearly absent whatever the window.
2. **2025 not yet linked.** Only 45 ACL papers carry a 2025 S2 year and no `2025.*` ACL
   id prefix appears; the release predates ACL id assignment for 2025 venues.

### ACL id prefixes and venues

Id prefixes: `W` (workshops) 16,552; `2024.` 7,985; `2022.` 7,433; `2020.` 6,795; `P` 6,575;
`2021.` 6,343; `L` 5,803; `2023.` 5,569; `C` 4,366; `D` 3,838; `N` 2,538; `S` 1,758;
`J` 1,692; `Y` 1,392; `E` 1,269; `I` 1,105.

S2 venue strings: ACL 10,624; LREC 9,062; EMNLP 8,132; COLING 5,805; NAACL 4,265; SemEval
3,127; NULL 2,592; EACL 2,314; JEP/TALN/RECITAL 1,902; PACLIC 1,848; "International
Conference on Computational Logic" 1,618 (S2 venue mapping, kept as-is); Findings 1,586;
MT Summit 1,378; IJCNLP 1,219; RANLP 1,110; CoNLL 1,049.

## How these numbers were computed

* Raw line and NULL counts: `zcat | wc -l` / `awk` over every shard (papers, abstracts,
  citations), 64 shards in parallel.
* Everything else: DuckDB 1.5.5 over `derived/*/bucket=*/compacted.parquet` and
  `derived/acl_ids/acl_papers.parquet` (600-900 GB memory limit, 96-128 threads). Distinct
  cited papers and distinct pairs are summed per bucket, which is exact because `citers`
  is bucketed on `citedcorpusid`; distinct citing papers is one global `COUNT(DISTINCT)`.
* Duplicate diagnosis: per-`citationid` grouping on `citers` bucket 0 plus a
  `citationid` overlap check on three raw shards (two from the 239-shard group, one from
  the 119-shard group): no cross-shard overlap and at most 9 within-shard repeats.
