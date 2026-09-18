# Agreement with the human annotator

20 abstracts, 169 sentences (`review.md`). kappa is 5-way; P/R are for the delete decision (RESULT or OTHER).

| labeller | kappa | P | R |
|---|---|---|---|
| Claude (second annotator) | 0.92 | 0.94 | 0.94 |
| DistilBERT-CSAbstruct | 0.70 | 0.82 | 0.82 |
| gemma-4-26B-A4B-it | 0.91 | 0.94 | 0.96 |

0 unparsed LLM labels. The LLM prompt was revised once after a first run on these abstracts, so its row is in-sample.
