# Main results

Aggregated from results/runs by scripts/make_tables.py — '—' means the metric was recorded unavailable/blocked for that run, never imputed. 'seeds' counts aggregated runs; a '*' marks a value reported by only a subset of the group's runs.

| System | n | seeds | Halluc↓ | Omit↓ | Calib ρ↑ | BLEU | ROUGE-L | BERTSc | Dist-2↑ | SelfBLEU↓ | Rep4g↓ | p50ms↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1_template | 689 | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.549 ± 0.000 | 5.863 ± 0.558 | 0.305 ± 0.008 | — | 0.423 ± 0.009 | 0.089 ± 0.018 | 0.040 ± 0.008 | 0.056 ± 0.000 |
| b1_template_nogamestate | 689 | 1 | 0.000 | 0.000 | 0.046 | 5.535 | 0.303 | — | 0.412 | 0.109 | 0.032 | 0.055 |
