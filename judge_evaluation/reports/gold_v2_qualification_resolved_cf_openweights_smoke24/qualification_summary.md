# Gold v2 Qualification Summary

| model_key | rows | correct | accuracy_pct | binary_complete | false_complete | missed_complete | net_complete_bias | errors | content_filters | providers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3.6_27b_venice_no_reasoning | 24/1880 | 16 | 0.851 | 619 | 2 | 1259 | -1257 | 0 | 0 | Venice:24 |
| qwen3.6_27b_venice_reasoning_medium | 0/1880 | 0 | 0.000 | 614 | 0 | 1266 | -1266 | 0 | 0 | none |
| qwen3.6_35b_a3b_akashml_no_reasoning | 24/1880 | 18 | 0.957 | 619 | 4 | 1257 | -1253 | 0 | 0 | AkashML:24 |
| qwen3.6_35b_a3b_akashml_reasoning_medium | 24/1880 | 18 | 0.957 | 620 | 3 | 1257 | -1254 | 1 | 1 | AkashML:24 |
