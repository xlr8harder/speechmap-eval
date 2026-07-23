# Gold v2 Qualification Summary

| model_key | rows | correct | accuracy_pct | binary_complete | false_complete | missed_complete | net_complete_bias | errors | content_filters | providers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| z-ai_glm-5.2_reasoning_medium | 397/397 | 349 | 87.909 | 363 | 29 | 5 | 24 | 1 | 1 | AkashML:18, Alibaba:22, Baidu:31, Cloudflare:16, DekaLLM:59, Fireworks:41, Friendli:32, Inceptron:47, Parasail:15, Phala:5, SiliconFlow:27, Wafer:36, WandB:30, Z.AI:18 |
| qwen_qwen3.5-397b-a17b_reasoning_medium | 397/397 | 334 | 84.131 | 358 | 17 | 22 | -5 | 0 | 0 | Novita:397 |
| qwen_qwen3.6-27b_reasoning_medium | 397/397 | 342 | 86.146 | 352 | 33 | 12 | 21 | 0 | 0 | Alibaba:397 |
| openai_gpt-5.5_reasoning_medium | 397/397 | 359 | 90.428 | 373 | 13 | 11 | 2 | 0 | 0 | Azure:38, OpenAI:359 |
