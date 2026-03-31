# Experiment Results

Generated for `max_steps=10000`.
Dataset preset: `wikitext103`.

| Model | Held-out Perplexity | Morph Acc | Training Time | GPU Memory (MB) | Checkpoint |
| --- | ---: | ---: | ---: | ---: | --- |
| TritMorph (ternary) | 680.73 | 1.000 | 0m 0s | 0 | `checkpoints/phase5_preview/tritmorph_step_10000.pt` |
| Vanilla BPE | 651.21 | 0.000 | 0m 0s | 0 | `/home/aaruss/Projects/AI/TritMorphLLM/tritmorph-llm/checkpoints/vanilla_bpe/vanilla_bpe_step_10000.pt` |

## Summary

TritMorph vs Vanilla: Perplexity 680.73->651.21 | Morph Gen 100.0%->0.0%

## Raw Logs

- TritMorph train log: `/home/aaruss/Projects/AI/TritMorphLLM/tritmorph-llm/results/tritmorph_train.log`
- TritMorph eval log: `/home/aaruss/Projects/AI/TritMorphLLM/tritmorph-llm/results/tritmorph_eval.log`
- Vanilla train log: `/home/aaruss/Projects/AI/TritMorphLLM/tritmorph-llm/results/vanilla_bpe_train.log`
- Vanilla eval log: `/home/aaruss/Projects/AI/TritMorphLLM/tritmorph-llm/results/vanilla_bpe_eval.log`
