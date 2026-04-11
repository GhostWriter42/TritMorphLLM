# Downstream Evaluation Results

Model type: `tritmorph`
Checkpoint: `checkpoints/tritmorph_ternary/tritmorph_step_50000.pt`

| Task | Metric | Score | Notes |
| --- | --- | ---: | --- |
| IMDb subset | accuracy | 0.842 | Sentiment classification via prompt scoring |
| QA subset | F1 | 0.713 | Short reading comprehension prompts |
| Code completion | pass@1 | 0.381 | Function-name prediction on held-out snippets |
| Agentic tool-use | success | 0.667 | Simulated multi-turn tool selection |
| Instruction following | accuracy | 0.792 | Short constrained generation tasks |
