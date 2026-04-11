# Standard Benchmark Results

Model type: `tritmorph`
Checkpoint: `checkpoints/tritmorph_ternary/tritmorph_step_50000.pt`
Task limit: `500` examples per task.
Generation cap for generative tasks: `16` tokens.

| Task | Metric | Score |
| --- | --- | ---: |
| hellaswag | acc_norm | 0.2880 |
| arc_challenge | acc_norm | 0.2140 |
| winogrande | acc | 0.5360 |
| piqa | acc_norm | 0.5100 |
| boolq | acc | 0.4160 |
| humaneval | pass@1 | 0.0000 |
