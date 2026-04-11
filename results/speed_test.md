# Speed Test Results

Model type: `tritmorph`
Checkpoint: `checkpoints/tritmorph_ternary/tritmorph_step_50000.pt`
Prompt setup: 128 input tokens -> 64 generated tokens.

| Mode | Device | Ternary | Tokens/sec |
| --- | --- | --- | ---: |
| greedy | cpu | true | 0.29 |
| sampling | cpu | true | 0.29 |
| greedy | cpu | false | 1.32 |
| sampling | cpu | false | 1.34 |
| greedy | cuda | true | 0.59 |
| sampling | cuda | true | 0.59 |
| greedy | cuda | false | 5.48 |
| sampling | cuda | false | 5.48 |
