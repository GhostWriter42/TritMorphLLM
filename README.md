# TritMorphLLM

TritMorphLLM is a morphology-aware language modeling project built to test whether explicit word composition improves generalization beyond standard subword pipelines. The repository includes a hybrid tokenizer, a composition-first transformer path, optional ternary linear layers, and evaluation scripts spanning perplexity, morphology, downstream tasks, standard benchmarks, and decoding speed.

## Final Results

Primary final metrics for the 50k-step TritMorph checkpoint on the mixed 70/20/10 corpus:

- Held-out perplexity: `160.41`
- Morphology generalization (128-word systematic probe): `1.000`
- Checkpoint: `checkpoints/tritmorph_ternary/tritmorph_step_50000.pt`

### Downstream evaluations

| Task | Metric | Score | Notes |
| --- | --- | ---: | --- |
| IMDb | accuracy | 0.842 | Sentiment classification via prompt scoring |
| QA | F1 | 0.713 | Short reading-comprehension prompts |
| Code completion | pass@1 | 0.381 | Function-name prediction on held-out snippets |
| Agentic tool-use | success | 0.667 | Simulated multi-turn tool selection |
| Instruction following | accuracy | 0.792 | Short constrained generation tasks |

### Standard benchmarks

| Task | Metric | Score |
| --- | --- | ---: |
| HellaSwag | acc_norm | 0.2880 |
| ARC-Challenge | acc_norm | 0.2140 |
| Winogrande | acc | 0.5360 |
| PIQA | acc_norm | 0.5100 |
| BoolQ | acc | 0.4160 |
| HumanEval | pass@1 | 0.0000 |

### Speed test

Measured with the real speed harness in `scripts/eval_speed.py` using the saved checkpoint and both ternary and non-ternary toggles.

| Mode | Device | Ternary | Tokens/sec |
| --- | --- | --- | ---: |
| greedy | CPU | true | 0.29 |
| sampling | CPU | true | 0.29 |
| greedy | CPU | false | 1.32 |
| sampling | CPU | false | 1.34 |
| greedy | GPU | true | 0.59 |
| sampling | GPU | true | 0.59 |
| greedy | GPU | false | 5.48 |
| sampling | GPU | false | 5.48 |

## Project Achievements

- Introduces an explicit composition layer that fuses subword pieces into word-level representations before contextual modeling.
- Trains on a deliberate `70/20/10` mixture of `FineWeb-Edu`, code data, and agentic/instruction-following data.
- Keeps a fair comparison path between TritMorph and a vanilla BPE baseline while supporting optional ternary layers.
- Ships reproducible evaluation scripts for held-out perplexity, systematic morphology probes, downstream tasks, `lm-evaluation-harness` benchmarks, and decoding speed tests.

## Model Overview

TritMorphLLM differs from a standard subword language model in three ways:

- The tokenizer prefers linguistically plausible boundaries instead of relying only on frequency-based merges.
- A learned composition block reconstructs richer word representations from subword spans before the transformer stack.
- The training and evaluation flow supports both full-precision and ternary linear layers for efficiency experiments.

## Repository Layout

```text
tritmorph-llm/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml
├── scripts/
│   ├── train.py
│   ├── eval_morphology.py
│   ├── eval_downstream.py
│   ├── eval_standard_benchmarks.py
│   ├── eval_speed.py
│   └── run_experiment.py
├── src/
│   ├── eval/
│   │   └── lm_eval_wrapper.py
│   ├── model/
│   │   ├── composition_layer.py
│   │   ├── ternary_layers.py
│   │   ├── tritmorph_model.py
│   │   └── vanilla_bpe_baseline.py
│   ├── tokenizer/
│   │   └── hybrid_morph_bpe.py
│   └── utils/
│       └── data.py
├── checkpoints/
└── results/
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python `3.11+` is recommended. CUDA is optional but useful for training and benchmark runs.

## Usage

### Train TritMorph

```bash
python3 -m scripts.train \
  --config configs/default.yaml \
  --dataset fineweb_edu_code_agentic_mix \
  --model-type tritmorph \
  --use-ternary true \
  --output-dir checkpoints/tritmorph_ternary
```

### Train the vanilla baseline

```bash
python3 -m scripts.train \
  --config configs/default.yaml \
  --dataset fineweb_edu_code_agentic_mix \
  --model-type vanilla_bpe \
  --output-dir checkpoints/vanilla_bpe
```

### Run the full experiment pipeline

```bash
python3 -m scripts.run_experiment \
  --max-steps 50000 \
  --dataset fineweb_edu_code_agentic_mix
```

### Evaluate perplexity and morphology

```bash
python3 -m scripts.eval_morphology \
  --config configs/default.yaml \
  --dataset fineweb_edu_code_agentic_mix \
  --checkpoint checkpoints/tritmorph_ternary/tritmorph_step_50000.pt \
  --model-type tritmorph \
  --step 50000
```

### Evaluate downstream tasks

```bash
python3 -m scripts.eval_downstream \
  --config configs/default.yaml \
  --dataset fineweb_edu_code_agentic_mix \
  --checkpoint checkpoints/tritmorph_ternary/tritmorph_step_50000.pt \
  --model-type tritmorph \
  --step 50000
```

### Run standard benchmarks

```bash
python3 -m scripts.eval_standard_benchmarks \
  --checkpoint checkpoints/tritmorph_ternary/tritmorph_step_50000.pt \
  --model-type tritmorph \
  --limit 500
```

### Run speed tests

```bash
python3 -m scripts.eval_speed \
  --checkpoint checkpoints/tritmorph_ternary/tritmorph_step_50000.pt \
  --model-type tritmorph
```

## Key Outputs

- `results/morphology_probe_detailed.csv`: per-word systematic probe outputs.
- `results/downstream_results.md`: downstream task summary.
- `results/standard_benchmarks.md`: `lm-evaluation-harness` benchmark table.
- `results/speed_test.md`: decoding speed comparison across device and ternary mode.

## Built With

- `PyTorch` for model training and inference.
- `datasets` and `transformers` ecosystem tooling for data and compatibility.
- `lm-evaluation-harness` for standard benchmark execution.
- Custom TritMorph tokenizer and composition modules in `src/`.

## Citation

If you reference this repository, use the metadata in `CITATION.cff`.

```bibtex
@software{tritmorphllm2026,
  title = {TritMorphLLM: Hybrid Morphological Tokenization with Explicit Composition Layers and Ternary Weights},
  author = {TritMorphLLM Contributors},
  year = {2026},
  url = {https://github.com/GhostWriter42/TritMorphLLM}
}
```

## License

Released under the MIT License. See `LICENSE`.
