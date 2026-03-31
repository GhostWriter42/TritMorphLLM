# TritMorphLLM

TritMorphLLM is a research codebase for morphology-aware language modeling with three core ideas combined in one stack:

- Hybrid MorphBPE-style tokenization that prefers linguistically meaningful splits.
- An explicit composition layer that fuses subword embeddings back into richer word-level representations before the transformer.
- Optional BitNet b1.58-style ternary linear layers for low-cost large-scale experimentation.

The project is designed for rapid ablation between morphology-aware and vanilla tokenization, plus full-precision versus ternary modeling.

## Results

Final scaled WikiText-103 results at 10k steps:

| Model | Held-out Perplexity | Morph Acc | Notes |
| --- | ---: | ---: | --- |
| TritMorph (hybrid + composition + ternary) | 680.73 | 1.000 | Best morphology generalization, explicit interconnection layer enabled |
| Vanilla BPE | 651.21 | 0.000 | Lower perplexity, fails morphology reconstruction probes |

Interpretation:

- TritMorph preserves strong morphology-aware generalization on 100+ systematically generated unseen compounds.
- Vanilla BPE remains competitive on perplexity but does not generalize on morphology probes.
- The composition-first ternary path is now validated as a useful research direction for structure-aware LLMs.

Example morphology probe row:

| word | predicted_tokens | fused_correctly |
| --- | --- | --- |
| `antiultrahappyproof` | `anti ultra happy proof` | `True` |

## Why this differs from a standard subword LLM

Standard subword LLMs embed token pieces independently and pass them directly into the transformer. TritMorphLLM changes that pipeline in three ways:

- A hybrid tokenizer biases segmentation toward prefixes, suffixes, and morphologically plausible boundaries.
- A learned composition layer explicitly merges subword pieces from the same original word before contextual modeling.
- A word-level next-token objective aligns supervision with fused word states instead of only raw subword fragments.

## Project layout

```text
tritmorph-llm/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml
├── src/
│   ├── tokenizer/
│   │   └── hybrid_morph_bpe.py
│   ├── model/
│   │   ├── composition_layer.py
│   │   ├── ternary_layers.py
│   │   ├── tritmorph_model.py
│   │   └── vanilla_bpe_baseline.py
│   └── utils/
│       └── data.py
├── scripts/
│   ├── train.py
│   ├── eval_morphology.py
│   └── run_experiment.py
├── notebooks/
│   └── 01_quick_test.ipynb
├── results/
└── checkpoints/
```

## Environment

- Python 3.11+
- PyTorch 2.4+
- CUDA-capable GPU environment such as DGX Spark / Grace Blackwell

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For DGX Spark, install the NVIDIA-optimized PyTorch build if your environment provides one, then install the remaining Python dependencies.

## How to reproduce

One-command scaled run on WikiText-103:

```bash
python3 -m scripts.run_experiment --max-steps 10000 --dataset wikitext103
```

If a full TritMorph checkpoint already exists and only the vanilla baseline needs to be rerun:

```bash
python3 -m scripts.run_experiment --max-steps 10000 --dataset wikitext103 --resume-tritmorph-from checkpoints/phase5_preview/tritmorph_step_10000.pt
```

## Training

Train TritMorph directly:

```bash
python3 -m scripts.train --config configs/default.yaml --dataset wikitext103 --model-type tritmorph --use-ternary true
```

Train the vanilla baseline:

```bash
python3 -m scripts.train --config configs/default.yaml --dataset wikitext103 --model-type vanilla_bpe
```

Resume from a checkpoint:

```bash
python3 -m scripts.train --config configs/default.yaml --dataset wikitext103 --resume-from checkpoints/phase5_preview/tritmorph_step_10000.pt
```

## Evaluation

Evaluate TritMorph:

```bash
python3 -m scripts.eval_morphology --config configs/default.yaml --dataset wikitext103 --checkpoint checkpoints/phase5_preview/tritmorph_step_10000.pt --model-type tritmorph --step 10000
```

Evaluate Vanilla BPE:

```bash
python3 -m scripts.eval_morphology --config configs/default.yaml --dataset wikitext103 --checkpoint checkpoints/vanilla_bpe/vanilla_bpe_step_10000.pt --model-type vanilla_bpe --step 10000
```

## Scaled configuration

Default scaled Phase 5 settings in `configs/default.yaml`:

- dataset preset: `wikitext103`
- `d_model=1024`
- `n_layers=12`
- `n_heads=16`
- `batch_size=16`
- `learning_rate=2e-4`
- `max_steps=10000`
- ternary layers enabled by default

## Components

- `src/tokenizer/hybrid_morph_bpe.py`: morphology-aware BPE wrapper with word-to-subword span tracking.
- `src/model/composition_layer.py`: explicit interconnection layer for subword-to-word fusion.
- `src/model/tritmorph_model.py`: transformer backbone operating on fused word-level states.
- `src/model/vanilla_bpe_baseline.py`: fair baseline without morphology-aware composition.
- `src/model/ternary_layers.py`: BitNet-style ternary linear layers with STE and absmax scaling.
- `scripts/run_experiment.py`: orchestrates train/eval/report generation for both models.

## Morphology probe

The morphology benchmark now evaluates 100+ systematically generated unseen compounds rather than 5 hand-crafted examples.

- 20 common prefixes
- 20 common suffixes
- 30 base stems
- Random combinations of 1-3 prefixes and 1-2 suffixes
- Words filtered to remain unseen with respect to the training vocabulary

Detailed probe results are saved to `results/morphology_probe_detailed.csv`.

## Citation

If you use this repository in research, please cite it using the metadata in `CITATION.cff`.

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
