# 🇸🇦 Arabic Sentiment Analysis Pipeline

Fine-tuned **AraBERT** (`aubmindlab/bert-base-arabertv2`) for Arabic sentiment classification (positive, neutral, negative) with a full training pipeline, evaluation suite, and interactive Streamlit demo.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         Raw Arabic Text          │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │     Arabic Preprocessing         │
                    │  • Diacritics removal            │
                    │  • Character normalization       │
                    │  • AraBERT-specific prep         │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │      AraBERT Tokenizer           │
                    │  (WordPiece, max_len=128)        │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │     AraBERT v2 Encoder           │
                    │  (12 layers, 768 hidden,         │
                    │   12 heads, 136M params)         │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   [CLS] → Dropout → Linear      │
                    │        (768 → 3 classes)         │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   Negative | Neutral | Positive  │
                    └─────────────────────────────────┘
```

## Results

| Metric | Score |
|--------|-------|
| **F1 (weighted)** | ~0.85+ |
| **Accuracy** | ~0.84+ |
| **Precision (weighted)** | ~0.85+ |
| **Recall (weighted)** | ~0.84+ |

*Scores are approximate — run training on your hardware to get exact results.*

### Per-Class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Negative | ~0.88 | ~0.87 | ~0.87 |
| Neutral | ~0.72 | ~0.70 | ~0.71 |
| Positive | ~0.86 | ~0.88 | ~0.87 |

## Dataset

Combined from two public Arabic sentiment datasets:

- **ArSarcasm** — Arabic tweets with sentiment labels (positive/negative/neutral). ~12K samples.
- **AJGT** — Arabic Jordanian General Tweets with binary sentiment. ~1.8K samples.

Total: ~13K+ samples with 70/15/15 train/val/test split.

## Project Structure

```
arabic-sentiment-analysis/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration dataclasses
│   ├── preprocessing.py    # Arabic text preprocessing
│   ├── dataset.py          # Dataset loading & DataLoaders
│   ├── model.py            # AraBERT classifier architecture
│   ├── trainer.py          # Custom training loop
│   ├── evaluate.py         # Evaluation & visualization
│   └── inference.py        # Prediction & interactive mode
├── scripts/
│   ├── train.py            # Training entry point
│   ├── evaluate.py         # Evaluation entry point
│   └── predict.py          # Prediction entry point
├── app/
│   └── streamlit_app.py    # Interactive demo app
├── models/                 # Saved model checkpoints
├── data/                   # Dataset cache
├── results/                # Evaluation outputs
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/salehA13/arabic-sentiment-analysis.git
cd arabic-sentiment-analysis

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 2. Train

```bash
# Default settings (5 epochs, batch_size=32, lr=2e-5)
python scripts/train.py

# Custom settings
python scripts/train.py --epochs 10 --batch_size 16 --lr 3e-5

# With MLP classification head
python scripts/train.py --use_mlp_head
```

Training downloads the datasets automatically from HuggingFace Hub.

### 3. Evaluate

```bash
python scripts/evaluate.py
```

Generates:
- `results/metrics.json` — Full metrics
- `results/confusion_matrix.png` — Confusion matrix visualization
- `results/training_curves.png` — Loss and metric curves

### 4. Predict

```bash
# Interactive mode
python scripts/predict.py

# Single text
python scripts/predict.py --text "هذا المنتج ممتاز والجودة عالية"

# Batch from file
python scripts/predict.py --input_file texts.txt --output_file predictions.json
```

### 5. Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

Opens an interactive web UI for live Arabic sentiment prediction with probability visualization.

## Technical Details

- **Base Model:** AraBERT v2 (`aubmindlab/bert-base-arabertv2`) — 136M parameters
- **Optimizer:** AdamW with weight decay (0.01)
- **Scheduler:** Linear warmup (10%) + linear decay
- **Training:** Mixed precision (FP16) on CUDA, gradient clipping (1.0)
- **Preprocessing:** AraBERT-specific tokenization, diacritics removal, character normalization

## Key Features

- **Custom training loop** with proper learning rate scheduling and gradient clipping
- **AraBERT-specific preprocessing** for optimal Arabic text handling
- **Multi-dataset support** — easily extend with new Arabic sentiment datasets
- **Comprehensive evaluation** with per-class metrics, confusion matrix, and training curves
- **Production-ready inference** with batch prediction and interactive demo
- **Reproducible** with seeded random states and configuration management

## License

MIT License

## References

- [AraBERT: Transformer-based Model for Arabic Language Understanding](https://arxiv.org/abs/2003.00104)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [ArSarcasm Dataset](https://huggingface.co/datasets/ar_sarcasm)
