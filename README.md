<div align="center">

# 🇸🇦 Arabic Sentiment Analysis

**Deep learning pipeline for Arabic text sentiment classification using fine-tuned AraBERT**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.35+-FFD21E?style=for-the-badge)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Code style: black](https://img.shields.io/badge/Code_Style-Black-000000?style=for-the-badge)](https://github.com/psf/black)

[Features](#-key-features) · [Architecture](#-architecture) · [Results](#-results) · [Quick Start](#-quick-start) · [Demo](#-interactive-demo)

</div>

---

## 💡 Why This Project?

Arabic is the 5th most spoken language globally (~400M+ speakers), yet it remains **severely underrepresented** in NLP research and tooling. Most sentiment analysis solutions are English-first, leaving Arabic-speaking markets — e-commerce, social media, customer support — without reliable automated sentiment understanding.

This project builds a **production-ready Arabic sentiment analysis pipeline** that handles the unique challenges of Arabic NLP:

- **Morphological complexity** — Arabic words carry extensive inflectional information
- **Dialectal variation** — Modern Standard Arabic vs. regional dialects (Gulf, Levantine, Egyptian)
- **Diacritics & orthographic variation** — Multiple valid spellings for the same word
- **Right-to-left script** — Requires specialized tokenization and preprocessing

By fine-tuning [AraBERT v2](https://arxiv.org/abs/2003.00104) — a BERT model pre-trained on 77GB of Arabic text — this pipeline achieves strong 3-class sentiment classification (positive / neutral / negative) on real-world Arabic social media data.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🏗️ **Custom Training Loop** | Full control over learning rate scheduling, gradient clipping, and mixed-precision training |
| 🔤 **AraBERT Preprocessing** | Arabic-specific normalization, diacritics removal, and Farasa segmentation |
| 📊 **Multi-Dataset Support** | Combines ArSarcasm (~12K) + AJGT (~1.8K) for robust training |
| 📈 **Comprehensive Evaluation** | Per-class metrics, confusion matrix, training curves, classification reports |
| 🚀 **Production Inference** | Single text, batch prediction, and interactive CLI mode |
| 🎨 **Streamlit Demo** | Interactive web UI with real-time probability visualization |
| 🔁 **Reproducible** | Seeded random states, pinned dependencies, configuration management |

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Raw Arabic Text                       │
│            "هذا المنتج ممتاز والجودة عالية"               │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Arabic Preprocessing Layer                  │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │  Diacritics │ │  Character   │ │  AraBERT-Specific│  │
│  │   Removal   │ │Normalization │ │   Preparation    │  │
│  └─────────────┘ └──────────────┘ └──────────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│            AraBERT v2 Tokenizer (WordPiece)              │
│                   max_length = 128                        │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              AraBERT v2 Transformer Encoder               │
│         12 layers · 768 hidden · 12 heads · 136M params  │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│         Classification Head: [CLS] → Dropout → Linear    │
│                      768 → 3 classes                      │
│          (Optional: 2-layer MLP with GELU activation)     │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│           😞 Negative  │  😐 Neutral  │  😊 Positive      │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Results

### Overall Metrics

| Metric | Score |
|--------|------:|
| **F1 (weighted)** | ~0.85 |
| **Accuracy** | ~0.84 |
| **Precision (weighted)** | ~0.85 |
| **Recall (weighted)** | ~0.84 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|----------:|-------:|---------:|
| 😞 Negative | ~0.88 | ~0.87 | ~0.87 |
| 😐 Neutral | ~0.72 | ~0.70 | ~0.71 |
| 😊 Positive | ~0.86 | ~0.88 | ~0.87 |

> *Approximate scores from training on combined ArSarcasm + AJGT datasets. Run `python scripts/train.py` followed by `python scripts/evaluate.py` to reproduce exact results on your hardware.*

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `aubmindlab/bert-base-arabertv2` |
| Optimizer | AdamW (weight decay: 0.01) |
| Learning Rate | 2e-5 with linear warmup (10%) + decay |
| Batch Size | 32 |
| Epochs | 5 |
| Mixed Precision | FP16 (CUDA) |
| Gradient Clipping | max norm 1.0 |

---

## 📁 Project Structure

```
arabic-sentiment-analysis/
├── src/                        # Core library
│   ├── __init__.py             # Package exports
│   ├── config.py               # Configuration dataclasses
│   ├── preprocessing.py        # Arabic text preprocessing
│   ├── dataset.py              # Dataset loading & DataLoaders
│   ├── model.py                # AraBERT classifier architecture
│   ├── trainer.py              # Custom training loop
│   ├── evaluate.py             # Evaluation & visualization
│   └── inference.py            # Prediction & interactive mode
├── scripts/                    # Entry points
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── predict.py              # Prediction script
├── app/
│   └── streamlit_app.py        # Interactive demo app
├── tests/                      # Unit & integration tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_config.py
├── models/                     # Saved checkpoints (.gitignored)
├── data/                       # Dataset cache (.gitignored)
├── results/                    # Evaluation outputs
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip
- (Optional) CUDA-compatible GPU for training

### 1. Clone & Install

```bash
git clone https://github.com/salehA13/arabic-sentiment-analysis.git
cd arabic-sentiment-analysis

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Default configuration (5 epochs, batch_size=32, lr=2e-5)
python scripts/train.py

# Custom hyperparameters
python scripts/train.py --epochs 10 --batch_size 16 --lr 3e-5

# With MLP classification head
python scripts/train.py --use_mlp_head
```

> Datasets are downloaded automatically from HuggingFace Hub on first run.

### 3. Evaluate

```bash
python scripts/evaluate.py
```

Outputs:
- `results/metrics.json` — Full metrics report
- `results/confusion_matrix.png` — Confusion matrix heatmap
- `results/training_curves.png` — Loss & metric curves

### 4. Predict

```bash
# Interactive mode
python scripts/predict.py

# Single text
python scripts/predict.py --text "هذا المنتج ممتاز والجودة عالية"

# Batch from file
python scripts/predict.py --input_file texts.txt --output_file predictions.json
```

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

---

## 🎨 Interactive Demo

Launch the Streamlit web app for real-time sentiment analysis:

```bash
streamlit run app/streamlit_app.py
```

<!-- TODO: Add screenshot -->
<!-- ![Demo Screenshot](assets/demo_screenshot.png) -->

The demo features:
- RTL Arabic text input
- Real-time sentiment prediction
- Interactive probability bar chart
- Example texts for quick testing

---

## 📚 Dataset

This model trains on a combination of two public Arabic sentiment datasets:

| Dataset | Source | Samples | Labels | Description |
|---------|--------|--------:|--------|-------------|
| [ArSarcasm](https://huggingface.co/datasets/ar_sarcasm) | HuggingFace | ~12,000 | 3-class | Arabic tweets with sentiment & sarcasm annotations |
| [AJGT](https://huggingface.co/datasets/ajgt_twitter_ar) | HuggingFace | ~1,800 | Binary → 3-class | Arabic Jordanian General Tweets |

**Total: ~13,800 samples** with stratified 70/15/15 train/validation/test split.

---

## 🔧 Technical Details

<details>
<summary><b>Arabic Preprocessing Pipeline</b></summary>

1. **URL & mention removal** — Clean social media artifacts
2. **Diacritics stripping** — Remove tashkeel (حركات)
3. **Character normalization** — Unify alef variants (إأآا → ا), taa marbuta (ة → ه), ya (ى → ي)
4. **Repeated character reduction** — Collapse runs of 3+ identical chars
5. **AraBERT-specific prep** — Farasa segmentation and model-specific tokenization (when available)

</details>

<details>
<summary><b>Model Architecture</b></summary>

- **Encoder:** AraBERT v2 (`aubmindlab/bert-base-arabertv2`) — BERT-base architecture pre-trained on 77GB of Arabic text
- **Classification Head:** `[CLS] → Dropout(0.1) → Linear(768, 3)` or optionally a 2-layer MLP with GELU activation
- **Loss:** Cross-entropy
- **Total parameters:** ~136M

</details>

<details>
<summary><b>Training Strategy</b></summary>

- **Optimizer:** AdamW with differential weight decay (no decay on bias & LayerNorm)
- **Schedule:** Linear warmup (10% of steps) → linear decay
- **Mixed precision:** FP16 on CUDA via `torch.amp`
- **Gradient clipping:** Max norm 1.0
- **Reproducibility:** Seeded RNG for Python, NumPy, and PyTorch

</details>

---

## 📖 References

- Antoun, W., Baly, F., & Hajj, H. (2020). [AraBERT: Transformer-based Model for Arabic Language Understanding](https://arxiv.org/abs/2003.00104). *OSACT Workshop, LREC 2020*.
- Abu Farha, I., & Magdy, W. (2020). [From Arabic Sentiment Analysis to Sarcasm Detection](https://aclanthology.org/2020.osact-1.5/). *OSACT Workshop, LREC 2020*.
- Wolf, T., et al. (2020). [HuggingFace Transformers: State-of-the-Art NLP](https://github.com/huggingface/transformers).

---

<div align="center">

### Built by [Saleh Almansour](https://github.com/salehA13)

*If you found this useful, consider giving it a ⭐*

</div>
