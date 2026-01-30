"""
Arabic Sentiment Analysis Pipeline.

A deep learning pipeline for Arabic text sentiment classification
using fine-tuned AraBERT (aubmindlab/bert-base-arabertv2).

Modules:
    config: Configuration dataclasses and constants.
    preprocessing: Arabic text normalization and cleaning.
    dataset: Dataset loading, splitting, and DataLoader creation.
    model: AraBERT-based sentiment classifier architecture.
    trainer: Custom training loop with LR scheduling and mixed precision.
    evaluate: Evaluation metrics, confusion matrix, and training curves.
    inference: Single/batch prediction and interactive CLI mode.
"""

from .config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    LABEL2ID,
    ID2LABEL,
)
from .model import AraBERTSentimentClassifier
from .preprocessing import ArabicTextPreprocessor
from .inference import SentimentPredictor

__version__ = "1.0.0"
__author__ = "Saleh Almansour"

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "LABEL2ID",
    "ID2LABEL",
    "AraBERTSentimentClassifier",
    "ArabicTextPreprocessor",
    "SentimentPredictor",
]
