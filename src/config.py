"""
Configuration for Arabic Sentiment Analysis Pipeline.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "aubmindlab/bert-base-arabertv2"
    num_labels: int = 3  # positive, negative, neutral
    max_length: int = 128
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "models/arabert-sentiment"
    num_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    fp16: bool = True
    gradient_accumulation_steps: int = 1

@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "ar_sarcasm"  # HuggingFace dataset
    data_dir: str = "data"
    test_size: float = 0.15
    val_size: float = 0.15
    max_samples: Optional[int] = None  # None = use all

# Label mappings
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Arabic preprocessing
ARABIC_DIACRITICS = r'[\u0617-\u061A\u064B-\u0652]'

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
