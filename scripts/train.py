#!/usr/bin/env python3
"""
Training script for Arabic Sentiment Analysis model.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 10 --batch_size 16 --lr 3e-5
"""
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random
from transformers import AutoTokenizer

from src.config import ModelConfig, TrainingConfig, DataConfig
from src.preprocessing import ArabicTextPreprocessor
from src.dataset import load_combined_dataset, create_dataloaders
from src.model import AraBERTSentimentClassifier
from src.trainer import SentimentTrainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Arabic Sentiment Model")
    parser.add_argument("--model_name", type=str, default="aubmindlab/bert-base-arabertv2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="models/arabert-sentiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_mlp_head", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("Arabic Sentiment Analysis - Training Pipeline")
    print("=" * 60)

    # Configs
    model_config = ModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
    )
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
    data_config = DataConfig(max_samples=args.max_samples)

    # Preprocessing
    print("\n1. Initializing preprocessor...")
    preprocessor = ArabicTextPreprocessor(model_name=args.model_name)

    # Load data
    print("\n2. Loading datasets...")
    data_splits = load_combined_dataset(data_config, preprocessor)

    # Tokenizer
    print("\n3. Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # DataLoaders
    print("\n4. Creating dataloaders...")
    loaders = create_dataloaders(
        data_splits, tokenizer, model_config, training_config
    )

    # Model
    print("\n5. Loading AraBERT model...")
    model = AraBERTSentimentClassifier(
        model_name=args.model_name,
        num_labels=model_config.num_labels,
        dropout=model_config.dropout,
        use_mlp_head=args.use_mlp_head,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Train
    print("\n6. Starting training...")
    trainer = SentimentTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=training_config,
    )
    history = trainer.train()

    print("\n✅ Training complete!")
    print(f"Best model saved to: {args.output_dir}/best_model.pt")
    print(f"Training history saved to: {args.output_dir}/training_history.json")


if __name__ == "__main__":
    main()
