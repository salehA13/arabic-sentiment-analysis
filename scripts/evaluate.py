#!/usr/bin/env python3
"""
Evaluation script for Arabic Sentiment Analysis model.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model_dir models/arabert-sentiment --output_dir results
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from src.config import ModelConfig, DataConfig, TrainingConfig
from src.preprocessing import ArabicTextPreprocessor
from src.dataset import load_combined_dataset, create_dataloaders
from src.model import AraBERTSentimentClassifier
from src.evaluate import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Arabic Sentiment Model")
    parser.add_argument("--model_dir", type=str, default="models/arabert-sentiment")
    parser.add_argument("--model_name", type=str, default="aubmindlab/bert-base-arabertv2")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Arabic Sentiment Analysis - Evaluation")
    print("=" * 60)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    print("\n1. Loading model...")
    checkpoint_path = os.path.join(args.model_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model not found at {checkpoint_path}")
        print("Run training first: python scripts/train.py")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    model = AraBERTSentimentClassifier(
        model_name=args.model_name,
        num_labels=config.get("num_labels", 3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Training F1: {checkpoint.get('f1', 'N/A')}")

    # Load test data
    print("\n2. Loading test data...")
    preprocessor = ArabicTextPreprocessor(model_name=args.model_name)
    data_config = DataConfig()
    data_splits = load_combined_dataset(data_config, preprocessor)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_config = ModelConfig(model_name=args.model_name)
    training_config = TrainingConfig(batch_size=args.batch_size)

    loaders = create_dataloaders(data_splits, tokenizer, model_config, training_config)

    # Evaluate
    print("\n3. Evaluating on test set...")
    metrics = evaluate_model(
        model=model,
        dataloader=loaders["test"],
        device=device,
        output_dir=args.output_dir,
    )

    print(f"\n✅ Evaluation complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
