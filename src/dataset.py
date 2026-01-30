"""
Dataset loading and preparation for Arabic sentiment analysis.
Supports multiple Arabic sentiment datasets from HuggingFace.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from .config import DataConfig, ModelConfig, LABEL2ID
from .preprocessing import ArabicTextPreprocessor


class ArabicSentimentDataset(Dataset):
    """PyTorch Dataset for Arabic sentiment analysis."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_ajgt_dataset() -> Tuple[list, list]:
    """
    Load Arabic Jordanian General Tweets (AJGT) dataset.
    Binary sentiment: positive/negative.
    """
    try:
        dataset = load_dataset("ajgt_twitter_ar", trust_remote_code=True)
        texts = dataset["train"]["text"]
        # AJGT labels: 0=negative, 1=positive -> remap to our schema
        labels = [0 if l == 0 else 2 for l in dataset["train"]["label"]]
        return texts, labels
    except Exception as e:
        print(f"Could not load AJGT dataset: {e}")
        return [], []


def load_ar_reviews() -> Tuple[list, list]:
    """
    Load Arabic book/hotel/product reviews dataset.
    Uses labr (Large-scale Arabic Book Reviews) from HuggingFace.
    """
    try:
        dataset = load_dataset("labr", trust_remote_code=True)
        texts = []
        labels = []
        for split in ["train", "test"]:
            if split in dataset:
                for item in dataset[split]:
                    text = item["text"]
                    rating = item["label"]
                    # Map 1-5 star ratings to sentiment
                    if rating <= 1:  # 1-2 stars
                        labels.append(0)  # negative
                        texts.append(text)
                    elif rating == 2:  # 3 stars
                        labels.append(1)  # neutral
                        texts.append(text)
                    else:  # 4-5 stars
                        labels.append(2)  # positive
                        texts.append(text)
        return texts, labels
    except Exception as e:
        print(f"Could not load LABR dataset: {e}")
        return [], []


def load_ar_sarcasm() -> Tuple[list, list]:
    """
    Load ArSarcasm dataset (Arabic sarcasm and sentiment).
    Has direct sentiment annotations: positive, negative, neutral.
    """
    try:
        dataset = load_dataset("ar_sarcasm", trust_remote_code=True)
        texts = []
        labels = []

        label_map = {"negative": 0, "neutral": 1, "positive": 2}

        for split in ["train", "test"]:
            if split in dataset:
                for item in dataset[split]:
                    text = item["tweet"]
                    sentiment = item["sentiment"]
                    if sentiment in label_map:
                        texts.append(text)
                        labels.append(label_map[sentiment])

        return texts, labels
    except Exception as e:
        print(f"Could not load ArSarcasm dataset: {e}")
        return [], []


def load_combined_dataset(
    config: DataConfig,
    preprocessor: Optional[ArabicTextPreprocessor] = None,
) -> Dict[str, Tuple[list, list]]:
    """
    Load and combine multiple Arabic sentiment datasets.

    Returns:
        Dictionary with 'train', 'val', 'test' splits,
        each containing (texts, labels) tuple.
    """
    all_texts = []
    all_labels = []

    # Load ArSarcasm (primary - has 3-class sentiment)
    print("Loading ArSarcasm dataset...")
    texts, labels = load_ar_sarcasm()
    print(f"  ArSarcasm: {len(texts)} samples")
    all_texts.extend(texts)
    all_labels.extend(labels)

    # Load AJGT tweets
    print("Loading AJGT dataset...")
    texts, labels = load_ajgt_dataset()
    print(f"  AJGT: {len(texts)} samples")
    all_texts.extend(texts)
    all_labels.extend(labels)

    # Optionally limit samples
    if config.max_samples and len(all_texts) > config.max_samples:
        indices = np.random.RandomState(42).choice(
            len(all_texts), config.max_samples, replace=False
        )
        all_texts = [all_texts[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]

    # Preprocess
    if preprocessor:
        print("Preprocessing texts...")
        all_texts = preprocessor.preprocess_batch(all_texts)

    # Filter empty texts
    valid = [(t, l) for t, l in zip(all_texts, all_labels) if t.strip()]
    all_texts = [t for t, _ in valid]
    all_labels = [l for _, l in valid]

    print(f"\nTotal samples: {len(all_texts)}")
    for label_name, label_id in LABEL2ID.items():
        count = sum(1 for l in all_labels if l == label_id)
        print(f"  {label_name}: {count} ({count/len(all_labels)*100:.1f}%)")

    # Split: train / val / test
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        all_texts, all_labels,
        test_size=config.test_size + config.val_size,
        random_state=42,
        stratify=all_labels,
    )

    relative_val = config.val_size / (config.test_size + config.val_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=1 - relative_val,
        random_state=42,
        stratify=temp_labels,
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_texts)}")
    print(f"  Val:   {len(val_texts)}")
    print(f"  Test:  {len(test_texts)}")

    return {
        "train": (train_texts, train_labels),
        "val": (val_texts, val_labels),
        "test": (test_texts, test_labels),
    }


def create_dataloaders(
    data_splits: Dict,
    tokenizer,
    model_config: ModelConfig,
    training_config,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Create PyTorch DataLoaders for each split."""
    loaders = {}

    for split_name, (texts, labels) in data_splits.items():
        dataset = ArabicSentimentDataset(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            max_length=model_config.max_length,
        )

        shuffle = split_name == "train"
        batch_size = training_config.batch_size

        loaders[split_name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
        )

    return loaders
