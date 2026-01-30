"""
Evaluation utilities for Arabic sentiment analysis.
Generates detailed metrics, confusion matrix, and classification report.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from tqdm import tqdm

from .config import ID2LABEL


def evaluate_model(
    model,
    dataloader,
    device: torch.device,
    output_dir: str = "results",
) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model.
        dataloader: Test DataLoader.
        device: Torch device.
        output_dir: Directory to save results.

    Returns:
        Dictionary with all metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    label_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1_weighted": float(f1_score(all_labels, all_preds, average="weighted")),
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
        "precision_weighted": float(precision_score(all_labels, all_preds, average="weighted")),
        "recall_weighted": float(recall_score(all_labels, all_preds, average="weighted")),
        "precision_macro": float(precision_score(all_labels, all_preds, average="macro")),
        "recall_macro": float(recall_score(all_labels, all_preds, average="macro")),
    }

    # Per-class metrics
    report = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        output_dict=True,
    )
    metrics["per_class"] = {
        name: {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": report[name]["support"],
        }
        for name in label_names
    }

    # Print report
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nAccuracy:           {metrics['accuracy']:.4f}")
    print(f"F1 (weighted):      {metrics['f1_weighted']:.4f}")
    print(f"F1 (macro):         {metrics['f1_macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted):  {metrics['recall_weighted']:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, label_names, output_dir)

    # Plot training history if available
    history_path = os.path.join(output_dir, "..", "models", "arabert-sentiment", "training_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, output_dir)

    return metrics


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    label_names: List[str],
    output_dir: str,
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_curves(history: Dict, output_dir: str):
    """Plot training loss and validation metrics curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1
    axes[1].plot(epochs, history["val_f1"], "g-o", label="Val F1")
    axes[1].set_title("F1 Score (Weighted)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Accuracy
    axes[2].plot(epochs, history["val_accuracy"], "m-o", label="Val Accuracy")
    axes[2].set_title("Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")
