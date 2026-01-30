"""
Training loop for Arabic sentiment analysis model.
"""
import os
import json
import time
from typing import Dict, Optional

import torch
import numpy as np
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from .config import TrainingConfig, ID2LABEL


class SentimentTrainer:
    """Custom trainer for Arabic sentiment model."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config: TrainingConfig,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device setup
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_params = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_params,
            lr=config.learning_rate,
        )

        # Scheduler
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision
        self.use_amp = config.fp16 and self.device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_accuracy": [],
        }
        self.best_f1 = 0.0

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            leave=True,
        )

        for batch in progress:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast("cuda"):
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs["loss"]
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss/num_batches:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, attention_mask, labels)
            total_loss += outputs["loss"].item()

            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        accuracy = accuracy_score(all_labels, all_preds)

        return {
            "loss": avg_loss,
            "f1": f1,
            "accuracy": accuracy,
        }

    def train(self) -> Dict:
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")

        os.makedirs(self.config.output_dir, exist_ok=True)
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Evaluate
            val_metrics = self.evaluate()

            # Log
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val F1:     {val_metrics['f1']:.4f}")
            print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])

            # Save best model
            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                save_path = os.path.join(self.config.output_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "f1": self.best_f1,
                    "config": {
                        "model_name": self.model.config.name_or_path,
                        "num_labels": self.model.num_labels,
                    },
                }, save_path)
                print(f"  ✓ Saved best model (F1: {self.best_f1:.4f})")

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete in {elapsed/60:.1f} minutes")
        print(f"Best Val F1: {self.best_f1:.4f}")
        print(f"{'='*60}\n")

        # Save training history
        history_path = os.path.join(self.config.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
