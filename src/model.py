"""
Model definition for Arabic sentiment analysis using AraBERT.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class AraBERTSentimentClassifier(nn.Module):
    """
    AraBERT-based sentiment classifier with configurable classification head.

    Architecture:
        AraBERT encoder -> Dropout -> Linear (768 -> num_labels)

    Optionally uses a 2-layer MLP head for better representation learning.
    """

    def __init__(
        self,
        model_name: str = "aubmindlab/bert-base-arabertv2",
        num_labels: int = 3,
        dropout: float = 0.1,
        use_mlp_head: bool = False,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.use_mlp_head = use_mlp_head

        # Load pre-trained AraBERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size  # 768

        self.dropout = nn.Dropout(dropout)

        if use_mlp_head:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels),
            )
        else:
            self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional ground truth labels [batch_size]

        Returns:
            Dictionary with 'logits' and optionally 'loss'.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    def predict(self, input_ids, attention_mask):
        """Get predictions (class indices)."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(outputs["logits"], dim=-1)
        return predictions

    def predict_proba(self, input_ids, attention_mask):
        """Get prediction probabilities."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1)
        return probs
