"""Tests for the AraBERT sentiment classifier model."""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import AraBERTSentimentClassifier


# Model loading is expensive — skip in CI or when no network
SKIP_MODEL_DOWNLOAD = os.environ.get("SKIP_MODEL_DOWNLOAD", "0") == "1"


@pytest.fixture(scope="module")
def model():
    """Load model once for all tests in this module."""
    if SKIP_MODEL_DOWNLOAD:
        pytest.skip("Skipping model download (SKIP_MODEL_DOWNLOAD=1)")
    return AraBERTSentimentClassifier(
        model_name="aubmindlab/bert-base-arabertv2",
        num_labels=3,
        dropout=0.1,
        use_mlp_head=False,
    )


@pytest.fixture(scope="module")
def mlp_model():
    """Load model with MLP head."""
    if SKIP_MODEL_DOWNLOAD:
        pytest.skip("Skipping model download (SKIP_MODEL_DOWNLOAD=1)")
    return AraBERTSentimentClassifier(
        model_name="aubmindlab/bert-base-arabertv2",
        num_labels=3,
        dropout=0.1,
        use_mlp_head=True,
    )


class TestModelInit:
    """Tests for model initialization."""

    def test_num_labels(self, model):
        assert model.num_labels == 3

    def test_has_bert_encoder(self, model):
        assert hasattr(model, "bert")

    def test_has_classifier(self, model):
        assert hasattr(model, "classifier")

    def test_has_dropout(self, model):
        assert hasattr(model, "dropout")

    def test_parameter_count(self, model):
        total = sum(p.numel() for p in model.parameters())
        # AraBERT base is ~136M params
        assert total > 100_000_000

    def test_mlp_head_has_more_params(self, model, mlp_model):
        linear_params = sum(p.numel() for p in model.parameters())
        mlp_params = sum(p.numel() for p in mlp_model.parameters())
        assert mlp_params > linear_params


class TestModelForward:
    """Tests for model forward pass."""

    def test_forward_without_labels(self, model):
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        result = model(input_ids, attention_mask)
        assert "logits" in result
        assert result["logits"].shape == (batch_size, 3)

    def test_forward_with_labels(self, model):
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.tensor([0, 2])

        result = model(input_ids, attention_mask, labels)
        assert "logits" in result
        assert "loss" in result
        assert result["loss"].dim() == 0  # scalar

    def test_predict_returns_class_indices(self, model):
        batch_size, seq_len = 4, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        preds = model.predict(input_ids, attention_mask)
        assert preds.shape == (batch_size,)
        assert all(0 <= p < 3 for p in preds)

    def test_predict_proba_sums_to_one(self, model):
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        probs = model.predict_proba(input_ids, attention_mask)
        assert probs.shape == (batch_size, 3)
        # Each row should sum to ~1.0
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(batch_size), atol=1e-5)
