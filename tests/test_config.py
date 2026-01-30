"""Tests for configuration module."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    LABEL2ID,
    ID2LABEL,
    PROJECT_ROOT,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        config = ModelConfig()
        assert config.model_name == "aubmindlab/bert-base-arabertv2"
        assert config.num_labels == 3
        assert config.max_length == 128
        assert config.dropout == 0.1

    def test_custom_values(self):
        config = ModelConfig(num_labels=5, max_length=256, dropout=0.2)
        assert config.num_labels == 5
        assert config.max_length == 256
        assert config.dropout == 0.2


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        config = TrainingConfig()
        assert config.num_epochs == 5
        assert config.batch_size == 32
        assert config.learning_rate == 2e-5
        assert config.weight_decay == 0.01
        assert config.warmup_ratio == 0.1
        assert config.seed == 42
        assert config.fp16 is True

    def test_custom_values(self):
        config = TrainingConfig(num_epochs=10, batch_size=16, learning_rate=3e-5)
        assert config.num_epochs == 10
        assert config.batch_size == 16
        assert config.learning_rate == 3e-5


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self):
        config = DataConfig()
        assert config.test_size == 0.15
        assert config.val_size == 0.15
        assert config.max_samples is None

    def test_split_ratios_sum(self):
        """Train + val + test should equal ~1.0."""
        config = DataConfig()
        train_ratio = 1.0 - config.test_size - config.val_size
        assert abs(train_ratio + config.val_size + config.test_size - 1.0) < 1e-9


class TestLabelMappings:
    """Tests for label mappings."""

    def test_label2id_keys(self):
        assert set(LABEL2ID.keys()) == {"negative", "neutral", "positive"}

    def test_id2label_values(self):
        assert set(ID2LABEL.values()) == {"negative", "neutral", "positive"}

    def test_roundtrip(self):
        """LABEL2ID and ID2LABEL should be inverse mappings."""
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label

    def test_three_classes(self):
        assert len(LABEL2ID) == 3
        assert len(ID2LABEL) == 3


class TestProjectPaths:
    """Tests for project path constants."""

    def test_project_root_exists(self):
        assert os.path.isdir(PROJECT_ROOT)

    def test_src_in_project_root(self):
        assert os.path.isdir(os.path.join(PROJECT_ROOT, "src"))
