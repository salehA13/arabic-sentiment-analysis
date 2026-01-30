"""Tests for Arabic text preprocessing module."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import ArabicTextPreprocessor


@pytest.fixture
def preprocessor():
    """Create preprocessor instance (without AraBERT dependency)."""
    prep = ArabicTextPreprocessor.__new__(ArabicTextPreprocessor)
    prep.arabert_prep = None  # Skip Farasa/Java dependency
    return prep


class TestDiacriticsRemoval:
    """Tests for Arabic diacritics removal."""

    def test_removes_fatha(self, preprocessor):
        assert "ك" in preprocessor.remove_diacritics("كَ")

    def test_removes_kasra(self, preprocessor):
        assert "ب" in preprocessor.remove_diacritics("بِ")

    def test_removes_damma(self, preprocessor):
        assert "م" in preprocessor.remove_diacritics("مُ")

    def test_removes_shadda(self, preprocessor):
        result = preprocessor.remove_diacritics("شدّة")
        assert "ّ" not in result

    def test_preserves_plain_text(self, preprocessor):
        text = "مرحبا بالعالم"
        assert preprocessor.remove_diacritics(text) == text


class TestArabicNormalization:
    """Tests for Arabic character normalization."""

    def test_normalizes_alef_hamza_above(self, preprocessor):
        assert preprocessor.normalize_arabic("أحمد") == "احمد"

    def test_normalizes_alef_hamza_below(self, preprocessor):
        assert preprocessor.normalize_arabic("إسلام") == "اسلام"

    def test_normalizes_alef_madda(self, preprocessor):
        assert preprocessor.normalize_arabic("آمال") == "امال"

    def test_normalizes_taa_marbuta(self, preprocessor):
        assert preprocessor.normalize_arabic("مدرسة") == "مدرسه"

    def test_normalizes_alef_maqsura(self, preprocessor):
        assert preprocessor.normalize_arabic("على") == "علي"


class TestTextCleaning:
    """Tests for text cleaning utilities."""

    def test_removes_urls(self, preprocessor):
        text = "شاهد https://example.com الموقع"
        result = preprocessor.clean_text(text)
        assert "https" not in result
        assert "example" not in result

    def test_removes_mentions(self, preprocessor):
        text = "مرحباً @user123 كيف حالك"
        result = preprocessor.clean_text(text)
        assert "@user123" not in result

    def test_removes_hashtag_symbol(self, preprocessor):
        text = "#عربي"
        result = preprocessor.clean_text(text)
        assert "#" not in result
        assert "عربي" in result

    def test_collapses_repeated_chars(self, preprocessor):
        text = "جميييييل"
        result = preprocessor.clean_text(text)
        assert "ييييي" not in result

    def test_strips_whitespace(self, preprocessor):
        text = "  مرحبا   بالعالم  "
        result = preprocessor.clean_text(text)
        assert result == "مرحبا بالعالم"


class TestFullPipeline:
    """Tests for the full preprocessing pipeline."""

    def test_empty_string(self, preprocessor):
        assert preprocessor.preprocess("") == ""
        assert preprocessor.preprocess("   ") == ""

    def test_non_string_input(self, preprocessor):
        assert preprocessor.preprocess(None) == ""

    def test_full_preprocess(self, preprocessor):
        text = "أحبّ هذا المنتج https://shop.com @store #ممتاز"
        result = preprocessor.preprocess(text, use_arabert=False)
        assert "https" not in result
        assert "@store" not in result
        assert "#" not in result
        assert len(result) > 0

    def test_batch_preprocess(self, preprocessor):
        texts = ["مرحبا", "أهلاً", ""]
        results = preprocessor.preprocess_batch(texts, use_arabert=False)
        assert len(results) == 3
        assert results[2] == ""
