"""
Arabic text preprocessing utilities.
Handles normalization, diacritics removal, and AraBERT-specific preprocessing.
"""
import re
import unicodedata
import logging
from typing import List, Optional

from .config import ARABIC_DIACRITICS

logger = logging.getLogger(__name__)


class ArabicTextPreprocessor:
    """Preprocessor for Arabic text with AraBERT-specific normalization."""

    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv2"):
        self.arabert_prep = None
        try:
            from arabert.preprocess import ArabertPreprocessor
            self.arabert_prep = ArabertPreprocessor(model_name=model_name)
        except Exception as e:
            logger.warning(
                f"Could not initialize ArabertPreprocessor (Java/Farasa may be missing): {e}. "
                "Falling back to basic Arabic preprocessing."
            )

    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritical marks (tashkeel)."""
        return re.sub(ARABIC_DIACRITICS, '', text)

    def normalize_arabic(self, text: str) -> str:
        """Normalize Arabic characters."""
        # Normalize alef variants
        text = re.sub(r'[إأآا]', 'ا', text)
        # Normalize taa marbuta
        text = re.sub(r'ة', 'ه', text)
        # Normalize ya
        text = re.sub(r'ى', 'ي', text)
        return text

    def clean_text(self, text: str) -> str:
        """Clean text: remove URLs, mentions, extra whitespace."""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtag symbols (keep the text)
        text = re.sub(r'#', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove repeated characters (more than 2)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        return text

    def preprocess(self, text: str, use_arabert: bool = True) -> str:
        """
        Full preprocessing pipeline.

        Args:
            text: Raw Arabic text.
            use_arabert: Whether to apply AraBERT-specific preprocessing.

        Returns:
            Preprocessed text ready for tokenization.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Basic cleaning
        text = self.clean_text(text)
        text = self.remove_diacritics(text)

        # AraBERT preprocessing (handles farasa segmentation internally)
        if use_arabert and self.arabert_prep is not None:
            text = self.arabert_prep.preprocess(text)

        return text

    def preprocess_batch(self, texts: List[str], use_arabert: bool = True) -> List[str]:
        """Preprocess a batch of texts."""
        return [self.preprocess(t, use_arabert) for t in texts]
