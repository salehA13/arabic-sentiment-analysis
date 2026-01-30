"""
Inference module for Arabic sentiment prediction.
Supports single text, batch, and interactive modes.
"""
import os
import torch
from typing import Dict, List, Union

from transformers import AutoTokenizer

from .config import ModelConfig, ID2LABEL, MODELS_DIR
from .model import AraBERTSentimentClassifier
from .preprocessing import ArabicTextPreprocessor


class SentimentPredictor:
    """
    Arabic sentiment predictor using fine-tuned AraBERT.

    Usage:
        predictor = SentimentPredictor.from_pretrained("models/arabert-sentiment")
        result = predictor.predict("هذا الفيلم رائع جداً")
        # {'label': 'positive', 'confidence': 0.95, 'probabilities': {...}}
    """

    def __init__(
        self,
        model: AraBERTSentimentClassifier,
        tokenizer: AutoTokenizer,
        preprocessor: ArabicTextPreprocessor,
        device: str = None,
        max_length: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_length = max_length

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        model_name: str = "aubmindlab/bert-base-arabertv2",
        device: str = None,
    ) -> "SentimentPredictor":
        """
        Load a trained model from directory.

        Args:
            model_dir: Path to saved model directory.
            model_name: Base model name for tokenizer.
            device: Device to use.

        Returns:
            SentimentPredictor instance.
        """
        # Load checkpoint
        checkpoint_path = os.path.join(model_dir, "best_model.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Initialize model
        config = checkpoint.get("config", {})
        num_labels = config.get("num_labels", 3)

        model = AraBERTSentimentClassifier(
            model_name=model_name,
            num_labels=num_labels,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # Tokenizer and preprocessor
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        preprocessor = ArabicTextPreprocessor(model_name=model_name)

        return cls(
            model=model,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            device=device,
        )

    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single text.

        Args:
            text: Arabic text input.

        Returns:
            Dictionary with label, confidence, and per-class probabilities.
        """
        # Preprocess
        processed = self.preprocessor.preprocess(text)

        # Tokenize
        encoding = self.tokenizer(
            processed,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Predict
        with torch.no_grad():
            probs = self.model.predict_proba(input_ids, attention_mask)

        probs = probs.cpu().numpy()[0]
        pred_id = int(probs.argmax())

        return {
            "text": text,
            "processed_text": processed,
            "label": ID2LABEL[pred_id],
            "confidence": float(probs[pred_id]),
            "probabilities": {
                ID2LABEL[i]: float(p) for i, p in enumerate(probs)
            },
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for a batch of texts."""
        return [self.predict(text) for text in texts]


def run_interactive():
    """Run interactive prediction mode."""
    print("\n" + "=" * 60)
    print("Arabic Sentiment Analysis - Interactive Mode")
    print("=" * 60)

    model_dir = os.path.join(MODELS_DIR, "arabert-sentiment")

    if not os.path.exists(os.path.join(model_dir, "best_model.pt")):
        print("Error: No trained model found. Run training first.")
        print(f"Expected model at: {model_dir}/best_model.pt")
        return

    print("Loading model...")
    predictor = SentimentPredictor.from_pretrained(model_dir)
    print("Model loaded! Enter Arabic text (or 'quit' to exit):\n")

    while True:
        text = input("📝 Enter text: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            print("مع السلامة! 👋")
            break

        if not text:
            continue

        result = predictor.predict(text)

        emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        emoji = emoji_map.get(result["label"], "")

        print(f"\n  Sentiment: {result['label']} {emoji}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities:")
        for label, prob in result["probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"    {label:>8}: {prob:.2%} {bar}")
        print()


if __name__ == "__main__":
    run_interactive()
