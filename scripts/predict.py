#!/usr/bin/env python3
"""
Prediction script for Arabic Sentiment Analysis.

Usage:
    # Interactive mode
    python scripts/predict.py

    # Single text
    python scripts/predict.py --text "هذا المنتج ممتاز"

    # From file (one text per line)
    python scripts/predict.py --input_file texts.txt --output_file predictions.json
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import SentimentPredictor, run_interactive


def parse_args():
    parser = argparse.ArgumentParser(description="Arabic Sentiment Prediction")
    parser.add_argument("--text", type=str, default=None, help="Single text to predict")
    parser.add_argument("--input_file", type=str, default=None, help="File with texts (one per line)")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file")
    parser.add_argument("--model_dir", type=str, default="models/arabert-sentiment")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.text is None and args.input_file is None:
        run_interactive()
        return

    # Load model
    print("Loading model...")
    predictor = SentimentPredictor.from_pretrained(args.model_dir)
    print("Model loaded!\n")

    if args.text:
        # Single prediction
        result = predictor.predict(args.text)
        emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        emoji = emoji_map.get(result["label"], "")

        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['label']} {emoji}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities:")
        for label, prob in result["probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"  {label:>8}: {prob:.2%} {bar}")

    elif args.input_file:
        # Batch prediction
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"Predicting {len(texts)} texts...")
        results = predictor.predict_batch(texts)

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output_file}")
        else:
            for r in results:
                print(f"{r['label']:>8} ({r['confidence']:.0%}): {r['text'][:80]}")


if __name__ == "__main__":
    main()
