import logging
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
MODEL = "facebook/bart-large-mnli"

def classify(text: str, labels: list[str]):
    logging.info("⏳ Running zero-shot classification…")
    classifier = pipeline("zero-shot-classification", model=MODEL)
    result = classifier(text, labels)
    return list(zip(result["labels"], result["scores"]))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Zero-shot classify TEXT into comma-separated LABELS"
    )
    p.add_argument("--text",   required=True, help="Sentence to classify")
    p.add_argument("--labels", required=True,
                   help="Comma-separated candidate labels")
    args = p.parse_args()

    labels = [l.strip() for l in args.labels.split(",")]
    logging.info(f"✅ Loaded model {MODEL}.")
    scores = classify(args.text, labels)

    print("\n🔍 Classification results:")
    for label, score in scores:
        print(f" • {label:<12} : {score:.3f}")
