from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from .dataset import DATA_PATH, create_dataloaders
from .model import CRNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "crnn_captcha.pth"
RESULTS_DIR = PROJECT_ROOT / "results"


def remove_duplicates(sequence):
    if not sequence:
        return []
    return [
        sequence[0],
        *[
            sequence[i]
            for i in range(1, len(sequence))
            if sequence[i] != sequence[i - 1]
        ],
    ]


def decode_predictions(logits, idx2char):
    tokens = logits.argmax(dim=2).cpu().numpy().T
    predictions = []

    for sequence in tokens:
        merged = remove_duplicates(sequence.tolist())
        chars = [idx2char[idx] for idx in merged if idx != 0]
        predictions.append("".join(chars))

    return predictions


def evaluate_model(model, loader, idx2char, device):
    model.eval()
    actuals, predictions = [], []

    with torch.no_grad():
        for images, texts in tqdm(loader, desc="Evaluating"):
            logits = model(images.to(device))
            predictions.extend(decode_predictions(logits, idx2char))
            actuals.extend(texts)

    return actuals, predictions


def character_level_metrics(actuals, predictions):
    y_true, y_pred = [], []

    for actual, predicted in zip(actuals, predictions):
        if len(actual) == len(predicted):
            y_true.extend(actual)
            y_pred.extend(predicted)

    if not y_true:
        raise RuntimeError("No same-length predictions available for character metrics.")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_results(actuals, predictions, char_metrics):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    exact_accuracy = accuracy_score(actuals, predictions)

    results = pd.DataFrame({
        "actual": actuals,
        "predicted": predictions,
    })
    results.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    with open(RESULTS_DIR / "metrics.txt", "w", encoding="utf-8") as file:
        file.write(f"Exact String Accuracy: {exact_accuracy * 100:.2f}%\n")
        file.write(
            f"Character Accuracy: {char_metrics['accuracy'] * 100:.2f}%\n"
        )
        file.write(
            f"Character Precision: {char_metrics['precision'] * 100:.2f}%\n"
        )
        file.write(
            f"Character Recall: {char_metrics['recall'] * 100:.2f}%\n"
        )
        file.write(
            f"Character F1: {char_metrics['f1'] * 100:.2f}%\n"
        )

    return exact_accuracy


def save_confusion_matrix(y_true, y_pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted Character")
    plt.ylabel("True Character")
    plt.title("Character-Level Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )

    _, test_loader, metadata = create_dataloaders(DATA_PATH)

    model = CRNN(num_chars=len(metadata["vocabulary"])).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    actuals, predictions = evaluate_model(
        model,
        test_loader,
        metadata["idx2char"],
        device,
    )

    exact_accuracy = accuracy_score(actuals, predictions)
    char_metrics = character_level_metrics(actuals, predictions)

    print("\n=== CAPTCHA Recognition Results ===")
    print(f"Exact String Accuracy: {exact_accuracy * 100:.2f}%")
    print(f"Character Accuracy:   {char_metrics['accuracy'] * 100:.2f}%")
    print(f"Character Precision:  {char_metrics['precision'] * 100:.2f}%")
    print(f"Character Recall:     {char_metrics['recall'] * 100:.2f}%")
    print(f"Character F1:         {char_metrics['f1'] * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            char_metrics["y_true"],
            char_metrics["y_pred"],
            zero_division=0,
        )
    )

    save_results(actuals, predictions, char_metrics)
    save_confusion_matrix(
        char_metrics["y_true"],
        char_metrics["y_pred"],
    )

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
