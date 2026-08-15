from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .dataset import BATCH_SIZE, DATA_PATH, create_dataloaders
from .model import CRNN

NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
RNN_HIDDEN_SIZE = 256
CLIP_NORM = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = MODEL_DIR / "crnn_captcha.pth"


def encode_batch(text_batch, char2idx):
    target_lengths = torch.tensor(
        [len(text) for text in text_batch],
        dtype=torch.int32,
    )
    targets = torch.tensor(
        [char2idx[ch] for text in text_batch for ch in text],
        dtype=torch.int32,
    )
    return targets, target_lengths


def compute_ctc_loss(text_batch, logits, char2idx):
    log_probs = F.log_softmax(logits, dim=2)
    input_lengths = torch.full(
        (log_probs.size(1),),
        log_probs.size(0),
        dtype=torch.int32,
    )
    targets, target_lengths = encode_batch(text_batch, char2idx)

    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    return criterion(log_probs, targets, input_lengths, target_lengths)


def plot_loss_curves(epoch_losses, iteration_losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CRNN Training Loss", fontweight="bold")

    axes[0].plot(epoch_losses, marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CTC Loss")
    axes[0].set_title("Epoch-wise Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iteration_losses, linewidth=0.8)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("CTC Loss")
    axes[1].set_title("Iteration-wise Loss")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Dataset: {DATA_PATH}")

    train_loader, _, metadata = create_dataloaders(DATA_PATH)
    char2idx = metadata["char2idx"]

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Vocabulary: {metadata['vocabulary']}")

    model = CRNN(
        num_chars=len(metadata["vocabulary"]),
        rnn_hidden=RNN_HIDDEN_SIZE,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=10,
        factor=0.5,
    )

    epoch_losses = []
    iteration_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        batch_losses = []

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{NUM_EPOCHS}",
        )

        for images, texts in loop:
            images = images.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = compute_ctc_loss(texts, logits, char2idx)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()

            value = loss.item()
            batch_losses.append(value)
            iteration_losses.append(value)
            loop.set_postfix(loss=f"{value:.4f}")

        epoch_loss = float(np.mean(batch_losses))
        epoch_losses.append(epoch_loss)
        scheduler.step(epoch_loss)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Loss: {epoch_loss:.4f}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    plot_loss_curves(epoch_losses, iteration_losses)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Loss curve saved to: {RESULTS_DIR / 'loss_curves.png'}")


if __name__ == "__main__":
    main()
