import os
import sys
import glob
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from PIL import Image
from tqdm import tqdm
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize



DATA_PATH   = r"C:\Ritanshu\9th Summer Sem\Projects\Soft Computing\Dataset\generated_captcha_images"
OUTPUT_DIR  = r"C:\Ritanshu\9th Summer Sem\Projects\Soft Computing\results1"
MODEL_PATH  = os.path.join(OUTPUT_DIR, "crnn_captcha.pth")

BATCH_SIZE      = 16
NUM_EPOCHS      = 30         
LR              = 0.001
WEIGHT_DECAY    = 1e-3
RNN_HIDDEN_SIZE = 256
CLIP_NORM       = 5
TEST_SIZE       = 0.2
RANDOM_STATE    = 0

# Fixed image dimensions — required for consistent CTC sequence lengths
IMG_HEIGHT = 64
IMG_WIDTH  = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n{'='*60}")
print(f"  CAPTCHA RECOGNITION — CRNN")
print(f"{'='*60}")
print(f"  Device   : {device}")
print(f"  Data     : {DATA_PATH}")
print(f"  Output   : {OUTPUT_DIR}")
print(f"  Epochs   : {NUM_EPOCHS}")
print(f"{'='*60}\n")



def build_vocabulary(data_path):
    image_fns = [f for f in os.listdir(data_path) if f.endswith('.png')]
    assert len(image_fns) > 0, f"No PNG images found in {data_path}"

    labels    = [fn.split(".")[0] for fn in image_fns]
    all_chars = sorted(list(set("".join(labels))))
    vocabulary = ["-"] + all_chars          # "-" is the CTC blank token (index 0)
    idx2char   = {k: v for k, v in enumerate(vocabulary)}
    char2idx   = {v: k for k, v in idx2char.items()}

    print(f"[Dataset]  Total images : {len(image_fns)}")
    print(f"[Dataset]  Vocabulary   : {len(vocabulary)} tokens  →  {vocabulary}")
    return image_fns, vocabulary, idx2char, char2idx

image_fns, vocabulary, idx2char, char2idx = build_vocabulary(DATA_PATH)
num_chars = len(char2idx)



image_fns_train, image_fns_test = train_test_split(
    image_fns, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"[Split]    Train : {len(image_fns_train)}  |  Test : {len(image_fns_test)}\n")



class CAPTCHADataset(Dataset):
    """Loads CAPTCHA images; filename (without extension) is the label."""

    def __init__(self, data_dir, image_fns):
        self.data_dir  = data_dir
        self.image_fns = image_fns
        # FIX: added Resize to ensure consistent H and W across all images.
        # CTC loss requires a fixed feature-map height, and padding to the
        # same width is cleaner when all images start at the same size.
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)
            )
        ])

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, idx):
        fn    = self.image_fns[idx]
        path  = os.path.join(self.data_dir, fn)
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        label = fn.split(".")[0]
        return image, label


# FIX: collate_fn is simplified — since all images are now resized to the
# same (H, W), no padding is needed. Kept for safety in case images still
# differ slightly after resize due to aspect ratio handling.
def collate_fn(batch):
    images, texts = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    padded = [F.pad(img, (0, max_w - img.shape[2])) for img in images]
    return torch.stack(padded), texts


trainset = CAPTCHADataset(DATA_PATH, image_fns_train)
testset  = CAPTCHADataset(DATA_PATH, image_fns_test)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, collate_fn=collate_fn)
test_loader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, collate_fn=collate_fn)

print(f"[Loader]   Train batches : {len(train_loader)}  |  Test batches : {len(test_loader)}\n")



_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)


def _get_cnn_output_size(cnn1, cnn2):
    """
    Run a single dummy forward pass through the two CNN blocks to find
    the exact (C * H) size that fc1 must accept.
    """
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
        x = cnn1(dummy)
        x = cnn2(x)
        _, C, H, _ = x.size()
    return C * H


class CRNN(nn.Module):
    """
    CRNN Architecture:
      Input  [B, 3, H, W]
        ↓  ResNet-18 backbone (first 6 modules)
        ↓  Custom Conv2D (256→256, kernel 3×6)
        ↓  Reshape  →  [B, T, C*H]
        ↓  Linear   →  [B, T, 256]       ← FIX: input size now auto-computed
        ↓  BiGRU-1  →  sum fwd+bwd  →  [B, T, 256]
        ↓  BiGRU-2  →  [B, T, 512]
        ↓  Linear   →  [T, B, num_chars]
      Output [T, B, |V|]   (fed to CTC loss)
    """

    def __init__(self, num_chars, rnn_hidden=256, dropout=0.1):
        super().__init__()
        self.num_chars  = num_chars
        self.rnn_hidden = rnn_hidden

        # ── CNN Backbone (ResNet-18, layers 1-6) ──
        backbone = list(_resnet.children())[:-3]
        self.cnn1 = nn.Sequential(*backbone)

        # ── Custom CNN head ──
        self.cnn2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 6), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # FIX: auto-detect the correct input size for fc1 instead of
        # hardcoding 512, which was wrong (actual size is C*H = 256*H_feat).
        cnn_feat_size = _get_cnn_output_size(self.cnn1, self.cnn2)
        print(f"[Model]    CNN feature size (C×H) : {cnn_feat_size}")
        self.fc1 = nn.Linear(cnn_feat_size, rnn_hidden)

        # ── Bidirectional GRU layers ──
        self.gru1 = nn.GRU(rnn_hidden, rnn_hidden, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(rnn_hidden, rnn_hidden, bidirectional=True, batch_first=True)

        self.fc2 = nn.Linear(rnn_hidden * 2, num_chars)

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn1(x)                         # [B, 256, H', W']
        x = self.cnn2(x)                         # [B, 256, H', T]

        # Reshape to sequence
        B, C, H, T = x.size()
        x = x.permute(0, 3, 1, 2)               # [B, T, C, H]
        x = x.reshape(B, T, C * H)              # [B, T, C*H]
        x = self.fc1(x)                          # [B, T, 256]

        # Bidirectional GRU 1 — sum fwd + bwd
        x, _ = self.gru1(x)                      # [B, T, 512]
        half  = x.size(2) // 2
        x     = x[:, :, :half] + x[:, :, half:] # [B, T, 256]

        # Bidirectional GRU 2
        x, _ = self.gru2(x)                      # [B, T, 512]

        x = self.fc2(x)                          # [B, T, num_chars]
        x = x.permute(1, 0, 2)                  # [T, B, num_chars]
        return x


def weights_init(m):
    cls = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif 'BatchNorm' in cls:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

def encode_batch(text_batch):
    lengths  = torch.IntTensor([len(t) for t in text_batch])
    indices  = torch.IntTensor([char2idx[c] for t in text_batch for c in t])
    return indices, lengths

def compute_loss(text_batch, logits):
    """logits: [T, B, C]"""
    log_probs = F.log_softmax(logits, dim=2)
    input_len = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.int32)
    targets, target_len = encode_batch(text_batch)
    return ctc_loss(log_probs, targets, input_len, target_len)



def remove_duplicates(seq):
    """Collapse consecutive duplicate characters."""
    if not seq:
        return []
    return [seq[0]] + [seq[i] for i in range(1, len(seq)) if seq[i] != seq[i-1]]

def decode_predictions(logits):
    """
    logits: [T, B, C]  →  list of decoded strings

    FIX: corrected decode order —
      1. argmax to get best token at each timestep
      2. remove consecutive duplicates
      3. remove blank tokens (index 0)
    Previously the code split on "-" then deduped, which could produce
    wrong results when blank tokens appeared in the middle of a sequence.
    """
    tokens = F.softmax(logits, dim=2).argmax(dim=2)  # [T, B]
    tokens = tokens.cpu().numpy().T                   # [B, T]
    results = []
    for seq in tokens:
        # Step 1: collapse consecutive duplicates (CTC merge)
        merged = remove_duplicates(seq.tolist())
        # Step 2: remove blank token (index 0)
        chars  = [idx2char[i] for i in merged if i != 0]
        results.append("".join(chars))
    return results



def train_model():
    model = CRNN(num_chars, rnn_hidden=RNN_HIDDEN_SIZE).to(device)
    model.apply(weights_init)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # FIX: patience raised to 10 so the scheduler is actually useful
    # across 30 epochs (was patience=5 with only 5 epochs — never triggered).
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    epoch_losses     = []
    iteration_losses = []

    print("=" * 60)
    print("  TRAINING")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        batch_losses = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS}", leave=False)
        for images, texts in loop:
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss   = compute_loss(texts, logits)

            val = loss.item()
            if np.isnan(val) or np.isinf(val):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()

            batch_losses.append(val)
            iteration_losses.append(val)
            loop.set_postfix(loss=f"{val:.4f}")

        epoch_loss = np.mean(batch_losses) if batch_losses else float('nan')
        epoch_losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS}  |  Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}\n")
    return model, epoch_losses, iteration_losses



def evaluate(model, loader):
    model.eval()
    actuals, predictions = [], []
    with torch.no_grad():
        for images, texts in tqdm(loader, desc="  Evaluating", leave=False):
            logits = model(images.to(device))
            preds  = decode_predictions(logits.cpu())
            actuals.extend(texts)
            predictions.extend(preds)
    return actuals, predictions




# ── 9a. Loss curves ──────────────────────────────────────────────────────────
def plot_loss_curves(epoch_losses, iter_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CRNN Training Loss", fontsize=14, fontweight='bold')

    ax1.plot(epoch_losses, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel("Epoch");  ax1.set_ylabel("CTC Loss")
    ax1.set_title("Epoch-wise Loss");  ax1.grid(True, alpha=0.3)

    ax2.plot(iter_losses, 'r-', linewidth=0.8, alpha=0.7)
    ax2.set_xlabel("Iteration");  ax2.set_ylabel("CTC Loss")
    ax2.set_title("Iteration-wise Loss");  ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Loss curves   → {path}")


# ── 9b. Accuracy table ───────────────────────────────────────────────────────
def print_accuracy_table(train_act, train_pred, test_act, test_pred):
    tr_acc = accuracy_score(train_act, train_pred)
    te_acc = accuracy_score(test_act,  test_pred)

    print("\n" + "=" * 50)
    print("  MODEL ACCURACY (Exact String Match)")
    print("=" * 50)
    table = [
        ["Training Set", len(train_act), f"{tr_acc*100:.2f}%"],
        ["Test Set",     len(test_act),  f"{te_acc*100:.2f}%"],
    ]
    print(tabulate(table, headers=["Dataset", "Samples", "Accuracy"],
                   tablefmt="fancy_grid"))
    return tr_acc, te_acc


# ── 9c. Precision / Recall / F1 (character-level) ───────────────────────────
def character_level_metrics(test_act, test_pred):
    """Flatten all characters for per-character metrics."""
    y_true_chars, y_pred_chars = [], []
    for act, pred in zip(test_act, test_pred):
        if len(pred) == len(act):
            y_true_chars.extend(list(act))
            y_pred_chars.extend(list(pred))

    labels = sorted(list(set(y_true_chars)))

    prec = precision_score(y_true_chars, y_pred_chars, average='macro', zero_division=0)
    rec  = recall_score   (y_true_chars, y_pred_chars, average='macro', zero_division=0)
    f1   = f1_score       (y_true_chars, y_pred_chars, average='macro', zero_division=0)
    acc  = accuracy_score (y_true_chars, y_pred_chars)

    print("\n" + "=" * 50)
    print("  CHARACTER-LEVEL METRICS (Test Set)")
    print("=" * 50)
    table = [
        ["Accuracy  (char)",      f"{acc*100:.2f}%"],
        ["Precision (macro avg)", f"{prec*100:.2f}%"],
        ["Recall    (macro avg)", f"{rec*100:.2f}%"],
        ["F1 Score  (macro avg)", f"{f1*100:.2f}%"],
    ]
    print(tabulate(table, headers=["Metric", "Score"], tablefmt="fancy_grid"))

    print("\n  Per-character Classification Report:\n")
    print(classification_report(y_true_chars, y_pred_chars, zero_division=0))

    return y_true_chars, y_pred_chars, labels


# ── 9d. Confusion Matrix ─────────────────────────────────────────────────────
def plot_confusion_matrix(y_true_chars, y_pred_chars, labels):
    cm = confusion_matrix(y_true_chars, y_pred_chars, labels=labels)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.3, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel("Predicted Character", fontsize=12)
    ax.set_ylabel("True Character",      fontsize=12)
    ax.set_title("Confusion Matrix — Character Level (Test Set)", fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Confusion matrix → {path}")


# ── 9e. ROC Curve ────────────────────────────────────────────────────────────
def plot_roc_curve(y_true_chars, y_pred_chars, labels):
    y_true_bin = label_binarize(y_true_chars, classes=labels)
    y_pred_bin = label_binarize(y_pred_chars, classes=labels)

    if y_true_bin.shape[1] < 2:
        print("  [ROC] Not enough classes for ROC — skipping.")
        return

    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    from collections import Counter
    top_labels = [l for l, _ in Counter(y_true_chars).most_common(8)]
    top_idx    = [labels.index(l) for l in top_labels if l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(fpr_micro, tpr_micro, 'b-', lw=2,
            label=f"Micro-avg ROC (AUC = {auc_micro:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Micro-Averaged ROC Curve", fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_idx)))
    for i, idx in enumerate(top_idx):
        if y_true_bin[:, idx].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_pred_bin[:, idx])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=colors[i], lw=1.5,
                 label=f"'{labels[idx]}' (AUC={roc_auc:.2f})")
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Per-Class ROC (Top 8 chars)", fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8); ax2.grid(True, alpha=0.3)

    fig.suptitle("ROC Curve Analysis — CRNN CAPTCHA Model", fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] ROC curve        → {path}")
    print(f"  Micro-avg AUC: {auc_micro:.4f}")


# ── 9f. Mistake analysis ─────────────────────────────────────────────────────
def analyze_mistakes(test_act, test_pred):
    df = pd.DataFrame({'actual': test_act, 'prediction': test_pred})
    mistakes = df[df['actual'] != df['prediction']].copy()

    print(f"\n  Total test samples  : {len(df)}")
    print(f"  Correct predictions : {len(df) - len(mistakes)}")
    print(f"  Mistakes            : {len(mistakes)}")
    print(f"  Mistake rate        : {len(mistakes)/len(df)*100:.2f}%\n")

    if len(mistakes) > 0:
        print("  Sample Mistakes:")
        sample = mistakes.head(10)[['actual', 'prediction']]
        print(tabulate(sample.values.tolist(),
                       headers=["Actual", "Predicted"], tablefmt="fancy_grid"))

    df['pred_len'] = df['prediction'].str.len()
    print("\n  Prediction Length Distribution:")
    print(df['pred_len'].value_counts().sort_index().to_string())

    return mistakes


# ── 9g. Per-char accuracy bar chart ──────────────────────────────────────────
def plot_per_char_accuracy(y_true_chars, y_pred_chars, labels):
    per_char_acc = {}
    for ch in labels:
        mask = [t == ch for t in y_true_chars]
        if sum(mask) == 0:
            continue
        t = [y_true_chars[i] for i in range(len(mask)) if mask[i]]
        p = [y_pred_chars[i] for i in range(len(mask)) if mask[i]]
        per_char_acc[ch] = accuracy_score(t, p)

    chars  = list(per_char_acc.keys())
    scores = list(per_char_acc.values())
    colors = ['#2196F3' if s >= 0.9 else '#FF9800' if s >= 0.7 else '#F44336' for s in scores]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(chars, [s * 100 for s in scores], color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(y=90, color='green',  linestyle='--', linewidth=1.5, label='90% threshold')
    ax.axhline(y=70, color='orange', linestyle='--', linewidth=1.5, label='70% threshold')
    ax.set_xlabel("Character", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Per-Character Recognition Accuracy (Test Set)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{score*100:.0f}%", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "per_char_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Per-char accuracy → {path}")


# ── 9h. Sample predictions grid ──────────────────────────────────────────────
def plot_sample_predictions(test_act, test_pred, n=12):
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    fig.suptitle("Sample Predictions — CRNN CAPTCHA Model", fontsize=13, fontweight='bold')

    indices = np.random.choice(len(test_act), min(n, len(test_act)), replace=False)
    for ax, idx in zip(axes.flat, indices):
        act  = test_act[idx]
        pred = test_pred[idx]
        img_path = os.path.join(DATA_PATH, act + ".png")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
        ax.axis('off')
        color = 'green' if act == pred else 'red'
        ax.set_title(f"True: {act}\nPred: {pred}", fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sample_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Sample preds      → {path}")


# ── 9i. Final summary table ───────────────────────────────────────────────────
def print_final_summary(tr_acc, te_acc, y_true_c, y_pred_c):
    prec = precision_score(y_true_c, y_pred_c, average='macro', zero_division=0)
    rec  = recall_score   (y_true_c, y_pred_c, average='macro', zero_division=0)
    f1   = f1_score       (y_true_c, y_pred_c, average='macro', zero_division=0)
    acc  = accuracy_score (y_true_c, y_pred_c)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    table = [
        ["String Accuracy — Train",     f"{tr_acc*100:.2f}%",  "Exact 5-char match"],
        ["String Accuracy — Test",      f"{te_acc*100:.2f}%",  "Exact 5-char match"],
        ["Char Accuracy  — Test",       f"{acc*100:.2f}%",     "Character level"],
        ["Precision      — Test (M)",   f"{prec*100:.2f}%",    "Macro avg, char level"],
        ["Recall         — Test (M)",   f"{rec*100:.2f}%",     "Macro avg, char level"],
        ["F1 Score       — Test (M)",   f"{f1*100:.2f}%",      "Macro avg, char level"],
    ]
    print(tabulate(table, headers=["Metric", "Value", "Notes"], tablefmt="fancy_grid"))
    print()



if __name__ == "__main__":

    # ── Train (or load saved weights) ─────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        print(f"[Info] Found saved model at {MODEL_PATH}")
        ans = input("  Load existing model? (y/n): ").strip().lower()
        if ans == 'y':
            model = CRNN(num_chars, rnn_hidden=RNN_HIDDEN_SIZE).to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            epoch_losses     = []
            iteration_losses = []
            print("  [Info] Loaded saved model — skipping training.\n")
        else:
            model, epoch_losses, iteration_losses = train_model()
    else:
        model, epoch_losses, iteration_losses = train_model()

    # ── Plot loss curves (skip if loaded) ─────────────────────────────────
    if epoch_losses:
        plot_loss_curves(epoch_losses, iteration_losses)

    # ── Evaluate on train + test ───────────────────────────────────────────
    print("\n[Evaluation] Running on train set...")
    train_loader_eval = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=0, collate_fn=collate_fn)
    train_act, train_pred = evaluate(model, train_loader_eval)

    print("[Evaluation] Running on test set...")
    test_act, test_pred = evaluate(model, test_loader)

    # ── Metrics ───────────────────────────────────────────────────────────
    tr_acc, te_acc = print_accuracy_table(train_act, train_pred, test_act, test_pred)

    mistakes = analyze_mistakes(test_act, test_pred)

    y_true_chars, y_pred_chars, char_labels = character_level_metrics(test_act, test_pred)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[Plots] Generating all visualizations...")
    plot_confusion_matrix(y_true_chars, y_pred_chars, char_labels)
    plot_roc_curve       (y_true_chars, y_pred_chars, char_labels)
    plot_per_char_accuracy(y_true_chars, y_pred_chars, char_labels)
    plot_sample_predictions(test_act, test_pred)

    # ── Final summary ─────────────────────────────────────────────────────
    print_final_summary(tr_acc, te_acc, y_true_chars, y_pred_chars)

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("  Files generated:")
    for f in ["loss_curves.png", "confusion_matrix.png",
              "roc_curve.png", "per_char_accuracy.png", "sample_predictions.png"]:
        print(f"    • {f}")