# CAPTCHA Recognition using CRNN

A deep learning CAPTCHA recognition system that reads the full character sequence from a CAPTCHA image end-to-end using a Convolutional Recurrent Neural Network (CRNN).

It combines a ResNet-18 visual backbone, bidirectional GRUs for sequence modeling, and Connectionist Temporal Classification (CTC) for alignment-free sequence recognition.

## Overview

Rather than classifying each character independently, the model treats CAPTCHA recognition as a sequence prediction problem:

**CAPTCHA image → ResNet-18 feature extraction → sequence formation → BiGRU → BiGRU → CTC decoding → predicted text**

The approach avoids manual character segmentation.

## Model Architecture

### Visual feature extraction

A pretrained ResNet-18 backbone extracts spatial image features. The resulting feature map is reshaped across the width dimension so each spatial position becomes a timestep in the sequence.

A convolutional refinement layer is applied before sequence modeling.

### Sequence modeling

The feature sequence is passed through two bidirectional GRU layers. The first BiGRU produces forward and backward representations that are merged before the second BiGRU.

### CTC decoding

A linear classifier produces character logits at each timestep. CTC loss learns the alignment between the image feature sequence and the target CAPTCHA text without requiring character-level bounding boxes.

At inference, repeated predictions are collapsed and the CTC blank token is removed to form the predicted string.

## Dataset

The project uses **9,955 CAPTCHA images**, each containing **4 characters**:

- Training: 7,964 images
- Held-out test: 1,991 images
- Split: 80/20
- Random state: 0

Character vocabulary:

```text
2 3 4 5 6 7 8 9
A B C D E F G H
J K L M N P Q R
S T U V W X Y Z
```

The dataset is intentionally not committed to the repository.

Expected local path:

```text
Dataset/
└── generated_captcha_images/
```

Each PNG filename is interpreted as its target text.

## Training Configuration

| Setting | Value |
|---|---|
| Input size | 64 × 200 |
| Batch size | 16 |
| Epochs | 30 |
| RNN hidden size | 256 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.001 |
| Gradient clipping | 5 |
| Loss | CTC |
| Train/test split | 80/20 |
| Random state | 0 |

Run training with:

```bash
python -m src.train
```

The trained checkpoint is written locally to:

```text
models/crnn_captcha.pth
```

Model checkpoints are ignored by Git and are not part of the repository.

## Results

The reported results below come from the held-out **1,991-image test split**.

| Metric | Score |
|---|---:|
| Exact string accuracy | **99.30%** |
| Character accuracy | **99.86%** |
| Character precision | **99.85%** |
| Character recall | **99.86%** |
| Character F1 | **99.86%** |

The exact-string metric is the primary end-to-end measure: it requires the entire 4-character CAPTCHA to be predicted correctly.

### Metric note

The character-level metrics in the current evaluation script are computed only for examples where the decoded prediction has the same length as the target. This is useful for analyzing character substitutions, but it does not fully penalize missing or extra decoded characters.

Therefore, the **99.30% exact-string accuracy** is the most direct measure of complete CAPTCHA recognition performance for this experiment.

## Evaluation

After training:

```bash
python -m src.evaluate
```

The evaluation script reports:

- exact-string accuracy
- character-level accuracy
- character-level precision, recall, and F1
- a classification report
- sample predictions saved to `results/predictions.csv`
- metrics saved to `results/metrics.txt`
- character confusion matrix saved to `results/confusion_matrix.png`

Training loss curves are saved to:

```text
results/loss_curves.png
```

## Project Structure

```text
captcha-recognition-crnn/
├── Dataset/                         # local dataset, not committed
│   └── generated_captcha_images/
├── models/                          # local checkpoints, ignored by Git
├── results/
│   ├── loss_curves.png
│   ├── confusion_matrix.png
│   └── ...
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   └── test_project_contracts.py
├── .github/
│   └── workflows/
│       └── tests.yml
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/ritanshu-kumar/captcha-recognition-crnn.git
cd captcha-recognition-crnn

python -m venv .venv
```

Activate the environment.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```text
Dataset/generated_captcha_images/
```

Then run:

```bash
python -m src.train
python -m src.evaluate
```

## Reproducibility

The dataset split uses `random_state=0`, making the train/test partition deterministic for the same input dataset.

The repository does not include the dataset or trained checkpoint, so reproducing the reported metrics requires the same dataset and a compatible PyTorch/torchvision environment.

## Limitations and Extensions

The current experiment is evaluated on one generated CAPTCHA distribution. Generalization to substantially different CAPTCHA styles, fonts, noise patterns, rotations, occlusions, or adversarial transformations has not been established by the reported test results.

Potential extensions include:

- stronger image augmentation
- beam-search CTC decoding
- robustness evaluation under noise, rotation, and occlusion
- comparison against transformer-based sequence models
- a lightweight inference API

## License

This project is licensed under the MIT License.
