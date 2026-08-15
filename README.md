# CAPTCHA Recognition using CRNN

A deep learning CAPTCHA recognition system that reads the full character sequence from a CAPTCHA image end-to-end, using a Convolutional Recurrent Neural Network (CRNN).

It combines a ResNet-18 backbone for visual feature extraction, bidirectional GRUs for sequence modeling, and Connectionist Temporal Classification (CTC) for decoding — so there's no need to segment individual characters before recognizing them.

## Overview

Rather than classifying each character independently, this treats CAPTCHA reading as a sequence recognition problem: the model looks at the whole image and predicts the character sequence directly.

Pipeline: CAPTCHA image → ResNet-18 feature extraction → sequence formation → two BiGRU layers → CTC decoding → predicted text.

## Model architecture

**Visual feature extraction** — a pretrained ResNet-18 backbone extracts spatial features from the image, which are reshaped into a sequence so each spatial position becomes a timestep.

**Sequence modeling** — the feature sequence passes through two bidirectional GRU layers, letting the model use context from both directions of the character sequence.

**CTC decoding** — a linear classifier on top of the GRU output is trained with CTC loss, which learns the alignment between image features and target characters without needing manually segmented characters. At inference, CTC decoding collapses repeated predictions and strips blank tokens to produce the final string.

## Dataset

9,955 CAPTCHA images, 4 characters each, split 7,964 train / 1,991 test (80/20).

Character vocabulary:

```
2 3 4 5 6 7 8 9
A B C D E F G H
J K L M N P Q R
S T U V W X Y Z
```

Not included in this repo — place it locally at:

```
Dataset/
└── generated_captcha_images/
```

## Training

- Architecture: ResNet-18 + BiGRU + BiGRU + CTC
- Input size: 64×200
- Batch size: 16, 30 epochs
- 80/20 train/test split, CTC loss

```bash
python -m src.train
```

Trained weights are saved to `models/`.

## Results

Evaluated on the 1,991-image held-out test set (7,952 characters total):

| Metric | Score |
|---|---|
| Exact string accuracy | 99.30% |
| Character accuracy | 99.86% |
| Character precision | 99.85% |
| Character recall | 99.86% |
| Character F1 | 99.86% |

The model gets the complete 4-character string exactly right 99.3% of the time.

## Evaluation

```bash
python -m src.evaluate
```

Produces exact-string and character-level accuracy, precision/recall/F1, a classification report, confusion matrix, and sample predictions, saved to `results/`.

Training loss shows fast convergence of the CTC objective, and the character-level confusion matrix is heavily diagonal — errors are rare and not concentrated on any particular character pair.

## Project structure

```
captcha-recognition-crnn/
├── Dataset/                 # local dataset, not committed
│   └── generated_captcha_images/
├── docs/
│   └── architecture.png
├── models/                  
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
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/ritanshu-kumar/captcha-recognition-crnn.git
cd captcha-recognition-crnn

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Place the dataset at `Dataset/generated_captcha_images/`, then:

```bash
python -m src.train
python -m src.evaluate
```

## Reproducibility

The train/test split uses a fixed random seed, so the partition is consistent across runs. The results above are from the CRNN model evaluated on that held-out split.

## Possible extensions

Augmentation for harder CAPTCHA variations, beam-search CTC decoding, testing against noise/rotation/occlusion, a transformer-based sequence model for comparison, and wrapping the model in an inference API.

## License

For educational and research use.