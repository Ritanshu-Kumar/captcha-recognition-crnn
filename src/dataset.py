from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Dataset is expected outside the Git repository or locally under Dataset/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Dataset" / "generated_captcha_images"

IMG_HEIGHT = 64
IMG_WIDTH = 200
BATCH_SIZE = 16
TEST_SIZE = 0.20
RANDOM_STATE = 0


def build_vocabulary(data_path: Path) -> Tuple[List[str], List[str], Dict[int, str], Dict[str, int]]:
    """Build the CTC vocabulary from CAPTCHA labels stored in filenames."""
    image_fns = sorted(
        f.name for f in data_path.iterdir()
        if f.is_file() and f.suffix.lower() == ".png"
    )
    if not image_fns:
        raise FileNotFoundError(f"No PNG images found in {data_path}")

    labels = [fn.rsplit(".", 1)[0] for fn in image_fns]
    all_chars = sorted(set("".join(labels)))

    vocabulary = ["-"] + all_chars  # index 0 is the CTC blank token
    idx2char = {idx: char for idx, char in enumerate(vocabulary)}
    char2idx = {char: idx for idx, char in idx2char.items()}

    return image_fns, vocabulary, idx2char, char2idx


class CAPTCHADataset(Dataset):
    """CAPTCHA images whose filename (without extension) is the target text."""

    def __init__(self, data_dir: Path, image_fns: List[str]):
        self.data_dir = Path(data_dir)
        self.image_fns = image_fns

        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self) -> int:
        return len(self.image_fns)

    def __getitem__(self, idx: int):
        filename = self.image_fns[idx]
        image = Image.open(self.data_dir / filename).convert("RGB")
        image = self.transform(image)
        label = filename.rsplit(".", 1)[0]
        return image, label


def collate_fn(batch):
    """Pad image widths if needed and return a tensor plus raw text labels."""
    images, texts = zip(*batch)
    max_width = max(image.shape[2] for image in images)
    padded = [
        F.pad(image, (0, max_width - image.shape[2]))
        for image in images
    ]
    return torch.stack(padded), texts


def create_dataloaders(data_path: Path = DATA_PATH):
    image_fns, vocabulary, idx2char, char2idx = build_vocabulary(data_path)

    train_fns, test_fns = train_test_split(
        image_fns,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_dataset = CAPTCHADataset(data_path, train_fns)
    test_dataset = CAPTCHADataset(data_path, test_fns)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    metadata = {
        "vocabulary": vocabulary,
        "idx2char": idx2char,
        "char2idx": char2idx,
        "train_fns": train_fns,
        "test_fns": test_fns,
    }

    return train_loader, test_loader, metadata
