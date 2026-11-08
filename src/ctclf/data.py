from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet statistics


def _add_noise(t):
    return (t + 0.03 * torch.randn_like(t)).clamp(0, 1)


# Training: equalize contrast, then simulate scanner differences.
# No horizontal flip: the heart sits on the left, so flipped chests aren't realistic.
train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomEqualize(p=1.0),
    T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.85, 1.15)),
    T.ColorJitter(brightness=0.3, contrast=0.3),
    T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.RandomApply([T.Lambda(_add_noise)], p=0.3),
    T.Normalize(MEAN, STD),
])

# Evaluation and the API: same contrast equalization, no randomness.
eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomEqualize(p=1.0),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def load_slice(path, box=None):
    """Open a CT slice as grayscale and crop it to the lung box if given."""
    img = Image.open(path).convert("L")
    return img.crop(box) if box is not None else img


class CTDataset(Dataset):
    """Rows of a splits/*.csv file -> (image tensor, label)."""

    def __init__(self, csv_path, image_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = load_slice(self.image_dir / r.filename, (r.xmin, r.ymin, r.xmax, r.ymax))
        return self.transform(img), int(r.label)