import base64
import io

import numpy as np
import torch
from PIL import Image

from . import CLASSES
from .data import IMG_SIZE, eval_transform
from .model import grad_cam, load_model


def crop_to_body(img, threshold=0.15, margin=8):
    """Crop a full CT slice to the patient's body, approximating the dataset's lung boxes."""
    a = np.asarray(img, dtype=np.float32)
    mask = a > threshold * a.max()
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if len(rows) == 0 or len(cols) == 0:
        return img
    h, w = a.shape
    box = (max(cols[0] - margin, 0), max(rows[0] - margin, 0),
           min(cols[-1] + margin, w), min(rows[-1] + margin, h))
    return img.crop(box)


def heatmap_png(gray, cam):
    """Overlay a Grad-CAM map on the slice; return it as a base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps

    heat = colormaps["jet"](cam)[..., :3]
    base = np.asarray(gray.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)[..., None] / 255
    blend = (0.6 * base + 0.4 * heat).clip(0, 1)
    buf = io.BytesIO()
    Image.fromarray((blend * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Predictor:
    def __init__(self, weights_path, device="cpu"):
        self.device = device
        self.model = load_model(weights_path, device)
        self.classes = CLASSES

    def predict(self, file, explain=False):
        img = crop_to_body(Image.open(file).convert("L"))
        x = eval_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        k = int(probs.argmax())
        result = {
            "prediction": CLASSES[k],
            "confidence": round(float(probs[k]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
        }
        if explain:
            result["gradcam_png"] = heatmap_png(img, grad_cam(self.model, x, target=k))
        return result