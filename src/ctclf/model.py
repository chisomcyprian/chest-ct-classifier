import timm
import torch
import torch.nn.functional as F

from . import CLASSES


def build_model(pretrained=True):
    return timm.create_model("efficientnet_b0", pretrained=pretrained,
                             num_classes=len(CLASSES), drop_rate=0.3)


def load_model(weights_path, device="cpu"):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval()


def grad_cam(model, x, target):
    """Heatmap (H x W, values 0-1) of where the model looked to score `target`."""
    with torch.enable_grad():
        feats = model.forward_features(x)
        feats.retain_grad()
        logits = model.forward_head(feats)
        model.zero_grad()
        logits[0, target].backward()
        weights = feats.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feats).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)