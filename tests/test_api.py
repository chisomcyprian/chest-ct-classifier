import io

import pytest
from PIL import Image

from app.main import create_app


class FakePredictor:
    classes = ["Normal", "Pneumonia", "COVID-19"]

    def predict(self, file, explain=False):
        Image.open(file).verify()   # fail on non-images, like the real one
        result = {"prediction": "COVID-19", "confidence": 0.9,
                  "probabilities": {"Normal": 0.05, "Pneumonia": 0.05, "COVID-19": 0.9}}
        if explain:
            result["gradcam_png"] = "abc"
        return result


@pytest.fixture
def client():
    return create_app(FakePredictor()).test_client()


def png_bytes():
    buf = io.BytesIO()
    Image.new("L", (32, 32)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health(client):
    assert client.get("/health").json["status"] == "ok"


def test_predict_returns_probabilities(client):
    r = client.post("/predict?explain=true", data={"file": (png_bytes(), "slice.png")})
    assert r.status_code == 200
    assert r.json["prediction"] == "COVID-19"
    assert "gradcam_png" in r.json and "disclaimer" in r.json


def test_missing_file(client):
    assert client.post("/predict").status_code == 400


def test_wrong_file_type(client):
    r = client.post("/predict", data={"file": (io.BytesIO(b"x"), "scan.dcm")})
    assert r.status_code == 400


def test_corrupt_image(client):
    r = client.post("/predict", data={"file": (io.BytesIO(b"not an image"), "x.png")})
    assert r.status_code == 422