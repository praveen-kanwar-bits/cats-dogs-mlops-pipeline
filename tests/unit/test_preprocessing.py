import csv
from collections import Counter
from pathlib import Path

import pytest

preprocessing = pytest.importorskip("cats_dogs_mlops.application.preprocessing")


def test_open_as_rgb_converts_grayscale():
    image = preprocessing.open_as_rgb(Path("tests/fixtures/sample_gray.png"))
    assert image.mode == "RGB"


def test_eval_transform_shape():
    torch = pytest.importorskip("torch")
    image = preprocessing.open_as_rgb(Path("tests/fixtures/sample_cat.jpg"))
    tensor = preprocessing.build_eval_transform(224)(image)
    assert tuple(tensor.shape) == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


def test_corrupt_image_rejected():
    with pytest.raises(ValueError):
        preprocessing.open_as_rgb(Path("tests/fixtures/corrupt.jpg"))


def test_eval_transform_deterministic():
    image = preprocessing.open_as_rgb(Path("tests/fixtures/sample_cat.jpg"))
    transform = preprocessing.build_eval_transform(224)
    a = transform(image)
    b = transform(image)
    assert (a == b).all()


def test_stratified_split_is_exact_80_10_10(tmp_path):
    records = [
        (tmp_path / f"{label}_{index}.jpg", label)
        for label in ("cat", "dog")
        for index in range(100)
    ]
    manifest = tmp_path / "split_manifest.csv"

    preprocessing.make_split_manifest(records, manifest, 0.8, 0.1, 0.1, seed=42)

    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert Counter(row["split"] for row in rows) == {
        "train": 160,
        "validation": 20,
        "test": 20,
    }
    assert Counter((row["split"], row["label"]) for row in rows) == {
        ("train", "cat"): 80,
        ("train", "dog"): 80,
        ("validation", "cat"): 10,
        ("validation", "dog"): 10,
        ("test", "cat"): 10,
        ("test", "dog"): 10,
    }


def test_training_transform_contains_augmentation():
    transform_names = {
        transform.__class__.__name__
        for transform in preprocessing.build_train_transform(224, flip_p=0.5, rotation=10).transforms
    }
    assert {"RandomHorizontalFlip", "RandomRotation", "ColorJitter"} <= transform_names
