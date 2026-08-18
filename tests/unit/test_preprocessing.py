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
