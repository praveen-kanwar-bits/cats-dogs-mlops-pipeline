import torch
from cats_dogs_mlops.application.training import SimpleCNN


def test_model_output_shape():
    model = SimpleCNN()
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    assert tuple(y.shape) == (4, 2)
