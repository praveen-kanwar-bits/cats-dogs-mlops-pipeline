import torch

from cats_dogs_mlops.application.training import SimpleCNN, optimizer_factory


def test_model_output_shape():
    model = SimpleCNN()
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    assert tuple(y.shape) == (4, 2)


def test_optimizer_factory_honors_configured_optimizer():
    model = SimpleCNN()
    assert isinstance(optimizer_factory("adam", model.parameters(), 0.001), torch.optim.Adam)
    assert isinstance(optimizer_factory("sgd", model.parameters(), 0.001), torch.optim.SGD)
