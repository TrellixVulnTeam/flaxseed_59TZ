from typing import Dict

from flax import optim
import flax.linen as nn
from flaxseed import FlaxseedModule
from flaxseed.datasets.vision import MNIST
from flaxseed.utils.data import DataLoader
import jax.numpy as np


def onehot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    return labels.reshape(-1, 1) == np.arange(num_classes).reshape(1, -1)


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return -np.mean(np.sum(onehot(labels) * logits, axis=-1))


def accuracy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.mean(labels == np.argmax(logits, axis=-1))


class Model(FlaxseedModule):
    @nn.compact
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return nn.log_softmax(x)

    def init_params(self, key: np.ndarray) -> Dict:
        dummy_inputs = np.ones((1, 28, 28, 1), np.float32)
        return self.init(key, dummy_inputs)["params"]

    def init_optimizer(self, params) -> Dict:
        return optim.Adam(learning_rate=1e-3).create(params)

    def training_step(self, params, batch, random_key):
        inputs, labels = batch
        logits = self.apply({"params": params}, inputs)
        return {
            "loss": cross_entropy_loss(logits, labels),
            "accuracy": accuracy(logits, labels),
        }

    def eval_step(self, params, batch, random_key):
        inputs, labels = batch
        logits = self.apply({"params": params}, inputs)
        return {
            "loss": cross_entropy_loss(logits, labels),
            "accuracy": accuracy(logits, labels),
        }


training_loader = DataLoader(MNIST("data/mnist", download=True), batch_size=64, shuffle=True)
eval_loader = DataLoader(MNIST("data/mnist", train=False), batch_size=64)

model = Model()
# breakpoint()
# model.train(training_loader, eval_loader)
model.load_checkpoint("mnist.checkpoint")
model.test(eval_loader)
# model.save_checkpoint("mnist.checkpoint")
