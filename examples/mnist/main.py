from typing import Any, Dict

import flax.linen as nn
import jax
import jax.numpy as np
import optax
from flax.jax_utils import replicate, unreplicate
from flax.training.common_utils import get_metrics, onehot, shard
from flax.training.train_state import TrainState
from jax import random
from tqdm import tqdm

from flaxseed.datasets.vision import MNIST
from flaxseed.utils.data import DataLoader


def loss_fn(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    onehot_labels = onehot(labels, num_classes=logits.shape[-1])
    return optax.softmax_cross_entropy(logits, onehot_labels).mean()


def accuracy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.mean(labels == np.argmax(logits, axis=-1))


class Model(nn.Module):
    num_features: int = 256
    num_outputs: int = 10
    dtype: np.dtype = np.float32

    @nn.compact
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(features=self.num_features)(x)
        x = nn.relu(x)
        return nn.Dense(features=self.num_outputs)(x)


def train_step(state: TrainState, batch):
    def compute_loss(params: Dict[str, Any]):
        inputs, labels = batch
        logits = state.apply_fn({"params": params}, inputs)
        return loss_fn(logits, labels)

    grad_fn = jax.value_and_grad(compute_loss)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)

    metrics = {"loss": loss}
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    return new_state, metrics


def eval_step(params: Dict[str, Any], batch):
    inputs, labels = batch
    logits = model.apply({"params": params}, inputs)
    assert isinstance(logits, np.ndarray)
    loss = loss_fn(logits, labels)

    metrics = {"loss": loss, "accuracy": accuracy(logits, labels)}
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return metrics


def collate_fn(batch):
    inputs = np.stack([x[0] for x in batch], axis=0)
    labels = np.array([x[1] for x in batch])
    return {"inputs": inputs, "labels": labels}


if __name__ == "__main__":
    num_epochs = 3
    rng = random.PRNGKey(42)

    model = Model()
    dummy_inputs = np.ones((1, 28, 28, 1), np.float32)
    params = model.init(rng, dummy_inputs)["params"]
    tx = optax.adam(learning_rate=1e-3)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_loader = DataLoader(
        MNIST("data/mnist", download=True), batch_size=64, shuffle=True
    )
    eval_loader = DataLoader(MNIST("data/mnist", train=False), batch_size=64)

    p_train_state = replicate(train_state)
    p_train_step = jax.pmap(train_step, "batch")
    p_eval_step = jax.pmap(eval_step, "batch")

    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")
        rng, input_rng = random.split(rng)
        train_metrics = []

        for batch in tqdm(train_loader, desc="Train"):
            batch = shard(batch)
            p_train_state, metrics = p_train_step(p_train_state, batch)
            train_metrics.append(metrics)

        eval_metrics = []
        for batch in tqdm(eval_loader, desc="Val"):
            batch = shard(batch)
            metrics = p_eval_step(p_train_state.params, batch)
            eval_metrics.append(metrics)

        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(np.mean, eval_metrics)
        print(eval_metrics)
