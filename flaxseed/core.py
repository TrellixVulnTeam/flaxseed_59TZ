from abc import abstractmethod
from functools import partial, wraps
import os
from typing import Any, Callable, Iterable, Dict, Tuple, Sequence, Union

import flax.linen as nn
from flax import optim, jax_utils
from flax.serialization import to_bytes, from_bytes
from flax.core.scope import Scope
import jax
from jax import random, lax
from jax import numpy as np
from tqdm import tqdm


def _shard(x: Union[np.ndarray, Sequence[np.ndarray]]) -> Sequence[np.ndarray]:
    n = jax.device_count()
    return jax.tree_map(lambda y: y.reshape(n, -1, *y.shape[1:]), x)


@jax.pmap
def _parallel_split(key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    key, subkey = random.split(key)
    return key, subkey


def _save_as_bytes(path: str, obj: Any):
    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)

    with open(path, "wb") as f:
        f.write(to_bytes(obj))


def _load_from_bytes(path: str, target: Any):
    with open(path, "rb") as f:
        return from_bytes(target, f.read())


def _training_epoch(
    optimizer: optim.Optimizer,
    random_key: np.ndarray,
    training_step: Callable = lambda *x: None,
    training_loader: Sequence = (),
    eval_step: Callable = lambda *x: None,
    eval_loader: Sequence = (),
    eval_frequency: int = 1,
    progress_bar: bool = True,
    desc: str = "",
) -> Tuple[optim.Optimizer, np.ndarray]:
    total = len(training_loader) + eval_frequency * len(eval_loader)
    eval_every = len(training_loader) / eval_frequency
    get_progress_bar = partial(tqdm, disable=not progress_bar, leave=False)

    prog_bar = get_progress_bar(total=total, desc=desc)
    optimizer = optimizer.replicate()
    random_key = random.split(random_key, num=jax.device_count())

    def eval_epoch(eval_loader: Iterable, random_key: np.ndarray) -> np.ndarray:
        loader = get_progress_bar(eval_loader, desc="Valid", position=1)
        for batch in loader:
            random_key, subkey = _parallel_split(random_key)
            metrics = eval_step(optimizer.target, _shard(batch), subkey)

            prog_bar.update()
            if "loss" in metrics:
                loss = np.mean(metrics["loss"]).item()
                loader.set_postfix_str(f"loss={loss:.4f}", refresh=False)

        return random_key

    for i, batch in enumerate(training_loader, 1):
        random_key, subkey = _parallel_split(random_key)
        metrics, optimizer = training_step(optimizer, _shard(batch), subkey)

        prog_bar.update()
        if "loss" in metrics:
            loss = np.mean(metrics["loss"]).item()
            prog_bar.set_postfix_str(f"loss={loss:.4f}", refresh=False)

        if eval_loader and int(i % eval_every) == 0:
            eval_epoch(eval_loader, random_key=random_key)

    prog_bar.close()
    optimizer = optimizer.unreplicate()
    random_key = random_key[0]

    return optimizer, random_key


def _test_epoch(
    params: Dict[str, np.ndarray],
    random_key: np.ndarray,
    test_step: Callable = lambda *x: None,
    test_loader: Sequence = (),
    progress_bar: bool = True,
    desc: str = "",
) -> np.ndarray:
    loader = tqdm(test_loader, desc=desc, disable=not progress_bar)
    params = jax_utils.replicate(params)
    random_key = random.split(random_key, num=jax.device_count())

    for batch in loader:
        random_key, subkey = _parallel_split(random_key)
        metrics = test_step(params, _shard(batch), subkey)

        if "loss" in metrics:
            loss = np.mean(metrics["loss"]).item()
            loader.set_postfix_str(f"loss={loss:.4f}", refresh=False)

    random_key = random_key[0]
    return random_key


def _make_step_fn_differentiable(step_fn: Callable):
    @wraps(step_fn)
    def _differentiable_fn(*args, **kwargs) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        metrics = step_fn(*args, **kwargs)
        return metrics["loss"], metrics

    return _differentiable_fn


class FlaxseedModule(nn.Module):
    _scope = Scope(
        name="flaxseed",
        mutable=True,
        variables={"optimizer": None, "hparams": {}},
        rngs={"main": random.PRNGKey(10)},
    )

    @abstractmethod
    def init_params(self, key: np.ndarray) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def init_optimizer(self, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pass

    @property
    def random_key(self):
        return self._scope.rngs["main"]

    def set_random_key(self, value):
        self._scope.rngs["main"] = value

    @property
    def optimizer(self):
        optimizer_ = self._scope.get_variable("flaxseed", "optimizer")
        if optimizer_ is None:
            optimizer_ = self.init_optimizer(self.init_params(self.random_key))
            self._scope.put_variable("flaxseed", "optimizer", optimizer_)

        return optimizer_

    def set_optimizer(self, value):
        self._scope.put_variable("flaxseed", "optimizer", value)

    @property
    def hparams(self):
        return self._scope.get_variable("flaxseed", "hparams")

    def set_hparams(self, value: Dict):
        self._scope.put_variable("flaxseed", "hparams", value)

    @property
    def params(self):
        return self.optimizer.target

    def set_params(self, value: Dict[str, np.ndarray]):
        opt = self.optimizer
        breakpoint()
        opt["params"]

    def transform(self, inputs):
        return self.apply({"params": self.params}, inputs)

    def _get_checkpoint(self):
        return {
            "hparams": self.hparams,
            "optimizer": self.optimizer,
        }

    def save_checkpoint(self, path: str):
        _save_as_bytes(path, self._get_checkpoint())

    def load_checkpoint(self, path: str):
        checkpoint = _load_from_bytes(path, self._get_checkpoint())
        self.set_hparams(checkpoint["hparams"])
        self.set_optimizer(checkpoint["optimizer"])

    def save_optimizer(self, path: str):
        _save_as_bytes(path, self.optimizer)

    def load_optimizer(self, path: str):
        self.set_optimizer(_load_from_bytes(path, self.optimizer))

    def save_params(self, path: str):
        _save_as_bytes(path, self.params)

    def load_params(self, path: str):
        state_dict = self.optimizer.state_dict()
        state_dict["target"] = _load_from_bytes(path, self.params)
        self.set_optimizer(self.optimizer.restore_state(state_dict))

    @abstractmethod
    def training_step(
        self,
        params: Dict[str, np.ndarray],
        batch: Sequence[np.ndarray],
        random_key: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        pass

    def eval_step(
        self,
        params: Dict[str, np.ndarray],
        batch: Sequence[np.ndarray],
        random_key: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        pass

    def train(
        self,
        training_loader: Sequence,
        eval_loader: Sequence = (),
        max_epochs: int = 5,
        eval_frequency: int = 1,
    ):
        @partial(jax.pmap, axis_name="batch")
        def training_step_(
            optimizer: optim.Optimizer,
            batch: Sequence[np.ndarray],
            random_key: np.ndarray,
        ) -> Tuple[Dict[str, np.ndarray], optim.Optimizer]:
            step_fn = _make_step_fn_differentiable(self.training_step)
            step_fn = partial(step_fn, batch=batch, random_key=random_key)
            (_, metrics), grad = jax.value_and_grad(step_fn, has_aux=True)(
                optimizer.target
            )
            optimizer = optimizer.apply_gradient(lax.pmean(grad, axis_name="batch"))
            return metrics, optimizer

        eval_step_ = jax.pmap(self.eval_step)

        for i in range(1, max_epochs + 1):
            optimizer, random_key = _training_epoch(
                self.optimizer,
                random_key=self.random_key,
                training_step=training_step_,
                training_loader=training_loader,
                eval_step=eval_step_,
                eval_loader=eval_loader,
                eval_frequency=eval_frequency,
                desc=f"Epoch {i}",
            )
            self.set_optimizer(optimizer)
            self.set_random_key(random_key)

    def test(self, test_loader: Sequence):
        random_key = _test_epoch(
            self.params,
            random_key=self.random_key,
            test_step=jax.pmap(self.eval_step),
            test_loader=test_loader,
            desc="Test",
        )
        self.set_random_key(random_key)
