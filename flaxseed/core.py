from abc import abstractmethod
from functools import partial, wraps
from typing import Any, Callable, Iterable, Dict, Tuple, Sequence, Union

import flax.linen as nn
from flax import optim, jax_utils
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


# def _load_checkpoint(path: str) -> Dict:
#     with open(path, "rb") as f:
#         context = f.read()

#     optimizer = context["optimizer"]

#     optimizer_def = checkpoint["optimizer_def"]
#     target_bytes = checkpoint["target"]
#     state_bytes = checkpoint["state"]


def _execute_callbacks(callbacks: Iterable, on: str, context: Dict[str, Any]):
    callbacks_ = filter(lambda callback: on in callback.on, callbacks)
    for callback in callbacks_:
        callback(context)


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


def train(
    optimizer: optim.Optimizer,
    training_step: Callable,
    training_loader: Sequence,
    eval_step: Callable = lambda *x: None,
    eval_loader: Sequence = (),
    random_key: np.ndarray = random.PRNGKey(0),
    max_epochs: int = 5,
    eval_frequency: int = 1,
):
    @partial(jax.pmap, axis_name="batch")
    def training_step_(
        optimizer: optim.Optimizer,
        batch: Sequence[np.ndarray],
        random_key: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], optim.Optimizer]:
        step_fn = _make_step_fn_differentiable(training_step)
        step_fn = partial(step_fn, batch=batch, random_key=random_key)
        (_, metrics), grad = jax.value_and_grad(step_fn, has_aux=True)(optimizer.target)
        optimizer = optimizer.apply_gradient(lax.pmean(grad, axis_name="batch"))
        return metrics, optimizer

    eval_step_ = jax.pmap(eval_step)

    for i in range(1, max_epochs + 1):
        optimizer, random_key = _training_epoch(
            optimizer,
            random_key=random_key,
            training_step=training_step_,
            training_loader=training_loader,
            eval_step=eval_step_,
            eval_loader=eval_loader,
            eval_frequency=eval_frequency,
            desc=f"Epoch {i}",
        )

    return optimizer, random_key


def test(
    optimizer: optim.Optimizer,
    test_step: Callable,
    test_loader: Sequence,
    random_key: np.ndarray = random.PRNGKey(0),
):
    random_key = _test_epoch(
        optimizer.target,
        random_key=random_key,
        test_step=jax.pmap(test_step),
        test_loader=test_loader,
        desc="Test",
    )

    return random_key


class FlaxseedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._random_key = random.PRNGKey(0)
        self._optimizer = None

    @abstractmethod
    def init_params(self, key: np.ndarray) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def init_optimizer(self, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pass

    @property
    def random_key(self):
        if not hasattr(self, "_random_key"):
            self._random_key = random.PRNGKey(0)
        return self._random_key

    @property
    def optimizer(self):
        if not hasattr(self, "_optimizer"):
            self._optimizer = None

        if self._optimizer is None:
            params = self.init_params(self.random_key)
            self._optimizer = self.init_optimizer(params)
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def params(self):
        return self.optimizer.target

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
            self.optimizer, self._random_key = _training_epoch(
                self.optimizer,
                random_key=self.random_key,
                training_step=training_step_,
                training_loader=training_loader,
                eval_step=eval_step_,
                eval_loader=eval_loader,
                eval_frequency=eval_frequency,
                desc=f"Epoch {i}",
            )

    def test(self, test_loader: Sequence):
        self._random_key = _test_epoch(
            self.optimizer.target,
            random_key=self.random_key,
            test_step=jax.pmap(self.eval_step),
            test_loader=test_loader,
            desc="Test",
        )
