import os
import pickle
from subprocess import check_call
from typing import Any, Dict, Iterable, List, Union

from flaxseed.callbacks.base import Callback


class Checkpoint(Callback):
    def __init__(
        self,
        on: Union[str, Iterable[str]] = "validation",
        root: str = "./checkpoints",
        filename: str = "{epoch}-{step}.ckpt",
        monitor: str = "loss",
        mode: str = "min",
        n: int = 1,
    ):
        super().__init__(on=on)
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.n = n

        self._codex: Dict[str, str] = {
            "epoch": "__EPOCH__",
            "step": "__STEP__",
            monitor: monitor,
        }
        self.checkpoints: List[Dict[str, Any]] = []

    def _format_filename(self, context: Dict[str, Any]) -> str:
        filename = self.filename
        for key in self._codex.keys():
            filename = filename.replace("{" + key, f"{key}=" + "{" + key)

        targets = {k: context[v] for k, v in self._codex.items() if k in filename}
        return filename.format(**targets)

    def _evaluate(self, checkpoint: Dict[str, Any]) -> Union[float, int]:
        if self.mode == "max":
            return checkpoint["monitor"]
        elif self.mode == "min":
            return -checkpoint["monitor"]
        else:
            raise ValueError(f"Mode '{self.mode}' not in ['max', 'min'].")

    def remove_checkpoint(self, idx: int) -> None:
        checkpoint = self.checkpoints.pop(idx)
        check_call(["rm", checkpoint["path"]])

    def save_checkpoint(self, path: str, context: Dict[str, Any]) -> None:
        with open(path, "wb") as f:
            pickle.dump(context, f)

    def add_checkpoint(
        self, checkpoint: Dict[str, str], context: Dict[str, Any]
    ) -> None:
        self.checkpoints.append(checkpoint)
        self.save_checkpoint(checkpoint["path"], context)
        self.checkpoints = sorted(self.checkpoints, key=self._evaluate)

    @property
    def best_checkpoint(self):
        return self.checkpoints[-1]

    def __call__(self, context: Dict[str, Any]) -> None:
        checkpoint = {
            "path": os.path.join(self.root, self._format_filename(context)),
            "monitor": context[self.monitor],
        }

        if len(self.checkpoints) < self.n:
            self.add_checkpoint(checkpoint, context)
        elif self._evaluate(checkpoint) > self._evaluate(self.checkpoints[0]):
            self.remove_checkpoint(0)
            self.add_checkpoint(checkpoint, context)


if __name__ == "__main__":
    import jax.numpy as np

    checkpoint = Checkpoint(n=2)
    checkpoint({"__EPOCH__": 0, "__STEP__": 128, "loss": 0.1, "__STATE_DICT__": np.ones((128, 128))})
    checkpoint({"__EPOCH__": 1, "__STEP__": 128, "loss": 0.01, "__STATE_DICT__": np.ones((128, 128))})
    checkpoint({"__EPOCH__": 2, "__STEP__": 128, "loss": 0.001, "__STATE_DICT__": np.ones((128, 128))})
