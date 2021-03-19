from abc import abstractmethod
from typing import Any, Iterable, Dict, Union


class Callback:
    def __init__(self, on: Union[str, Iterable[str]] = ()):
        if isinstance(on, str):
            self.on: Iterable[str] = (on,)
        elif isinstance(on, Iterable):
            self.on = list(on)
        else:
            raise ValueError("Kwarg 'on' must be either 'str' or 'Iterable[str]'.")

    @abstractmethod
    def __call__(self, context: Dict[str, Any]) -> None:
        pass
