from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Generic, TypeVar

T = TypeVar("T")

class BaseFactory(Generic[T], ABC):
    @abstractmethod
    def create(self, **kwargs) -> T:
        pass
