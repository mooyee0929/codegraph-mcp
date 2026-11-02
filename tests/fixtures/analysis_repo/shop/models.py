"""Persistence base classes."""

from abc import ABC, abstractmethod


class Base:
    """Root of the model hierarchy."""

    def save(self) -> None:
        validate(self)

    def __repr__(self) -> str:
        return type(self).__name__


def validate(obj: Base) -> bool:
    return obj is not None


class Repository(ABC):
    """Where models live."""

    @abstractmethod
    def all(self) -> list[Base]:
        """Every stored model."""


class MemoryRepository(Repository):
    """Keeps models in a list."""

    def all(self) -> list[Base]:
        return []
