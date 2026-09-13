"""Domain models."""


class Base:
    """Root of the model hierarchy."""

    def save(self) -> None:
        validate(self)


def validate(obj: Base) -> bool:
    return obj is not None
