"""Money formatting."""


class Money(int):
    """Whole cents."""


def fmt(cents: int) -> str:
    return f"${to_dollars(cents):.2f}"


def to_dollars(cents: int) -> float:
    return cents / 100
