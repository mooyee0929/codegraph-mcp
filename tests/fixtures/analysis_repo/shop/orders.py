"""Orders."""

from shop.models import Base, MemoryRepository
from shop.payments import charge_card
from shop.util.money import Money, fmt


class Order(Base):
    """One order line."""

    def __init__(self, sku: str, quantity: int) -> None:
        self.sku = sku
        self.quantity = quantity

    @property
    def total(self) -> int:
        return self.quantity * 100


def place_order(sku: str, quantity: int) -> Order:
    """Create, persist and charge an order."""
    order = Order(sku, quantity)
    order.save()
    charge_card(order.total)
    print(fmt(order.total))
    return order


def find_orders() -> list[Order]:
    return MemoryRepository().all()


def price_label(amount: Money) -> str:
    return fmt(amount)


def _unused_private() -> None:
    return None
