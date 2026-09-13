"""Billing logic."""

from pkg import models
from .util import fmt


class Invoice(models.Base):
    """An invoice."""

    def total(self) -> int:
        return self.rate() * 2

    def rate(self) -> int:
        return 1


def charge(inv: Invoice) -> str:
    inv.save()
    return fmt(inv.total())


def unused_helper() -> None:
    return None
