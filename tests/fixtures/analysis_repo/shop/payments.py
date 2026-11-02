"""Payment gateways."""


class Gateway:
    """Talks to a payment provider."""

    def charge(self, cents: int) -> bool:
        return cents > 0


class StripeGateway(Gateway):
    """The provider we actually use."""

    def charge(self, cents: int) -> bool:
        return cents < 10_000


class Receipt:
    """Proof of payment; only ever named in annotations."""


def charge_card(cents: int) -> bool:
    gateway = StripeGateway()
    return gateway.charge(cents)


def refund(receipt: Receipt) -> bool:
    return charge_card(-1)
