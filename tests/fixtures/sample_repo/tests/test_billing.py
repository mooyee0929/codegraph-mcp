from pkg.billing import Invoice, charge


def test_charge() -> None:
    assert charge(Invoice()) == "$2"
