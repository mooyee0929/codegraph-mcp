from shop.orders import place_order


def test_place_order() -> None:
    assert place_order("x", 1).quantity == 1
