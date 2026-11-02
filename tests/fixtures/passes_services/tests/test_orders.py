from orders.api import get_order


def test_get_order(client) -> None:
    assert client.get("/api/orders/1").status_code == 200
    assert get_order(1) == {"id": 1}
