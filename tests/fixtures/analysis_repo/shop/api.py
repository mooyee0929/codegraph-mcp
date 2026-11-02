"""HTTP API."""

from shop.orders import find_orders

app = object()


@app.route("/orders")
def list_orders() -> list[str]:
    """Every order id."""
    return [o.sku for o in find_orders()]
