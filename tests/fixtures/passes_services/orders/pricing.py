def compute_total(items: list[dict[str, float]], tax_rate: float) -> float:
    subtotal = 0.0
    for item in items:
        if item.get("quantity", 0) <= 0:
            continue
        price = item["price"] * item["quantity"]
        if item.get("discount"):
            price = price - price * item["discount"]
        subtotal += price
    tax = subtotal * tax_rate
    return round(subtotal + tax, 2)


def describe(order_id: int, status: str) -> str:
    parts = [f"order {order_id}"]
    while len(parts) < 3:
        try:
            parts.append(status.upper())
        except AttributeError:
            break
    with open("/dev/null", "w") as sink:
        sink.write(" ".join(parts))
    return " ".join(parts)
