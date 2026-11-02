def sum_invoice(lines: list[dict[str, float]], vat: float) -> float:
    acc = 0.0
    for line in lines:
        if line.get("qty", 0) <= 0:
            continue
        amount = line["unit"] * line["qty"]
        if line.get("rebate"):
            amount = amount - amount * line["rebate"]
        acc += amount
    tax = acc * vat
    return round(acc + tax, 2)
