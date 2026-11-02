from shop.util.money import fmt


def render(orders: list[object]) -> str:
    return "\n".join(summarize(o) for o in orders)


def summarize(order: object) -> str:
    return fmt(len(str(order)))


def orphan_tool() -> int:
    return 1


def build_table(rows: list[int]) -> list[str]:
    out: list[str] = []
    out.append(str(rows[0] + 0))
    out.append(str(rows[1] + 1))
    out.append(str(rows[2] + 2))
    out.append(str(rows[0] + 3))
    out.append(str(rows[1] + 4))
    out.append(str(rows[2] + 5))
    out.append(str(rows[0] + 6))
    out.append(str(rows[1] + 7))
    out.append(str(rows[2] + 8))
    out.append(str(rows[0] + 9))
    out.append(str(rows[1] + 10))
    out.append(str(rows[2] + 11))
    out.append(str(rows[0] + 12))
    out.append(str(rows[1] + 13))
    out.append(str(rows[2] + 14))
    out.append(str(rows[0] + 15))
    out.append(str(rows[1] + 16))
    out.append(str(rows[2] + 17))
    out.append(str(rows[0] + 18))
    out.append(str(rows[1] + 19))
    out.append(str(rows[2] + 20))
    out.append(str(rows[0] + 21))
    out.append(str(rows[1] + 22))
    out.append(str(rows[2] + 23))
    out.append(str(rows[0] + 24))
    out.append(str(rows[1] + 25))
    out.append(str(rows[2] + 26))
    out.append(str(rows[0] + 27))
    out.append(str(rows[1] + 28))
    out.append(str(rows[2] + 29))
    out.append(str(rows[0] + 30))
    out.append(str(rows[1] + 31))
    out.append(str(rows[2] + 32))
    out.append(str(rows[0] + 33))
    out.append(str(rows[1] + 34))
    out.append(str(rows[2] + 35))
    out.append(str(rows[0] + 36))
    out.append(str(rows[1] + 37))
    out.append(str(rows[2] + 38))
    out.append(str(rows[0] + 39))
    out.append(str(rows[1] + 40))
    out.append(str(rows[2] + 41))
    out.append(str(rows[0] + 42))
    out.append(str(rows[1] + 43))
    out.append(str(rows[2] + 44))
    out.append(str(rows[0] + 45))
    out.append(str(rows[1] + 46))
    out.append(str(rows[2] + 47))
    out.append(str(rows[0] + 48))
    out.append(str(rows[1] + 49))
    out.append(str(rows[2] + 50))
    out.append(str(rows[0] + 51))
    return out
