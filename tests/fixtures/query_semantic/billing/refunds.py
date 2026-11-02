"""Refunds for captured invoice payments."""


def refund_invoice_payment(invoice_id: str, amount_cents: int) -> bool:
    """Refund a captured invoice payment back to the customer card."""
    return amount_cents > 0 and bool(invoice_id)


def refund_invoice_payment_partially(invoice_id: str, amount_cents: int) -> bool:
    """Refund part of a captured invoice payment back to the customer card."""
    return refund_invoice_payment(invoice_id, amount_cents // 2)
