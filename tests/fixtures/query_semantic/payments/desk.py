"""Support-desk actions that delegate to billing."""

from billing.refunds import refund_invoice_payment


def desk_refund_invoice_payment(invoice_id: str, amount_cents: int) -> bool:
    """Refund a captured invoice payment back to the customer card from the desk."""
    return refund_invoice_payment(invoice_id, amount_cents)
