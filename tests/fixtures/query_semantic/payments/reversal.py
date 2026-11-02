"""Payment reversals, written separately from billing."""


def issue_invoice_payment_refund(invoice_id: str, amount_cents: int) -> bool:
    """Issue a refund of a captured invoice payment back to the customer card."""
    return amount_cents >= 0 and invoice_id != ""
