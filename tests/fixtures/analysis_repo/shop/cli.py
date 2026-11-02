"""Command line front end."""

from shop.orders import place_order
from shop.report import render


def run_cli() -> None:
    """Place a demo order and print the report."""
    order = place_order("widget", 3)
    print(render([order]))
