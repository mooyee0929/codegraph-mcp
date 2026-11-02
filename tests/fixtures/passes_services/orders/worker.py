"""Consumes order events and runs background jobs."""

from celery import shared_task
from kafka import KafkaConsumer


def consume() -> None:
    for message in KafkaConsumer("orders.created"):
        send_receipt.delay(message.value)


@shared_task
def send_receipt(payload: bytes) -> None:
    print(payload)
