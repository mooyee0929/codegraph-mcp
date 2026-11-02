"""Orders service HTTP API."""

from fastapi import APIRouter, FastAPI
from kafka import KafkaProducer

app = FastAPI()
router = APIRouter(prefix="/orders")
producer = KafkaProducer()

ORDER_CREATED = "orders.created"


@router.get("/{order_id}")
def get_order(order_id: int) -> dict[str, int]:
    return {"id": order_id}


@router.post("")
def create_order() -> dict[str, str]:
    publish_created()
    return {}


def publish_created() -> None:
    producer.send(ORDER_CREATED, b"{}")


app.include_router(router, prefix="/api")
