from __future__ import annotations

from codegraph.services.channels import (
    detect_all_channels,
    detect_channels,
    python_signals,
)
from codegraph.services.sources import SourceFile, language_of


def _source(path: str, text: str) -> SourceFile:
    labels = language_of(path)
    assert labels is not None
    return SourceFile(path=path, language=labels[0], family=labels[1], text=text)


def _sites(path: str, text: str) -> set[tuple[str, str, str]]:
    sites = detect_channels(_source(path, text))
    return {(s.direction, s.name, s.broker) for s in sites if s.pending is None}


def test_socketio_eventemitter_and_process() -> None:
    found = _sites(
        "web/chat.ts",
        "import { Server } from 'socket.io';\n"
        "const ROOM_EVENT = 'room:joined';\n"
        "io.on('connection', (socket) => {\n"
        "  socket.on('chat message', onMessage);\n"
        "  io.to(room).emit(ROOM_EVENT, user);\n"
        "});\n"
        "process.on('SIGTERM', shutdown);\n"
        "stream.on('data', chunk => 1);\n"
        "this.$emit('update');\n",
    )
    assert found == {
        ("listens_on", "chat message", "socket.io"),
        ("emits", "room:joined", "socket.io"),
        ("listens_on", "process.SIGTERM", "process"),
    }
    emitter = _sites("bus.js", "bus.emit('order.placed', o);\nbus.on('order.placed', ship);\n")
    assert emitter == {
        ("emits", "order.placed", "eventemitter"),
        ("listens_on", "order.placed", "eventemitter"),
    }


def test_js_kafka_rabbit_redis_nats() -> None:
    found = _sites(
        "svc/q.js",
        "const { Kafka } = require('kafkajs');\n"
        "await producer.send({ topic: 'orders', messages });\n"
        "await consumer.subscribe({ topics: ['payments', 'refunds'] });\n"
        "ch.sendToQueue('emails', buf);\n"
        "ch.consume('emails', handleEmail);\n",
    )
    assert found == {
        ("emits", "orders", "kafka"),
        ("listens_on", "payments", "kafka"),
        ("listens_on", "refunds", "kafka"),
        ("emits", "emails", "rabbitmq"),
        ("listens_on", "emails", "rabbitmq"),
    }
    assert _sites(
        "r.ts", "import Redis from 'ioredis';\nredis.publish('news', m);\nsub.subscribe('news');\n"
    ) == {("emits", "news", "redis"), ("listens_on", "news", "redis")}


def test_python_celery_tasks_resolve_across_files() -> None:
    sources = [
        _source(
            "proj/tasks.py",
            "from celery import shared_task\n"
            "@shared_task\ndef send_email(to):\n    pass\n"
            "@app.task(name='billing.charge')\ndef charge(x):\n    pass\n",
        ),
        _source(
            "proj/views.py",
            "from proj.tasks import send_email\n"
            "def signup():\n    send_email.delay('a@b')\n"
            "    app.send_task('billing.charge', args=[1])\n"
            "    unknown_thing.delay()\n",
        ),
    ]
    sites = {(s.direction, s.name, s.file) for s in detect_all_channels(sources)}
    assert sites == {
        ("listens_on", "celery:proj.tasks.send_email", "proj/tasks.py"),
        ("listens_on", "celery:billing.charge", "proj/tasks.py"),
        ("emits", "celery:proj.tasks.send_email", "proj/views.py"),
        ("emits", "celery:billing.charge", "proj/views.py"),
    }


def test_python_signals_django_and_blinker() -> None:
    signals = _source(
        "shop/signals.py",
        "from django.dispatch import Signal\nfrom blinker import signal\n"
        "order_placed = Signal()\nuser_created = signal('user-created')\n",
    )
    assert python_signals(signals) == {
        "order_placed": "signal:order_placed",
        "user_created": "signal:user-created",
    }
    sources = [
        signals,
        _source(
            "shop/handlers.py",
            "from django.db.models.signals import post_save\n"
            "@receiver(post_save, sender=Order)\ndef audit(sender, **kw):\n    pass\n"
            "@receiver(order_placed)\ndef notify(sender, **kw):\n    pass\n"
            "user_created.connect(welcome)\n",
        ),
        _source("shop/orders.py", "def place():\n    order_placed.send(sender=None)\n"),
    ]
    sites = {(s.direction, s.name, s.handler) for s in detect_all_channels(sources)}
    assert sites == {
        ("listens_on", "django:post_save", "audit"),
        ("listens_on", "signal:order_placed", "notify"),
        ("listens_on", "signal:user-created", "welcome"),
        ("emits", "signal:order_placed", None),
    }


def test_python_kafka_pika_redis_nats_temporal() -> None:
    found = _sites(
        "w.py",
        "from kafka import KafkaProducer, KafkaConsumer\n"
        "TOPIC = 'clicks'\n"
        "producer.send(TOPIC, b'x')\n"
        "consumer = KafkaConsumer('views', 'clicks', group_id='g')\n"
        "channel.basic_publish(exchange='', routing_key='tasks', body=b'')\n"
        "channel.basic_consume(queue='tasks', on_message_callback=work)\n",
    )
    assert found == {
        ("emits", "clicks", "kafka"),
        ("listens_on", "views", "kafka"),
        ("listens_on", "clicks", "kafka"),
        ("emits", "tasks", "rabbitmq"),
        ("listens_on", "tasks", "rabbitmq"),
    }
    assert _sites(
        "n.py",
        "import nats\nawait nc.publish('updates', b'')\n"
        "await nc.subscribe('updates', cb=on_update)\n",
    ) == {("emits", "updates", "nats"), ("listens_on", "updates", "nats")}
    temporal = _sites(
        "flows.py",
        "@activity.defn\nasync def compose(name: str) -> str:\n    return name\n"
        "@workflow.defn(name='Greeting')\nclass GreetingWorkflow:\n"
        "    @workflow.run\n    async def run(self):\n"
        "        return await workflow.execute_activity(compose, 'x')\n"
        "await client.start_workflow(GreetingWorkflow.run, id='1', task_queue='q')\n",
    )
    assert temporal == {
        ("listens_on", "temporal:activity:compose", "temporal"),
        ("listens_on", "temporal:workflow:Greeting", "temporal"),
        ("emits", "temporal:activity:compose", "temporal"),
        ("emits", "temporal:workflow:GreetingWorkflow", "temporal"),
    }


def test_go_kafka_nats_redis_temporal() -> None:
    found = _sites(
        "svc/main.go",
        'import "github.com/nats-io/nats.go"\nconst topic = "events"\nfunc f() {\n'
        "\tmsg := &sarama.ProducerMessage{Topic: topic, Value: v}\n"
        '\tr := kafka.NewReader(kafka.ReaderConfig{Brokers: b, Topic: "audit"})\n'
        '\tnc.Publish("greet", []byte("hi"))\n'
        '\tnc.Subscribe("greet", func(m *nats.Msg) {})\n'
        "\tworkflow.ExecuteActivity(ctx, SendEmail, input)\n"
        "\tw.RegisterActivity(SendEmail)\n}\n",
    )
    assert found == {
        ("emits", "events", "kafka"),
        ("listens_on", "audit", "kafka"),
        ("emits", "greet", "nats"),
        ("listens_on", "greet", "nats"),
        ("emits", "temporal:activity:SendEmail", "temporal"),
        ("listens_on", "temporal:activity:SendEmail", "temporal"),
    }
    redis = _sites("r.go", 'import "github.com/redis/go-redis/v9"\nrdb.Publish(ctx, "chat", m)\n')
    assert redis == {("emits", "chat", "redis")}


def test_jvm_listeners_templates_and_spring_events() -> None:
    found = _sites(
        "src/main/java/demo/Orders.java",
        "public class Orders {\n"
        '  @KafkaListener(topics = {"orders", "returns"}, groupId = "g")\n'
        "  public void onOrder(String msg) {}\n"
        '  @RabbitListener(queues = "mail")\n  public void onMail(Message m) {}\n'
        "  @EventListener\n  public void created(OrderCreated event) {}\n"
        "  void place() {\n"
        '    kafkaTemplate.send("orders", order);\n'
        '    rabbitTemplate.convertAndSend("mail", msg);\n'
        "    publisher.publishEvent(new OrderCreated(this));\n"
        "  }\n}\n",
    )
    assert found == {
        ("listens_on", "orders", "kafka"),
        ("listens_on", "returns", "kafka"),
        ("listens_on", "mail", "rabbitmq"),
        ("listens_on", "event:OrderCreated", "spring-events"),
        ("emits", "orders", "kafka"),
        ("emits", "mail", "rabbitmq"),
        ("emits", "event:OrderCreated", "spring-events"),
    }


def test_csharp_confluent_kafka() -> None:
    assert _sites(
        "Worker.cs",
        'await producer.ProduceAsync("metrics", msg);\nconsumer.Subscribe("metrics");\n',
    ) == {("emits", "metrics", "kafka"), ("listens_on", "metrics", "kafka")}


def test_test_files_are_skipped_by_detect_all() -> None:
    sources = [_source("tests/test_bus.py", "bus.publish('x', 1)\n")]
    assert detect_all_channels(sources) == []
