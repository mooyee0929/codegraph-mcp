"""Message channels: who publishes to a topic or event name and who listens on it.

A :class:`ChannelSite` records one end of a channel. Names are the topic or event
string itself (``orders.created``) so producers and consumers written against different
client libraries, or in different services, meet on the same ``channel:<name>`` node.
Names that are not topics carry a namespace instead: ``process.SIGTERM``,
``django:post_save``, ``signal:<name>`` (Django ``Signal()`` / blinker),
``celery:<task>``, ``temporal:workflow:<Name>``, ``temporal:activity:<name>`` and
``event:<Type>`` (Spring application events).

Topic arguments may be literals, same-file string constants or lists of either. Celery
``.delay()`` calls and Django signal ``.send()`` calls name a Python object rather than a
string; those sites are emitted as *pending* and :func:`resolve_pending` maps them to a
task or signal defined anywhere in the repository.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal

from codegraph.parsers.base import module_qname_for
from codegraph.services.route_detectors import METHOD_HEADER, annotation_groups, handler_ref
from codegraph.services.sources import SourceFile
from codegraph.services.text import (
    Arg,
    keyword_arg,
    matching_close,
    module_constants,
    pick_arg,
    split_args,
    string_literal,
)

Direction = Literal["emits", "listens_on"]


@dataclass(frozen=True, slots=True)
class ChannelSite:
    name: str
    direction: Direction
    broker: str
    file: str
    line: int
    language: str
    handler: str | None = None
    decorated: bool = False
    pending: str | None = None
    """``celery`` / ``signal`` when ``name`` is still a Python reference to resolve."""


_BUILTIN_EVENTS = frozenset(
    [
        "data",
        "end",
        "error",
        "close",
        "finish",
        "drain",
        "readable",
        "connect",
        "connection",
        "disconnect",
        "message",
        "open",
        "ready",
        "listening",
        "request",
        "response",
        "upgrade",
        "timeout",
        "exit",
        "pipe",
        "unpipe",
        "abort",
        "line",
        "click",
        "change",
        "input",
        "submit",
        "keydown",
        "keyup",
        "keypress",
        "mousedown",
        "mouseup",
        "mousemove",
        "mouseover",
        "mouseout",
        "resize",
        "scroll",
        "load",
        "unload",
        "focus",
        "blur",
        "success",
        "complete",
        "progress",
        "beforeunload",
    ]
)
_DJANGO_SIGNALS = frozenset(
    [
        "pre_init",
        "post_init",
        "pre_save",
        "post_save",
        "pre_delete",
        "post_delete",
        "m2m_changed",
        "class_prepared",
        "pre_migrate",
        "post_migrate",
        "request_started",
        "request_finished",
        "got_request_exception",
        "setting_changed",
        "connection_created",
    ]
)
_TOPIC_TEXT = re.compile(r"^[\w.:/*>#{}$-]+(?: [\w.:/*>#{}$-]+)*$")
_NOT_A_SIGNAL = re.compile(
    r"(?i)produc|kafka|sock|conn|client|session|channel|queue|ws|mail|^self$"
)


def _topics(value: str, family: str, constants: dict[str, str]) -> list[str]:
    value = value.strip()
    if value[:1] in "[{" or value.startswith(
        ("[]string{", "List.of(", "listOf(", "arrayOf(", "new String[]")
    ):
        open_at = min(i for i in (value.find("["), value.find("{"), value.find("(")) if i != -1)
        if value.startswith("[]string{"):
            open_at = value.find("{")
        items = split_args(value, open_at, family) or []
        return [t for item in items for t in _topics(item.value, family, constants)]
    literal = string_literal(value, family)
    if literal is not None:
        text = re.sub(
            r"\$\{\s*([\w.]+)\s*\}",
            lambda m: constants.get(m.group(1), "{" + m.group(1) + "}"),
            literal.content,
        )
        return [text] if text and _TOPIC_TEXT.match(text) else []
    name = re.split(r"\s*(?:\.|::|->)\s*", value)[-1].lstrip("$")
    if name in constants:
        return [constants[name]] if _TOPIC_TEXT.match(constants[name]) else []
    return []


@dataclass(slots=True)
class _Scan:
    source: SourceFile
    constants: dict[str, str]
    sites: list[ChannelSite]

    def add(
        self,
        offset: int,
        name: str,
        direction: Direction,
        broker: str,
        handler: str | None = None,
        decorated: bool = False,
        pending: str | None = None,
    ) -> None:
        self.sites.append(
            ChannelSite(
                name=name,
                direction=direction,
                broker=broker,
                file=self.source.path,
                line=self.source.line(offset),
                language=self.source.language,
                handler=handler,
                decorated=decorated,
                pending=pending,
            )
        )

    def calls(self, pattern: re.Pattern[str]) -> Iterable[tuple[re.Match[str], list[Arg]]]:
        text = self.source.text
        for match in pattern.finditer(text):
            if self.source.in_comment(match.start()):
                continue
            args = split_args(text, match.end() - 1, self.source.family)
            if args is not None:
                yield match, args

    def topics(self, arg: Arg | None) -> list[str]:
        return _topics(arg.value, self.source.family, self.constants) if arg is not None else []


def _broker_for(text: str, receiver: str = "") -> str:
    lowered = text.lower()
    hint = receiver.lower()
    if "kafka" in hint or "producer" in hint or "consumer" in hint:
        return "kafka"
    if "nats" in hint or re.search(r"\bnc\b", hint):
        return "nats"
    if "redis" in hint or "pubsub" in hint:
        return "redis"
    for broker, needle in (
        ("nats", "nats"),
        ("redis", "redis"),
        ("kafka", "kafka"),
        ("rabbitmq", "amqp"),
        ("rabbitmq", "pika"),
    ):
        if needle in lowered:
            return broker
    return "pubsub"


# -- generic publish / subscribe, every family -----------------------------------

_PUBLISH = re.compile(r"(?P<recv>[\w$.@>-]*?)\s*(?:\.|->)\s*(?P<fn>publish|Publish)\s*\(")
_SUBSCRIBE = re.compile(
    r"(?P<recv>[\w$.@>-]*?)\s*(?:\.|->)\s*(?P<fn>subscribe|Subscribe|psubscribe|PSubscribe|"
    r"QueueSubscribe|ChanSubscribe)\s*\("
)


def _generic_pubsub(scan: _Scan) -> None:
    text = scan.source.text
    for match, args in scan.calls(_PUBLISH):
        candidates = [
            a for a in args[:3] if not re.fullmatch(r"ctx|context|context\.\w+\(\)", a.value)
        ]
        if len(candidates) >= 3 and "amqp" in text.lower():
            exchange, key = scan.topics(candidates[0]), scan.topics(candidates[1])
            names = exchange or key
            broker = "rabbitmq"
        else:
            names = scan.topics(candidates[0]) if candidates else []
            broker = _broker_for(text, match.group("recv"))
        for name in names:
            scan.add(match.start(), name, "emits", broker)
    for match, args in scan.calls(_SUBSCRIBE):
        candidates = [a for a in args if not re.fullmatch(r"ctx|context|context\.\w+\(\)", a.value)]
        if not candidates or candidates[0].value.startswith("{") and scan.source.family == "js":
            continue
        handler_arg = keyword_arg(args, "cb", "callback") or (
            candidates[-1] if len(candidates) > 1 else None
        )
        handler = handler_ref(handler_arg.text, scan.source.family) if handler_arg else None
        broker = _broker_for(text, match.group("recv"))
        names = scan.topics(candidates[0])
        if candidates[0].value.startswith("[") or match.group("fn") in {"subscribe", "psubscribe"}:
            names = [
                n
                for a in candidates
                if not a.value.startswith(("lambda", "func", "("))
                for n in scan.topics(a)
            ]
        for name in names:
            scan.add(match.start(), name, "listens_on", broker, handler=handler)


# -- JavaScript / TypeScript ------------------------------------------------------

_JS_EMIT = re.compile(
    r"(?P<recv>[\w$.]+(?:\([^()\n]*\))?(?:\s*\.\s*[\w$]+(?:\([^()\n]*\))?)*)\s*\.\s*emit\s*\("
)
_JS_ON = re.compile(r"(?P<recv>[\w$.]+)\s*\.\s*(?P<fn>on|once|addListener|prependListener)\s*\(")
_JS_KAFKA = re.compile(r"(?P<recv>[\w$.]+)\s*\.\s*(?P<fn>send|sendBatch|subscribe)\s*\((?=\s*\{)")
_JS_AMQP = re.compile(r"(?P<recv>[\w$.]+)\s*\.\s*(?P<fn>sendToQueue|consume|assertQueue)\s*\(")
_JS_TEMPORAL_WORKFLOW = re.compile(
    r"\.\s*workflow\s*\.\s*(?:start|execute|signalWithStart)\s*\(|\.\s*(?:startChild|executeChild)\s*\("
)
_JS_PROXY_ACTIVITIES = re.compile(r"const\s*\{([^}]*)\}\s*=\s*proxyActivities\b")


def _detect_js(scan: _Scan) -> None:
    text = scan.source.text
    socket_file = "socket.io" in text
    for match, args in scan.calls(_JS_EMIT):
        recv = match.group("recv")
        if recv.endswith("$") or not args:
            continue
        broker = (
            "socket.io"
            if socket_file or re.search(r"(?i)socket|\bio\b|nsp", recv)
            else "eventemitter"
        )
        for name in scan.topics(args[0]):
            if recv.split(".")[0] == "process":
                scan.add(match.start(), f"process.{name}", "emits", "process")
            elif name not in _BUILTIN_EVENTS:
                scan.add(match.start(), name, "emits", broker)
    for match, args in scan.calls(_JS_ON):
        recv = match.group("recv")
        if not args:
            continue
        handler = handler_ref(args[1].text, "js") if len(args) > 1 else None
        for name in scan.topics(args[0]):
            if recv == "process":
                scan.add(match.start(), f"process.{name}", "listens_on", "process", handler=handler)
            elif name not in _BUILTIN_EVENTS:
                broker = (
                    "socket.io"
                    if socket_file or re.search(r"(?i)socket|\bio\b", recv)
                    else "eventemitter"
                )
                scan.add(match.start(), name, "listens_on", broker, handler=handler)
    for match, args in scan.calls(_JS_KAFKA):
        obj = args[0].value if args else ""
        props = split_args(obj, 0, "js") or []
        topic_arg = keyword_arg(props, "topic", "topics")
        direction: Direction = "listens_on" if match.group("fn") == "subscribe" else "emits"
        for name in scan.topics(topic_arg):
            scan.add(match.start(), name, direction, "kafka")
    for match, args in scan.calls(_JS_AMQP):
        if match.group("fn") == "assertQueue" or not args:
            continue
        direction = "listens_on" if match.group("fn") == "consume" else "emits"
        handler = (
            handler_ref(args[1].text, "js") if direction == "listens_on" and len(args) > 1 else None
        )
        for name in scan.topics(args[0]):
            scan.add(match.start(), name, direction, "rabbitmq", handler=handler)
    for match, args in scan.calls(_JS_TEMPORAL_WORKFLOW):
        first = args[0].value if args else ""
        literal = string_literal(first, "js")
        name = literal.content if literal else (handler_ref(first, "js") or "").split(".")[-1]
        if name:
            scan.add(match.start(), f"temporal:workflow:{name}", "emits", "temporal")
    for match in _JS_PROXY_ACTIVITIES.finditer(text):
        names = [n.split(":")[-1].strip() for n in match.group(1).split(",") if n.strip()]
        for name in names:
            for call in re.finditer(rf"(?<![\w$.]){re.escape(name)}\s*\(", text[match.end() :]):
                scan.add(
                    match.end() + call.start(), f"temporal:activity:{name}", "emits", "temporal"
                )


# -- Python -----------------------------------------------------------------------

_PY_TASK = re.compile(
    r"^[ \t]*@(?:(?:[\w.]+\.)?task|shared_task|(?:[\w.]+\.)?periodic_task)\b(?P<args>\s*\()?", re.M
)
_PY_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+(\w+)", re.M)
_PY_DELAY = re.compile(r"(?<![\w.])(?P<ref>[A-Za-z_][\w.]*)\s*\.\s*(?:delay|apply_async)\s*\(")
_PY_SEND_TASK = re.compile(r"(?<![\w])[\w.]*\.\s*send_task\s*\(")
_PY_SIGNAL_DEF = re.compile(
    r"^(?P<var>\w+)\s*=\s*(?:[\w.]+\.)?(?:(?P<django>Signal|ModelSignal)\s*\(|signal\s*\(\s*(?P<lit>['\"][^'\"]+['\"]))",
    re.M,
)
_PY_SIGNAL_SEND = re.compile(
    r"(?<![\w])(?P<ref>[A-Za-z_][\w.]*)\s*\.\s*(?:send|send_robust|send_async)\s*\("
)
_PY_RECEIVER = re.compile(r"^[ \t]*@receiver\s*\(", re.M)
_PY_CONNECT = re.compile(r"(?<![\w@])(?P<ref>[A-Za-z_][\w.]*)\s*\.\s*connect\s*\(")
_PY_CONNECT_DECORATOR = re.compile(r"^[ \t]*@(?P<ref>[A-Za-z_][\w.]*)\s*\.\s*connect\b", re.M)
_PY_KAFKA_SEND = re.compile(r"(?P<recv>[\w.]+)\s*\.\s*(?:send|send_and_wait|produce)\s*\(")
_PY_KAFKA_CONSUMER = re.compile(r"(?<![\w.])(?:KafkaConsumer|AIOKafkaConsumer)\s*\(")
_PY_PIKA = re.compile(r"\.\s*(?P<fn>basic_publish|basic_consume)\s*\(")
_PY_TEMPORAL_DEFN = re.compile(
    r"^[ \t]*@(?P<kind>workflow|activity)\s*\.\s*defn\b(?P<args>\s*\()?", re.M
)
_PY_TEMPORAL_START = re.compile(
    r"\.\s*(?P<fn>start_workflow|execute_workflow|start_child_workflow|execute_child_workflow|"
    r"execute_activity|start_activity|execute_activity_method|start_activity_method|"
    r"execute_local_activity|start_local_activity)\s*\("
)


def _decorator_args(scan: _Scan, match: re.Match[str]) -> list[Arg]:
    if not match.group("args"):
        return []
    return split_args(scan.source.text, match.end() - 1, "python") or []


def _detect_python(scan: _Scan) -> None:
    text = scan.source.text
    module = module_qname_for(scan.source.path)
    for match in _PY_TASK.finditer(text):
        if scan.source.in_comment(match.start()):
            continue
        definition = _PY_DEF.search(text, match.end())
        if definition is None:
            continue
        explicit = keyword_arg(_decorator_args(scan, match), "name")
        names = scan.topics(explicit) if explicit else []
        name = names[0] if names else f"{module}.{definition.group(1)}"
        scan.add(
            match.start(),
            f"celery:{name}",
            "listens_on",
            "celery",
            handler=definition.group(1),
            decorated=True,
        )
    for match, _args in scan.calls(_PY_DELAY):
        scan.add(match.start(), match.group("ref"), "emits", "celery", pending="celery")
    for match, args in scan.calls(_PY_SEND_TASK):
        for name in scan.topics(pick_arg(args, 0, ("name",))):
            scan.add(match.start(), name, "emits", "celery", pending="celery-name")
    for match, _args in scan.calls(_PY_SIGNAL_SEND):
        if _NOT_A_SIGNAL.search(match.group("ref").split(".")[-1]):
            continue
        scan.add(match.start(), match.group("ref"), "emits", "signal", pending="signal")
    for match, args in scan.calls(_PY_RECEIVER):
        first = pick_arg(args, 0, ("signal",))
        refs = re.findall(r"[A-Za-z_][\w.]*", first.value) if first is not None else []
        definition = _PY_DEF.search(text, match.end())
        for ref in refs:
            scan.add(
                match.start(),
                ref,
                "listens_on",
                "signal",
                handler=definition.group(1) if definition else None,
                decorated=True,
                pending="signal",
            )
    for match, args in scan.calls(_PY_CONNECT):
        handler = handler_ref(args[0].text, "python") if args else None
        scan.add(
            match.start(),
            match.group("ref"),
            "listens_on",
            "signal",
            handler=handler,
            pending="signal",
        )
    for match in _PY_CONNECT_DECORATOR.finditer(text):
        definition = _PY_DEF.search(text, match.end())
        scan.add(
            match.start(),
            match.group("ref"),
            "listens_on",
            "signal",
            handler=definition.group(1) if definition else None,
            decorated=True,
            pending="signal",
        )
    if "kafka" in text.lower():
        for match, args in scan.calls(_PY_KAFKA_SEND):
            if re.search(r"(?i)produc|kafka|\bp$", match.group("recv")):
                for name in scan.topics(pick_arg(args, 0, ("topic",))):
                    scan.add(match.start(), name, "emits", "kafka")
        for match, args in scan.calls(_PY_KAFKA_CONSUMER):
            for arg in args:
                if arg.keyword is None:
                    for name in scan.topics(arg):
                        scan.add(match.start(), name, "listens_on", "kafka")
    for match, args in scan.calls(_PY_PIKA):
        if match.group("fn") == "basic_publish":
            exchange = scan.topics(keyword_arg(args, "exchange") or pick_arg(args, 0))
            key = scan.topics(keyword_arg(args, "routing_key") or pick_arg(args, 1))
            for name in exchange or key:
                scan.add(match.start(), name, "emits", "rabbitmq")
        else:
            callback = keyword_arg(args, "on_message_callback") or pick_arg(args, 1)
            handler = handler_ref(callback.text, "python") if callback else None
            for name in scan.topics(keyword_arg(args, "queue") or pick_arg(args, 0)):
                scan.add(match.start(), name, "listens_on", "rabbitmq", handler=handler)
    for match in _PY_TEMPORAL_DEFN.finditer(text):
        definition = _PY_DEF.search(text, match.end())
        if definition is None or scan.source.in_comment(match.start()):
            continue
        explicit = keyword_arg(_decorator_args(scan, match), "name")
        names = scan.topics(explicit) if explicit else []
        name = names[0] if names else definition.group(1)
        scan.add(
            match.start(),
            f"temporal:{match.group('kind')}:{name}",
            "listens_on",
            "temporal",
            handler=definition.group(1),
            decorated=True,
        )
    for match, args in scan.calls(_PY_TEMPORAL_START):
        kind = "workflow" if "workflow" in match.group("fn") else "activity"
        first = pick_arg(args, 0)
        if first is None:
            continue
        literal = string_literal(first.value, "python")
        if literal is not None:
            name = literal.content
        else:
            ref = handler_ref(first.value, "python") or ""
            parts = ref.split(".")
            name = parts[0] if kind == "workflow" and len(parts) > 1 else parts[-1]
        if name:
            scan.add(match.start(), f"temporal:{kind}:{name}", "emits", "temporal")


# -- Go ---------------------------------------------------------------------------

_GO_KAFKA_STRUCT = re.compile(r"\b(?P<kind>ProducerMessage|Writer|WriterConfig|ReaderConfig)\s*\{")
_GO_TOPIC_FIELD = re.compile(r"\bTopic\s*:\s*")
_GO_CONSUME = re.compile(r"\.\s*(?:ConsumePartition|Consume)\s*\(")
_GO_TEMPORAL = re.compile(
    r"\.\s*(?P<fn>ExecuteActivity|ExecuteWorkflow|ExecuteChildWorkflow|RegisterWorkflow|RegisterActivity)\s*\("
)


def _detect_go(scan: _Scan) -> None:
    text = scan.source.text
    for match in _GO_KAFKA_STRUCT.finditer(text):
        close = matching_close(text, match.end() - 1, "go")
        if close is None or scan.source.in_comment(match.start()):
            continue
        field = _GO_TOPIC_FIELD.search(text, match.end(), close)
        if field is None:
            continue
        value = re.match(r"[^,\n}]+", text[field.end() : close])
        direction: Direction = "listens_on" if match.group("kind") == "ReaderConfig" else "emits"
        for name in scan.topics(Arg(text=value.group(0), start=field.end()) if value else None):
            scan.add(match.start(), name, direction, "kafka")
    for match, args in scan.calls(_GO_CONSUME):
        for arg in args[:2]:
            for name in scan.topics(arg):
                scan.add(match.start(), name, "listens_on", "kafka")
    for match, args in scan.calls(_GO_TEMPORAL):
        fn = match.group("fn")
        if fn.startswith("Register"):
            ref = handler_ref(args[0].text, "go") if args else None
            if ref:
                kind = "workflow" if fn == "RegisterWorkflow" else "activity"
                scan.add(
                    match.start(),
                    f"temporal:{kind}:{ref.split('.')[-1]}",
                    "listens_on",
                    "temporal",
                    handler=ref,
                )
            continue
        index = 2 if fn == "ExecuteWorkflow" else 1
        target = pick_arg(args, index)
        if target is None:
            continue
        literal = string_literal(target.value, "go")
        name = (
            literal.content if literal else (handler_ref(target.value, "go") or "").split(".")[-1]
        )
        kind = "activity" if fn == "ExecuteActivity" else "workflow"
        if name:
            scan.add(match.start(), f"temporal:{kind}:{name}", "emits", "temporal")


# -- JVM --------------------------------------------------------------------------

_JVM_LISTENERS = {
    "KafkaListener": ("kafka", ("topics", "value", "topicPattern")),
    "RabbitListener": ("rabbitmq", ("queues", "value")),
    "JmsListener": ("jms", ("destination", "value")),
    "SqsListener": ("sqs", ("value", "queueNames")),
}
_JVM_SEND = re.compile(r"(?P<recv>[\w.]+?)\s*\.\s*(?P<fn>send|convertAndSend|sendDefault)\s*\(")
_JVM_PUBLISH_EVENT = re.compile(r"\.\s*publishEvent\s*\(\s*new\s+(?P<type>\w+)")
_JVM_STUB = re.compile(
    r"\bnew(?P<kind>Workflow|Activity|ChildWorkflow)Stub\s*\(\s*(?P<type>\w+)\s*\.\s*class"
)
_JVM_PARAM_TYPE = re.compile(
    r"\(\s*(?:final\s+)?(?:@\w+\s+)*(?P<type>[A-Z]\w*)(?:<[^>]*>)?\s+\w+\s*[,)]"
)
_JVM_CLASS = re.compile(
    r"\s*(?:(?:public|private|protected|abstract|final|static)\s+)*(?:class|interface)\s+(?P<name>\w+)"
)


def _detect_jvm(scan: _Scan) -> None:
    text = scan.source.text
    for group, end in annotation_groups(scan.source):
        header = text[end : end + 400]
        method = METHOD_HEADER.match(header)
        class_match = _JVM_CLASS.match(header)
        for annotation, annotation_args, at in group:
            if annotation in _JVM_LISTENERS and method is not None:
                broker, keys = _JVM_LISTENERS[annotation]
                parsed = split_args(f"({annotation_args or ''})", 0, "jvm") or []
                topic_arg = pick_arg(parsed, 0, keys)
                handler = method.group("kname") or method.group("jname")
                for name in scan.topics(topic_arg):
                    scan.add(at, name, "listens_on", broker, handler=handler, decorated=True)
            elif (
                annotation in {"EventListener", "TransactionalEventListener"} and method is not None
            ):
                types = re.findall(r"(\w+)\s*\.\s*class", annotation_args or "")
                if not types:
                    param = _JVM_PARAM_TYPE.search(header, method.end() - 1)
                    types = [param.group("type")] if param else []
                handler = method.group("kname") or method.group("jname")
                for event_type in types:
                    scan.add(
                        at,
                        f"event:{event_type}",
                        "listens_on",
                        "spring-events",
                        handler=handler,
                        decorated=True,
                    )
            elif (
                annotation in {"WorkflowInterface", "ActivityInterface"} and class_match is not None
            ):
                kind = "workflow" if annotation == "WorkflowInterface" else "activity"
                scan.add(
                    at,
                    f"temporal:{kind}:{class_match.group('name')}",
                    "listens_on",
                    "temporal",
                    handler=class_match.group("name"),
                    decorated=True,
                )
    for match, args in scan.calls(_JVM_SEND):
        recv = match.group("recv").lower()
        if "kafka" in recv:
            broker, first = "kafka", pick_arg(args, 0)
            names = scan.topics(first)
        elif "rabbit" in recv or "amqp" in recv or "jms" in recv:
            broker = "jms" if "jms" in recv else "rabbitmq"
            exchange = scan.topics(pick_arg(args, 0)) if len(args) >= 3 else []
            names = exchange or scan.topics(pick_arg(args, 1 if len(args) >= 3 else 0))
        else:
            continue
        for name in names:
            scan.add(match.start(), name, "emits", broker)
    for match in _JVM_PUBLISH_EVENT.finditer(text):
        if not scan.source.in_comment(match.start()):
            scan.add(match.start(), f"event:{match.group('type')}", "emits", "spring-events")
    for match in _JVM_STUB.finditer(text):
        kind = "activity" if match.group("kind") == "Activity" else "workflow"
        scan.add(match.start(), f"temporal:{kind}:{match.group('type')}", "emits", "temporal")


# -- C# -----------------------------------------------------------------------------

_CS_KAFKA = re.compile(r"(?P<recv>[\w.]+?)\s*\.\s*(?P<fn>ProduceAsync|Produce|Subscribe)\s*\(")


def _detect_csharp(scan: _Scan) -> None:
    for match, args in scan.calls(_CS_KAFKA):
        direction: Direction = "listens_on" if match.group("fn") == "Subscribe" else "emits"
        for name in scan.topics(pick_arg(args, 0)):
            scan.add(match.start(), name, direction, "kafka")


_FAMILY_DETECTORS = {
    "js": _detect_js,
    "python": _detect_python,
    "go": _detect_go,
    "jvm": _detect_jvm,
    "csharp": _detect_csharp,
}


def detect_channels(source: SourceFile) -> list[ChannelSite]:
    """Channel sites in one file; Celery and signal sites may still be pending."""
    if source.family in {"proto", "graphql"}:
        return []
    scan = _Scan(source=source, constants=module_constants(source.text, source.family), sites=[])
    detector = _FAMILY_DETECTORS.get(source.family)
    if detector is not None:
        detector(scan)
    if source.family != "csharp":
        _generic_pubsub(scan)
    unique: dict[tuple[str, str, int, str | None], ChannelSite] = {}
    for site in scan.sites:
        unique.setdefault((site.name, site.direction, site.line, site.pending), site)
    return list(unique.values())


def python_signals(source: SourceFile) -> dict[str, str]:
    """Module-level ``NAME = Signal()`` / ``NAME = signal("x")`` -> channel name."""
    found: dict[str, str] = {}
    for match in _PY_SIGNAL_DEF.finditer(source.text):
        literal = string_literal(match.group("lit"), "python") if match.group("lit") else None
        found[match.group("var")] = f"signal:{literal.content if literal else match.group('var')}"
    return found


def resolve_pending(sites: list[ChannelSite], sources: list[SourceFile]) -> list[ChannelSite]:
    """Turn Celery ``.delay`` / ``send_task`` and signal references into channel names.

    A Celery reference resolves when exactly one task in the repository has that bare
    function name; ``send_task("x")`` matches a task whose full name equals or ends
    with ``.x``. Signal references resolve against module-level signal definitions
    and the built-in Django signals; anything else is dropped rather than guessed.
    """
    tasks: defaultdict[str, set[str]] = defaultdict(set)
    task_names: set[str] = set()
    for site in sites:
        if site.broker == "celery" and site.pending is None and site.direction == "listens_on":
            full = site.name.removeprefix("celery:")
            task_names.add(full)
            if site.handler:
                tasks[site.handler].add(full)
    signals: defaultdict[str, set[str]] = defaultdict(set)
    for source in sources:
        if source.family == "python":
            for var, signal_name in python_signals(source).items():
                signals[var].add(signal_name)
    resolved: list[ChannelSite] = []
    for site in sites:
        if site.pending is None:
            resolved.append(site)
            continue
        name: str | None = None
        last = site.name.split(".")[-1]
        if site.pending == "celery":
            matches = tasks.get(last, set())
            name = f"celery:{next(iter(matches))}" if len(matches) == 1 else None
        elif site.pending == "celery-name":
            matches = {t for t in task_names if t == site.name or t.endswith("." + site.name)}
            name = f"celery:{next(iter(matches))}" if len(matches) == 1 else f"celery:{site.name}"
        elif site.pending == "signal":
            if last in _DJANGO_SIGNALS and last not in signals:
                name = f"django:{last}"
            elif len(signals.get(last, set())) == 1:
                name = next(iter(signals[last]))
        if name is not None:
            resolved.append(replace(site, name=name, pending=None))
    return resolved


def detect_all_channels(sources: list[SourceFile]) -> list[ChannelSite]:
    sites = [s for source in sources if not source.is_test for s in detect_channels(source)]
    return resolve_pending(sites, sources)


__all__ = [
    "ChannelSite",
    "detect_all_channels",
    "detect_channels",
    "python_signals",
    "resolve_pending",
]
