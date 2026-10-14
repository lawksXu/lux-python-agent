"""Kafka ACK follows durable receipt; delivery retries original canonical facts."""

import asyncio
import logging
from typing import Protocol

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition

from lux_agent.protocol import MAX_KAFKA_FETCH_BYTES, MAX_KAFKA_RECORD_BYTES
from lux_agent.store import (
    COMMAND_TOPIC,
    EVENT_TOPIC,
    AgentStore,
    InboundRecord,
    Publication,
)

_LOGGER = logging.getLogger(__name__)


class OwnershipLost(RuntimeError):
    """A durably received record can be redelivered to its new partition owner."""


class CommandTransport(Protocol):
    async def poll(self) -> InboundRecord | None: ...
    async def commit(self, record: InboundRecord) -> None: ...
    async def publish(self, publication: Publication) -> None: ...
    async def close(self) -> None: ...


class KafkaTransport:
    def __init__(self, bootstrap_servers: str, group_id: str) -> None:
        self._consumer = AIOKafkaConsumer(
            COMMAND_TOPIC,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            max_poll_records=1,
            max_poll_interval_ms=300000,
            max_partition_fetch_bytes=MAX_KAFKA_RECORD_BYTES,
            fetch_max_bytes=MAX_KAFKA_FETCH_BYTES,
        )
        self._producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            acks="all",
            enable_idempotence=True,
            request_timeout_ms=10000,
            max_request_size=MAX_KAFKA_RECORD_BYTES,
        )

    async def start(self) -> None:
        try:
            async with asyncio.timeout(20):
                await self._producer.start()
                await self._consumer.start()
        except BaseException:
            await self.close()
            raise

    async def poll(self) -> InboundRecord | None:
        try:
            record = await asyncio.wait_for(self._consumer.getone(), timeout=0.2)
        except TimeoutError:
            return None
        return InboundRecord(
            record.topic, record.partition, record.offset, record.key, record.value
        )

    async def commit(self, record: InboundRecord) -> None:
        partition = TopicPartition(record.topic, record.partition)
        if partition not in self._consumer.assignment():
            raise OwnershipLost("Command partition ownership changed")
        async with asyncio.timeout(10):
            await self._consumer.commit({partition: record.offset + 1})

    async def publish(self, publication: Publication) -> None:
        async with asyncio.timeout(10):
            await self._producer.send_and_wait(
                EVENT_TOPIC,
                key=publication.partition_key.encode(),
                value=publication.envelope.encode(),
            )

    async def close(self) -> None:
        try:
            async with asyncio.timeout(10):
                await self._consumer.stop()
        finally:
            async with asyncio.timeout(10):
                await self._producer.stop()


class DeliveryWorker:
    def __init__(self, store: AgentStore, transport: CommandTransport) -> None:
        self._store = store
        self._transport = transport
        self._record: InboundRecord | None = None

    async def step(self) -> None:
        if self._record is None:
            self._record = await self._transport.poll()
        if self._record is not None:
            # Retain a failed offset so no later offset acknowledges it implicitly.
            self._store.receive(self._record)
            try:
                await self._transport.commit(self._record)
            except OwnershipLost:
                self._record = None
            else:
                self._record = None
        for publication in self._store.pending_publications():
            try:
                await self._transport.publish(publication)
            except Exception as error:
                self._store.publication_failed(
                    publication.id,
                    publication.dispatch_generation,
                    type(error).__name__,
                )
                _LOGGER.warning(
                    "Agent event publication deferred (%s)", type(error).__name__
                )
            else:
                self._store.publication_delivered(
                    publication.id, publication.dispatch_generation
                )

    async def run(self) -> None:
        try:
            while True:
                try:
                    await self.step()
                except Exception as error:
                    _LOGGER.warning(
                        "Agent delivery deferred (%s)", type(error).__name__
                    )
                    await asyncio.sleep(1)
                await asyncio.sleep(0.01)
        finally:
            await self._transport.close()
