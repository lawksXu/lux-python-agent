import asyncio

import pytest
from test_protocol_v2 import SESSION, USER, input_json
from test_store_v2 import intent, running

from lux_agent.store import COMMAND_TOPIC, AgentStore, InboundRecord


class Transport:
    def __init__(self, record):
        self.record = record
        self.polled = 0
        self.committed = []
        self.fail_commit = True
        self.fail_publish = True
        self.published = []

    async def poll(self):
        self.polled += 1
        return self.record

    async def commit(self, record):
        if self.fail_commit:
            self.fail_commit = False
            raise OSError("broker unavailable")
        self.committed.append(record)
        self.record = None

    async def publish(self, publication):
        if self.fail_publish:
            self.fail_publish = False
            raise OSError("broker unavailable")
        self.published.append(publication)

    async def close(self):
        pass


def test_ack_follows_durable_receipt_and_failed_offset_does_not_advance(tmp_path):
    async def scenario():
        from lux_agent.delivery import DeliveryWorker

        record = InboundRecord(
            COMMAND_TOPIC, 0, 3, f"{USER}:{SESSION}".encode(), input_json().encode()
        )
        transport = Transport(record)
        with AgentStore(tmp_path / "agent.db") as store:
            worker = DeliveryWorker(store, transport)
            with pytest.raises(OSError):
                await worker.step()
            assert len(store.commands(SESSION)) == 1
            await worker.step()
            assert transport.polled == 1 and transport.committed == [record]
            store.append_fact(intent(running()))
            await worker.step()
            publication = store.pending_publications(at=10**15)[0]
            assert publication.attempts == 1
            store.publication_failed(publication.id, publication.dispatch_generation)
            assert not transport.published

    asyncio.run(scenario())


def test_disk_failure_prevents_ack_and_bad_record_is_diagnosed_before_ack(tmp_path):
    async def scenario():
        import sqlite3

        from lux_agent.delivery import DeliveryWorker

        class UnavailableStore(AgentStore):
            unavailable = True

            def receive(self, record):
                if self.unavailable:
                    raise sqlite3.OperationalError("disk unavailable")
                return super().receive(record)

        record = InboundRecord(COMMAND_TOPIC, 0, 8, None, b"{invalid")
        transport = Transport(record)
        transport.fail_commit = False
        with UnavailableStore(tmp_path / "agent.db") as store:
            worker = DeliveryWorker(store, transport)
            with pytest.raises(sqlite3.OperationalError):
                await worker.step()
            assert not transport.committed
            store.unavailable = False
            await worker.step()
            assert transport.committed == [record] and transport.polled == 1
            assert store.receipt_error(record) is not None

    asyncio.run(scenario())


def test_kafka_transport_sets_coordinated_record_and_fetch_capacity(monkeypatch):
    from unittest.mock import Mock

    from lux_agent import delivery

    consumer, producer = Mock(), Mock()
    monkeypatch.setattr(delivery, "AIOKafkaConsumer", consumer)
    monkeypatch.setattr(delivery, "AIOKafkaProducer", producer)
    delivery.KafkaTransport("localhost:9092", "test-group")
    assert producer.call_args.kwargs["max_request_size"] == 33554432
    assert consumer.call_args.kwargs["max_partition_fetch_bytes"] == 33554432
    assert consumer.call_args.kwargs["fetch_max_bytes"] == 67108864
