
"""Traffic window aggregation; no packet payload retention."""

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class PacketEvent:
    timestamp: float
    uav_id: int
    protocol: str
    size_bytes: int
    flow_id: str
    retransmission: bool = False


class TelemetryCollector:
    def __init__(self, window_seconds=2.0):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        self.window_seconds = window_seconds
        self.events = defaultdict(list)

    def ingest(self, event: PacketEvent):
        if event.size_bytes < 0:
            raise ValueError("Negative packet size")

        self.events[event.uav_id].append(event)

    def snapshot(self, now, *, clear=False):
        output = {}

        for uid, records in self.events.items():
            active = [
                e for e in records
                if now - self.window_seconds < e.timestamp <= now
            ]

            if not active:
                continue

            protocols = {
                p: sum(e.protocol.lower() == p for e in active)
                for p in ("tcp", "udp", "icmp")
            }

            output[uid] = {
                "packet_count": len(active),
                "byte_count": sum(e.size_bytes for e in active),
                "packet_rate": len(active) / self.window_seconds,
                "flow_count": len({e.flow_id for e in active}),
                "retransmission_rate": (
                    sum(e.retransmission for e in active) / len(active)
                ),
                "tcp_fraction": protocols["tcp"] / len(active),
                "udp_fraction": protocols["udp"] / len(active),
                "icmp_fraction": protocols["icmp"] / len(active),
            }

            if clear:
                self.events[uid] = [
                    e for e in records if e.timestamp > now
                ]

        return output
