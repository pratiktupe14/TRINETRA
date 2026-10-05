"""5-tuple flow aggregator for CyberHeal AI.

Responsibility: consume RawPacketRecord streams and group packets into
bidirectional or unidirectional flows using idle/active timeout logic.
Outputs raw flow containers ready for feature extraction.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

from .pcap_parser import RawPacketRecord

logger = logging.getLogger(__name__)

# TCP flag bit positions
_FLAG_FIN = 0x01
_FLAG_SYN = 0x02
_FLAG_RST = 0x04
_FLAG_PSH = 0x08
_FLAG_ACK = 0x10
_FLAG_URG = 0x20
_FLAG_ECE = 0x40
_FLAG_CWR = 0x80


class FlowKey(NamedTuple):
    """5-tuple identifying a network conversation."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int

    def canonical(self) -> FlowKey:
        """Return canonical key where lower endpoint is always first."""
        ep1 = (self.src_ip, self.src_port)
        ep2 = (self.dst_ip, self.dst_port)
        if ep1 <= ep2:
            return self
        return FlowKey(
            src_ip=self.dst_ip,
            dst_ip=self.src_ip,
            src_port=self.dst_port,
            dst_port=self.src_port,
            protocol=self.protocol,
        )

    def is_forward(self, packet: RawPacketRecord) -> bool:
        """Check if packet matches the forward direction of this key."""
        return (
            packet.src_ip == self.src_ip
            and packet.src_port == self.src_port
            and packet.dst_ip == self.dst_ip
            and packet.dst_port == self.dst_port
        )


@dataclass(slots=True)
class FlowRecord:
    """Accumulated raw flow data before feature extraction."""

    flow_id: str
    key: FlowKey
    flow_start_time: float  # POSIX seconds of first packet
    flow_end_time: float  # POSIX seconds of last packet
    last_packet_time: float  # used for timeout tracking

    # Forward direction (src_ip:src_port -> dst_ip:dst_port of the first packet)
    fwd_packets: list[RawPacketRecord] = field(default_factory=list)
    # Backward direction
    bwd_packets: list[RawPacketRecord] = field(default_factory=list)

    # Active / idle segment tracking
    active_start: float = 0.0
    last_active_time: float = 0.0
    active_durations: list[float] = field(default_factory=list)
    idle_durations: list[float] = field(default_factory=list)
    activity_threshold_s: float = 2.0  # gap > this -> new idle segment
    fin_seen_fwd: bool = False
    fin_seen_bwd: bool = False

    def add_packet(self, pkt: RawPacketRecord, is_forward: bool) -> None:
        """Append a packet to the appropriate direction list and update timing."""
        if is_forward:
            self.fwd_packets.append(pkt)
        else:
            self.bwd_packets.append(pkt)

        if pkt.protocol == 6 and (pkt.tcp_flags & _FLAG_FIN):
            if is_forward:
                self.fin_seen_fwd = True
            else:
                self.fin_seen_bwd = True

        gap = pkt.timestamp - self.last_packet_time
        if gap > self.activity_threshold_s:
            # Close active segment, open idle segment
            active_dur = self.last_packet_time - self.active_start
            if active_dur > 0:
                self.active_durations.append(active_dur)
            self.idle_durations.append(gap)
            self.active_start = pkt.timestamp
        self.last_packet_time = pkt.timestamp
        self.flow_end_time = pkt.timestamp

    @property
    def all_packets(self) -> list[RawPacketRecord]:
        return sorted(self.fwd_packets + self.bwd_packets, key=lambda p: p.timestamp)

    def finalise_active_idle(self) -> None:
        """Close the last active segment."""
        active_dur = self.last_packet_time - self.active_start
        if active_dur > 0:
            self.active_durations.append(active_dur)


def aggregate_flows(
    packets: Iterator[RawPacketRecord],
    idle_timeout_s: float = 30.0,
    active_timeout_s: float = 120.0,
    mode: str = "bidirectional",
    tcp_close_policy: str = "rst_or_timeout",
) -> Iterator[FlowRecord]:
    """Group a packet stream into flows using 5-tuple and timeout logic.

    A flow is terminated when:
    - The idle timeout expires (gap between consecutive packets exceeds
      idle_timeout_s), or
    - The active timeout expires (flow duration exceeds active_timeout_s), or
    - A terminating TCP flag is seen under the chosen tcp_close_policy.

    Args:
        packets: Iterator of RawPacketRecord objects.
        idle_timeout_s: Maximum idle gap between packets in seconds.
        active_timeout_s: Maximum total duration of a flow in seconds.
        mode: "bidirectional" or "unidirectional".
        tcp_close_policy: "rst_or_timeout" (default) or "fin_and_ack".

    Yields:
        Completed FlowRecord objects.
    """
    bidirectional = mode == "bidirectional"
    flows: dict[FlowKey, FlowRecord] = {}
    flow_counter = 0

    def _flush_flow(fkey: FlowKey) -> FlowRecord:
        record = flows.pop(fkey)
        record.finalise_active_idle()
        return record

    for pkt in packets:
        fwd_key = FlowKey(
            src_ip=pkt.src_ip,
            dst_ip=pkt.dst_ip,
            src_port=pkt.src_port,
            dst_port=pkt.dst_port,
            protocol=pkt.protocol,
        )

        fkey: FlowKey
        is_fwd: bool

        if bidirectional:
            bwd_key = FlowKey(
                src_ip=pkt.dst_ip,
                dst_ip=pkt.src_ip,
                src_port=pkt.dst_port,
                dst_port=pkt.src_port,
                protocol=pkt.protocol,
            )
            if fwd_key in flows:
                fkey = fwd_key
                is_fwd = True
            elif bwd_key in flows:
                fkey = bwd_key
                is_fwd = False
            else:
                fkey = fwd_key
                is_fwd = True
        else:
            fkey = fwd_key
            is_fwd = True

        if fkey in flows:
            flow = flows[fkey]

            idle_expired = (pkt.timestamp - flow.last_packet_time) > idle_timeout_s
            active_expired = (pkt.timestamp - flow.flow_start_time) > active_timeout_s

            if idle_expired or active_expired:
                yield _flush_flow(fkey)

        if fkey not in flows:
            flow_counter += 1
            flow_id = (
                f"{pkt.src_ip}:{pkt.src_port}-{pkt.dst_ip}:{pkt.dst_port}"
                f"-{pkt.protocol}-{flow_counter}"
            )
            flows[fkey] = FlowRecord(
                flow_id=flow_id,
                key=fkey,
                flow_start_time=pkt.timestamp,
                flow_end_time=pkt.timestamp,
                last_packet_time=pkt.timestamp,
                active_start=pkt.timestamp,
                last_active_time=pkt.timestamp,
            )
            is_fwd = True

        flow = flows[fkey]
        flow.add_packet(pkt, is_forward=is_fwd)

        if pkt.protocol == 6:
            should_close = False
            if tcp_close_policy == "rst_or_timeout":
                if pkt.tcp_flags & _FLAG_RST:
                    should_close = True
            elif tcp_close_policy == "fin_and_ack":
                if flow.fin_seen_fwd and flow.fin_seen_bwd:
                    should_close = True

            if should_close:
                yield _flush_flow(fkey)

    for remaining_key in list(flows.keys()):
        yield _flush_flow(remaining_key)
