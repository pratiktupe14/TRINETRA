"""PCAP packet decoder for CyberHeal AI.

Responsibility: decode raw packet bytes from a PCAP file or byte buffer and
yield normalised, protocol-filtered packet records. No feature computation here.
"""

from __future__ import annotations

import io
import logging
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Scapy is imported with fallback
try:
    from scapy.layers.inet import ICMP, IP, TCP, UDP  # type: ignore[import-untyped]
    from scapy.layers.inet6 import IPv6  # type: ignore[import-untyped]
    from scapy.packet import Packet  # type: ignore[import-untyped]
    from scapy.utils import PcapReader  # type: ignore[import-untyped]

    _SCAPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCAPY_AVAILABLE = False


# Maps IANA protocol numbers to human-readable names
PROTOCOL_MAP: dict[int, str] = {1: "ICMP", 6: "TCP", 17: "UDP"}
SUPPORTED_PROTOCOLS: frozenset[int] = frozenset(PROTOCOL_MAP)


@dataclass(slots=True)
class RawPacketRecord:
    """Minimal normalised representation of a single decoded packet.

    Fields 1-10 represent the core canonical packet representation.
    Fields 11-14 provide Phase 8 packet-level enrichment (TTL, TCP sequence numbers,
    window sizes) with backward-compatible defaults.
    """

    timestamp: float  # POSIX seconds (fractional)
    src_ip: str
    dst_ip: str
    src_port: int  # 0 for ICMP
    dst_port: int  # 0 for ICMP
    protocol: int  # IANA number: 1, 6, or 17
    ip_length: int  # total IP datagram length in bytes
    payload_length: int  # transport-layer payload in bytes
    tcp_flags: int  # bitmask; 0 for non-TCP
    ip_version: int  # 4 or 6
    # Phase 8 Packet-level enrichment fields (backward-compatible defaults)
    ttl: int | None = None  # IPv4 TTL or IPv6 hop limit; None if unavailable
    tcp_seq: int = 0  # TCP sequence number (0 for non-TCP)
    tcp_ack: int = 0  # TCP acknowledgment number (0 for non-TCP)
    tcp_window: int = 0  # TCP window size (0 for non-TCP)


def _decode_packet(
    pkt: Packet,
    allowed_protocols: frozenset[int],
    allowed_versions: frozenset[int],
) -> RawPacketRecord | None:
    """Extract a RawPacketRecord from a Scapy packet, or return None if skipped."""
    ip_version: int
    ttl: int | None = None
    if IP in pkt:
        ip_version = 4
        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        ip_len = int(ip_layer.len) if ip_layer.len is not None else len(ip_layer)
        protocol = int(ip_layer.proto)
        ttl_val = getattr(ip_layer, "ttl", None)
        ttl = int(ttl_val) if ttl_val is not None else None
    elif IPv6 in pkt:
        ip_version = 6
        ip_layer = pkt[IPv6]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        ip_len = (
            int(ip_layer.plen) + 40
            if ip_layer.plen is not None
            else len(ip_layer)
        )
        protocol = int(ip_layer.nh)
        hlim_val = getattr(ip_layer, "hlim", None)
        ttl = int(hlim_val) if hlim_val is not None else None
    else:
        return None

    if ip_version not in allowed_versions:
        return None
    if protocol not in allowed_protocols:
        return None

    src_port = 0
    dst_port = 0
    payload_len = 0
    tcp_flags = 0
    tcp_seq = 0
    tcp_ack = 0
    tcp_window = 0

    if protocol == 6 and TCP in pkt:
        tcp = pkt[TCP]
        src_port = int(tcp.sport)
        dst_port = int(tcp.dport)
        tcp_flags = int(tcp.flags)
        payload_len = len(tcp.payload)
        seq_val = getattr(tcp, "seq", 0)
        tcp_seq = int(seq_val) if seq_val is not None else 0
        ack_val = getattr(tcp, "ack", 0)
        tcp_ack = int(ack_val) if ack_val is not None else 0
        win_val = getattr(tcp, "window", 0)
        tcp_window = int(win_val) if win_val is not None else 0
    elif protocol == 17 and UDP in pkt:
        udp = pkt[UDP]
        src_port = int(udp.sport)
        dst_port = int(udp.dport)
        payload_len = len(udp.payload)
    elif protocol == 1 and ICMP in pkt:
        src_port = 0
        dst_port = 0
        payload_len = len(pkt[ICMP].payload)

    ts = float(pkt.time)

    return RawPacketRecord(
        timestamp=ts,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        ip_length=ip_len,
        payload_length=payload_len,
        tcp_flags=tcp_flags,
        ip_version=ip_version,
        ttl=ttl,
        tcp_seq=tcp_seq,
        tcp_ack=tcp_ack,
        tcp_window=tcp_window,
    )


def iter_packets(
    pcap_path: Path | str,
    include_protocols: tuple[int, ...] = (1, 6, 17),
    ip_versions: tuple[int, ...] = (4,),
) -> Iterator[RawPacketRecord]:
    """Iterate over decoded packet records from a PCAP file.

    The function filters to the requested IANA protocol numbers and IP
    versions, and skips malformed or unsupported packets without raising.

    Args:
        pcap_path: Path to a .pcap or .pcapng file.
        include_protocols: IANA protocol numbers to include.
                           Defaults to ICMP(1), TCP(6), UDP(17).
        ip_versions: IP versions to accept. Defaults to (4,) — IPv4 only.

    Yields:
        RawPacketRecord for each valid packet.

    Raises:
        FileNotFoundError: If pcap_path does not exist.
        ImportError: If scapy is not installed.
    """
    if not _SCAPY_AVAILABLE:
        raise ImportError(
            "scapy is required for PCAP parsing. Install it with: pip install scapy"
        )

    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    allowed = frozenset(include_protocols)
    allowed_versions = frozenset(ip_versions)

    logger.info("Opening PCAP: %s", pcap_path)
    with PcapReader(str(pcap_path)) as reader:
        pkt: Packet
        for pkt in reader:
            record = _decode_packet(pkt, allowed, allowed_versions)
            if record is not None:
                yield record


def iter_packets_from_bytes(
    data: bytes,
    include_protocols: tuple[int, ...] = (1, 6, 17),
    ip_versions: tuple[int, ...] = (4,),
) -> Iterator[RawPacketRecord]:
    """Iterate over decoded packet records from in-memory PCAP bytes.

    Useful for processing PCAP uploads directly from API requests without
    persisting them to permanent storage.

    Args:
        data: Raw byte string containing a PCAP file.
        include_protocols: IANA protocol numbers to include.
        ip_versions: IP versions to accept.

    Yields:
        RawPacketRecord for each valid packet.
    """
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
        tmp_name = tmp.name
        tmp.write(data)

    try:
        yield from iter_packets(
            tmp_name,
            include_protocols=include_protocols,
            ip_versions=ip_versions,
        )
    finally:
        p = Path(tmp_name)
        if p.exists():
            p.unlink(missing_ok=True)
