"""Synthetic PCAP generator for CyberHeal AI testing.

Writes a small, fully deterministic PCAP file that exercises TCP, UDP, and ICMP
flow pipelines without requiring external dataset downloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_BASE_TS = 1_499_162_400.0  # 2017-07-04 10:00:00 UTC

FLOWS_SPEC: list[dict[str, Any]] = [
    # Flow 0: TCP HTTP-like (port 80)
    {
        "src_ip": "192.168.1.10",
        "src_port": 54321,
        "dst_ip": "10.0.0.1",
        "dst_port": 80,
        "proto": 6,
        "packets": [
            (0.000, "fwd", 60, 0, 0x02),      # SYN
            (0.001, "bwd", 60, 0, 0x12),      # SYN-ACK
            (0.002, "fwd", 52, 0, 0x10),      # ACK
            (0.050, "fwd", 200, 148, 0x18),   # PSH+ACK
            (0.080, "bwd", 1500, 1448, 0x10), # Data
            (0.150, "fwd", 52, 0, 0x11),      # FIN+ACK
            (0.151, "bwd", 52, 0, 0x11),      # FIN+ACK
        ],
    },
    # Flow 1: TCP SSH-like (port 22)
    {
        "src_ip": "192.168.1.20",
        "src_port": 22222,
        "dst_ip": "10.0.0.2",
        "dst_port": 22,
        "proto": 6,
        "packets": [
            (0.000, "fwd", 60, 0, 0x02),
            (0.001, "bwd", 60, 0, 0x12),
            (0.002, "fwd", 52, 0, 0x10),
            (1.000, "fwd", 300, 248, 0x18),
            (1.050, "bwd", 300, 248, 0x18),
            (2.000, "fwd", 52, 0, 0x11),
            (2.001, "bwd", 52, 0, 0x11),
        ],
    },
    # Flow 2: UDP DNS-like (port 53)
    {
        "src_ip": "192.168.1.10",
        "src_port": 53001,
        "dst_ip": "8.8.8.8",
        "dst_port": 53,
        "proto": 17,
        "packets": [
            (0.000, "fwd", 72, 44, 0),
            (0.005, "bwd", 120, 92, 0),
        ],
    },
    # Flow 3: ICMP Echo / Reply
    {
        "src_ip": "192.168.1.30",
        "src_port": 0,
        "dst_ip": "10.0.0.1",
        "dst_port": 0,
        "proto": 1,
        "packets": [
            (0.000, "fwd", 84, 56, 0),
            (0.001, "bwd", 84, 56, 0),
        ],
    },
]


def create_synthetic_pcap(dest_path: Path | str) -> Path:
    """Build and write the synthetic PCAP to dest_path."""
    from scapy.layers.inet import ICMP, IP, TCP, UDP  # type: ignore[import-untyped]
    from scapy.packet import Raw  # type: ignore[import-untyped]
    from scapy.utils import wrpcap  # type: ignore[import-untyped]

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    packets = []
    for flow in FLOWS_SPEC:
        src_ip = flow["src_ip"]
        dst_ip = flow["dst_ip"]
        src_port = flow["src_port"]
        dst_port = flow["dst_port"]
        proto = flow["proto"]

        for rel_t, direction, ip_len, payload_len, tcp_flags in flow["packets"]:
            pkt_ts = _BASE_TS + rel_t
            s_ip = src_ip if direction == "fwd" else dst_ip
            d_ip = dst_ip if direction == "fwd" else src_ip
            s_port = src_port if direction == "fwd" else dst_port
            d_port = dst_port if direction == "fwd" else src_port

            ip_hdr = IP(src=s_ip, dst=d_ip, proto=proto)
            payload = Raw(b"\x00" * payload_len) if payload_len > 0 else b""

            if proto == 6:
                transport = TCP(sport=s_port, dport=d_port, flags=tcp_flags)
            elif proto == 17:
                transport = UDP(sport=s_port, dport=d_port)
            elif proto == 1:
                icmp_type = 8 if direction == "fwd" else 0
                transport = ICMP(type=icmp_type)
            else:
                continue

            pkt = ip_hdr / transport / payload
            pkt.time = pkt_ts
            packets.append(pkt)

    # Sort packets chronologically
    packets.sort(key=lambda p: float(p.time))
    wrpcap(str(dest_path), packets)
    return dest_path
