"""Decoders for Adamo teleop control messages.

The Adamo web client publishes ROS messages CDR-encoded, wrapped in a small
length-prefixed envelope. These are little-endian CDR readers matching the
adamo-network wire format (see adamo-ts/tools/listen.py for the reference).
"""

from __future__ import annotations

import struct
from typing import Optional

import numpy as np


def decode_envelope(data: bytes):
    """Strip the envelope: [4B BE topic_len][topic][4B BE type_len][type][CDR]."""
    off = 0
    (tlen,) = struct.unpack_from(">I", data, off); off += 4
    topic = data[off:off + tlen].decode("utf-8"); off += tlen
    (nlen,) = struct.unpack_from(">I", data, off); off += 4
    type_name = data[off:off + nlen].decode("utf-8"); off += nlen
    return topic, type_name, data[off:]


class CdrReader:
    """Minimal little-endian CDR reader (alignment is relative to the payload start)."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.off = 4            # skip the 4-byte CDR header
        self.payload_start = 4

    def align(self, n: int) -> None:
        rel = self.off - self.payload_start
        self.off += (n - (rel % n)) % n

    def i32(self) -> int:
        self.align(4); (v,) = struct.unpack_from("<i", self.data, self.off); self.off += 4; return v

    def u32(self) -> int:
        self.align(4); (v,) = struct.unpack_from("<I", self.data, self.off); self.off += 4; return v

    def f32(self) -> float:
        self.align(4); (v,) = struct.unpack_from("<f", self.data, self.off); self.off += 4; return v

    def f64(self) -> float:
        self.align(8); (v,) = struct.unpack_from("<d", self.data, self.off); self.off += 8; return v

    def u8(self) -> int:
        (v,) = struct.unpack_from("<B", self.data, self.off); self.off += 1; return v

    def string(self) -> str:
        n = self.u32()  # length includes the null terminator
        if n == 0:
            return ""
        s = self.data[self.off:self.off + n - 1].decode("utf-8")
        self.off += n
        return s


def _skip_header(r: CdrReader) -> None:
    r.i32(); r.u32(); r.string()  # std_msgs/Header: stamp sec, nsec, frame_id


def decode_pose_stamped(cdr: bytes):
    """geometry_msgs/PoseStamped -> (position(3,), quaternion xyzw(4,)) or None."""
    try:
        r = CdrReader(cdr); _skip_header(r)
        pos = np.array([r.f64(), r.f64(), r.f64()], np.float64)
        quat = np.array([r.f64(), r.f64(), r.f64(), r.f64()], np.float64)  # x, y, z, w
        return pos, quat
    except Exception:  # noqa: BLE001
        return None


def decode_pose_array(cdr: bytes) -> Optional[np.ndarray]:
    """geometry_msgs/PoseArray -> (N, 7) [x,y,z, qx,qy,qz,qw] or None (e.g. hand joints)."""
    try:
        r = CdrReader(cdr); _skip_header(r)
        n = r.u32()
        out = np.empty((n, 7), np.float64)
        for i in range(n):
            out[i] = [r.f64(), r.f64(), r.f64(), r.f64(), r.f64(), r.f64(), r.f64()]
        return out
    except Exception:  # noqa: BLE001
        return None


def decode_joy(cdr: bytes):
    """sensor_msgs/Joy -> (axes, buttons) or None."""
    try:
        r = CdrReader(cdr); _skip_header(r)
        axes = [r.f32() for _ in range(r.u32())]
        buttons = [r.i32() for _ in range(r.u32())]
        return axes, buttons
    except Exception:  # noqa: BLE001
        return None


def decode_joint_state(cdr: bytes):
    """sensor_msgs/JointState -> {names, positions, velocities, efforts} or None."""
    try:
        r = CdrReader(cdr); _skip_header(r)
        names = [r.string() for _ in range(r.u32())]
        positions = [r.f64() for _ in range(r.u32())]
        velocities = [r.f64() for _ in range(r.u32())]
        efforts = [r.f64() for _ in range(r.u32())]
        return {"names": names, "positions": positions,
                "velocities": velocities, "efforts": efforts}
    except Exception:  # noqa: BLE001
        return None


def decode_bool(cdr: bytes):
    """std_msgs/Bool -> bool or None."""
    try:
        return CdrReader(cdr).u8() != 0
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Encoders (mirror of the readers above) -- used by the publish side (sender).
# ---------------------------------------------------------------------------


class CdrWriter:
    """Minimal little-endian CDR writer (the inverse of CdrReader)."""

    def __init__(self) -> None:
        self.buf = bytearray(b"\x00\x01\x00\x00")  # LE CDR header
        self.payload_start = 4

    def align(self, n: int) -> None:
        rel = len(self.buf) - self.payload_start
        self.buf += b"\x00" * ((n - (rel % n)) % n)

    def i32(self, v: int) -> None:
        self.align(4); self.buf += struct.pack("<i", int(v))

    def u32(self, v: int) -> None:
        self.align(4); self.buf += struct.pack("<I", int(v))

    def f32(self, v: float) -> None:
        self.align(4); self.buf += struct.pack("<f", float(v))

    def f64(self, v: float) -> None:
        self.align(8); self.buf += struct.pack("<d", float(v))

    def u8(self, v: int) -> None:
        self.buf += struct.pack("<B", int(v))

    def string(self, s: str) -> None:
        b = s.encode("utf-8") + b"\x00"
        self.u32(len(b)); self.buf += b

    def bytes(self) -> bytes:
        return bytes(self.buf)


def encode_envelope(topic: str, type_name: str, cdr: bytes) -> bytes:
    """[4B BE topic_len][topic][4B BE type_len][type][CDR]."""
    return (struct.pack(">I", len(topic)) + topic.encode("utf-8") +
            struct.pack(">I", len(type_name)) + type_name.encode("utf-8") + cdr)


def _write_header(w: CdrWriter, frame_id: str = "", stamp=(0, 0)) -> None:
    w.i32(stamp[0]); w.u32(stamp[1]); w.string(frame_id)


def encode_pose_stamped(position, quat_xyzw, frame_id: str = "") -> bytes:
    w = CdrWriter(); _write_header(w, frame_id)
    for v in position:
        w.f64(v)
    for v in quat_xyzw:
        w.f64(v)
    return w.bytes()


def encode_pose_array(poses, frame_id: str = "") -> bytes:
    """poses: (N, 7) [x,y,z, qx,qy,qz,qw]."""
    arr = np.asarray(poses, np.float64).reshape(-1, 7)
    w = CdrWriter(); _write_header(w, frame_id)
    w.u32(len(arr))
    for row in arr:
        for v in row:
            w.f64(v)
    return w.bytes()


def encode_joy(axes, buttons, frame_id: str = "") -> bytes:
    w = CdrWriter(); _write_header(w, frame_id)
    w.u32(len(axes))
    for a in axes:
        w.f32(a)
    w.u32(len(buttons))
    for b in buttons:
        w.i32(b)
    return w.bytes()


def encode_joint_state(names, positions, velocities=(), efforts=(), frame_id: str = "") -> bytes:
    w = CdrWriter(); _write_header(w, frame_id)
    w.u32(len(names))
    for n in names:
        w.string(n)
    for seq in (positions, velocities, efforts):
        w.u32(len(seq))
        for v in seq:
            w.f64(v)
    return w.bytes()


def encode_bool(value: bool) -> bytes:
    w = CdrWriter(); w.u8(1 if value else 0)
    return w.bytes()
