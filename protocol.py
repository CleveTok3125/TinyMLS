"""Wire protocol for the local spellchecker socket.

Framing: 4-byte big-endian payload length, then a msgpack-encoded map.
A length prefix rather than newline delimiting, because a correction can contain
any character and the payload is binary anyway.

One request per frame, one response per frame, in order. The connection stays
open so the caller keeps a warm model.
"""

import struct
from collections.abc import Callable

import msgpack

MAX_FRAME_BYTES = 1 << 20
HEADER_BYTES = 4
_HEADER = struct.Struct(">I")


class ProtocolError(Exception):
    """Malformed frame. Fatal: the stream can no longer be trusted."""


class RequestError(Exception):
    """A well-formed frame asking for something the server cannot do.

    Recoverable: the caller gets an error response and the connection stays open,
    so a caller does not have to reconnect after one bad request.
    """


def read_frame(recv: Callable[[int], bytes]) -> bytes | None:
    """Read one frame. Returns None on a clean end of stream."""
    header = _recv_exactly(recv, HEADER_BYTES)
    if header is None:
        return None

    (length,) = _HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ProtocolError(
            f"Frame dài {length} byte, vượt giới hạn {MAX_FRAME_BYTES}."
        )
    if length == 0:
        return b""

    body = _recv_exactly(recv, length)
    if body is None:
        raise ProtocolError("Kết thúc giữa chừng payload.")
    return body


def write_frame(sendall: Callable[[bytes], None], payload: bytes) -> None:
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError(
            f"Frame dài {len(payload)} byte, vượt giới hạn {MAX_FRAME_BYTES}."
        )
    sendall(_HEADER.pack(len(payload)) + payload)


def encode_request(message: dict) -> bytes:
    return msgpack.packb(message, use_bin_type=True)


def decode_request(payload: bytes) -> dict:
    if not payload:
        raise ProtocolError("Frame rỗng.")
    try:
        message = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    except Exception as exc:
        raise ProtocolError(f"Không giải mã được msgpack: {exc}") from exc
    if not isinstance(message, dict):
        raise ProtocolError(f"Request phải là map, nhận {type(message).__name__}.")
    return message


def encode_response(message: dict) -> bytes:
    return msgpack.packb(message, use_bin_type=True)


def decode_response(payload: bytes) -> dict:
    try:
        message = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    except Exception as exc:
        raise ProtocolError(f"Không giải mã được msgpack: {exc}") from exc
    if not isinstance(message, dict):
        raise ProtocolError(f"Response phải là map, nhận {type(message).__name__}.")
    return message


def _recv_exactly(recv: Callable[[int], bytes], count: int) -> bytes | None:
    """Read exactly count bytes, or None if the peer closed before any byte."""
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = recv(remaining)
        if not chunk:
            if not chunks:
                return None
            raise ProtocolError("Kết thúc giữa chừng.")
        chunks.append(chunk)
        remaining -= len(chunk)
    joined = b"".join(chunks)
    assert isinstance(joined, bytes)
    return joined
