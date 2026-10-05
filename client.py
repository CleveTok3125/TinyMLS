"""Blocking client for the spellchecker socket.

Reference implementation for the wire protocol, and what the tests use. A native
caller in another language follows protocol.py: 4-byte big-endian length then a
msgpack map.
"""

import socket
from typing import Any

from protocol import decode_response, encode_request, read_frame, write_frame


class SpellCheckerError(Exception):
    """Server replied with ok=false, or the connection failed."""


class SpellCheckerClient:
    def __init__(self, socket_path: str, timeout: float | None = None) -> None:
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(timeout)
        try:
            self._socket.connect(socket_path)
        except OSError:
            self._socket.close()
            raise

    def __enter__(self) -> "SpellCheckerClient":  # noqa: PYI034
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._socket.close()
        except OSError:
            pass

    def call(self, **request: Any) -> dict:
        write_frame(self._socket.sendall, encode_request(request))
        payload = read_frame(self._socket.recv)
        if payload is None:
            raise SpellCheckerError("Server đã đóng kết nối.")
        response = decode_response(payload)
        if not response.get("ok"):
            raise SpellCheckerError(str(response.get("error", "Lỗi không xác định.")))
        return response

    def ping(self) -> dict:
        return self.call(op="ping")

    def check(self, text: str, top_k: int = 5, personalized: bool = False) -> dict:
        return self.call(
            op="check", text=text, top_k=top_k, personalized=personalized
        )

    def learn(self, context: list[str]) -> dict:
        return self.call(op="learn", context=context)

    def learn_text(self, text: str) -> dict:
        return self.call(op="learn_text", text=text)

    def profile(self) -> dict:
        return self.call(op="profile")

    def clear(self) -> dict:
        return self.call(op="clear")
