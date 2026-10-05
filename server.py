"""Unix domain socket server for the spellchecker.

One connection per caller, kept open so the model stays warm. Connections are
handled on their own thread so a caller may hold an idle connection open, but
the checker calls themselves are serialised: correct_sentence is CPU-bound and
bound by the GIL, so running two at once only adds scheduling jitter.
"""

import os
import signal
import socket
import stat
import threading
from types import FrameType

from protocol import (
    ProtocolError,
    RequestError,
    decode_request,
    encode_response,
    read_frame,
    write_frame,
)
from service import SpellCheckerService

OP_CHECK = "check"
OP_LEARN = "learn"
OP_LEARN_TEXT = "learn_text"
OP_PROFILE = "profile"
OP_CLEAR = "clear"
OP_PING = "ping"

_LISTEN_BACKLOG = 64


def _remove_stale_socket(path: str) -> None:
    """Delete a leftover socket from a previous run; refuse anything else.

    A mistyped --socket pointing at a regular file must not delete that file, so
    the type is checked rather than assumed.
    """
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise FileExistsError(
            f"'{path}' đã tồn tại và không phải socket. "
            "Chỉ định một đường dẫn socket khác."
        )
    os.unlink(path)
_RECV_CHUNK = 65536


class SpellCheckerServer:
    def __init__(
        self,
        socket_path: str,
        service: SpellCheckerService | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._service = service if service is not None else SpellCheckerService()
        self._call_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._server: socket.socket | None = None
        self._workers: list[threading.Thread] = []

    @property
    def socket_path(self) -> str:
        return self._socket_path

    def serve_forever(self) -> None:
        server = self._bind()
        self._server = server
        try:
            while not self._shutdown.is_set():
                try:
                    conn, _ = server.accept()
                except OSError:
                    if self._shutdown.is_set():
                        break
                    raise
                worker = threading.Thread(
                    target=self._serve_connection, args=(conn,), daemon=True
                )
                worker.start()
                self._workers.append(worker)
                self._workers = [w for w in self._workers if w.is_alive()]
        finally:
            self._close()

    def shutdown(self) -> None:
        self._shutdown.set()
        if self._server is not None:
            try:
                self._server.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _bind(self) -> socket.socket:
        directory = os.path.dirname(os.path.abspath(self._socket_path))
        os.makedirs(directory, exist_ok=True)
        _remove_stale_socket(self._socket_path)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self._socket_path)
        server.listen(_LISTEN_BACKLOG)
        os.chmod(self._socket_path, 0o600)
        return server

    def _close(self) -> None:
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        for worker in self._workers:
            worker.join(timeout=1.0)
        self._workers.clear()
        try:
            _remove_stale_socket(self._socket_path)
        except (OSError, FileExistsError):
            pass

    def _serve_connection(self, conn: socket.socket) -> None:
        with conn:
            recv = conn.recv
            sendall = conn.sendall
            while not self._shutdown.is_set():
                try:
                    payload = read_frame(recv)
                except ProtocolError as exc:
                    self._send_error(sendall, str(exc))
                    return
                except OSError:
                    return
                if payload is None:
                    return
                try:
                    response = self._dispatch(decode_request(payload))
                except ProtocolError as exc:
                    # Framing is broken, so the stream cannot be resynchronised.
                    self._send_error(sendall, str(exc))
                    return
                except RequestError as exc:
                    self._send_error(sendall, str(exc))
                    continue
                except Exception as exc:  # noqa: BLE001
                    self._send_error(sendall, f"{type(exc).__name__}: {exc}")
                    continue
                try:
                    write_frame(sendall, encode_response(response))
                except ProtocolError as exc:
                    self._send_error(sendall, str(exc))
                    return
                except OSError:
                    return

    def _dispatch(self, request: dict) -> dict:
        op = request.get("op")
        if not isinstance(op, str) or not op:
            raise RequestError("Thiếu trường 'op'.")

        # Serialised: the checker is a shared, mutable cache.
        with self._call_lock:
            if op == OP_CHECK:
                result = self._service.check(
                    text=request.get("text") or "",
                    top_k=request.get("top_k") or 5,
                    personalized=bool(request.get("personalized")),
                )
            elif op == OP_LEARN:
                result = self._service.learn(
                    context=request.get("context") or [],
                )
            elif op == OP_LEARN_TEXT:
                result = self._service.learn_text(text=request.get("text") or "")
            elif op == OP_PROFILE:
                result = self._service.profile()
            elif op == OP_CLEAR:
                result = self._service.clear()
            elif op == OP_PING:
                result = {"status": "ok"}
            else:
                raise RequestError(f"Operation không hỗ trợ: '{op}'.")

        return {"ok": True, "op": op, **result}

    @staticmethod
    def _send_error(sendall, message: str) -> None:
        try:
            write_frame(sendall, encode_response({"ok": False, "error": message}))
        except OSError:
            pass


def serve(socket_path: str, model_path: str | None = None) -> None:
    """Run until SIGINT or SIGTERM."""
    service = SpellCheckerService(model_path=model_path)
    service.preload()
    server = SpellCheckerServer(socket_path, service=service)

    def _on_signal(signum: int, _frame: FrameType | None) -> None:
        server.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    print(f"Spellchecker sẵn sàng trên {socket_path}")
    server.serve_forever()
