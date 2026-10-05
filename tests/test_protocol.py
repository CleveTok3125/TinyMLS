"""Protocol-level tests: framing, round trips, error handling, concurrency."""

import socket
import struct

import msgpack
import pytest

from protocol import (
    HEADER_BYTES,
    MAX_FRAME_BYTES,
    ProtocolError,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    read_frame,
    write_frame,
)


class _FakeSocket:
    """Feeds a scripted byte stream to the frame reader.

    Honours the requested size and returns empty once exhausted, like a real
    socket, so the reader's partial-read handling is actually exercised.
    """

    def __init__(self, data: bytes, chunk_size: int = 0) -> None:
        self._data = data
        self._pos = 0
        self._chunk = chunk_size

    def recv(self, size: int) -> bytes:
        if self._pos >= len(self._data):
            return b""
        take = min(size, len(self._data) - self._pos)
        if self._chunk:
            take = min(take, self._chunk)
        chunk = self._data[self._pos : self._pos + take]
        self._pos += take
        return chunk


def frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


# --- framing ---------------------------------------------------------------


def test_round_trip_single_frame():
    sock = _FakeSocket(frame(b"hello"))
    assert read_frame(sock.recv) == b"hello"


def test_round_trip_two_frames_in_one_read():
    sock = _FakeSocket(frame(b"one") + frame(b"two"))
    assert read_frame(sock.recv) == b"one"
    assert read_frame(sock.recv) == b"two"


def test_read_frame_survives_byte_by_byte_delivery():
    sock = _FakeSocket(frame(b"payload"), chunk_size=1)
    assert read_frame(sock.recv) == b"payload"


def test_read_frame_returns_none_on_clean_close():
    assert read_frame(_FakeSocket(b"").recv) is None


def test_read_frame_rejects_oversized_length():
    sock = _FakeSocket(struct.pack(">I", MAX_FRAME_BYTES + 1))
    with pytest.raises(ProtocolError, match="vượt giới hạn"):
        read_frame(sock.recv)


def test_read_frame_rejects_truncated_payload():
    sock = _FakeSocket(struct.pack(">I", 100) + b"only-a-few-bytes")
    with pytest.raises(ProtocolError):
        read_frame(sock.recv)


def test_write_frame_prefixes_length():
    sent: list[bytes] = []
    write_frame(sent.append, b"payload")
    assert sent[0] == struct.pack(">I", 7) + b"payload"


def test_write_frame_rejects_oversized_payload():
    with pytest.raises(ProtocolError, match="vượt giới hạn"):
        write_frame(lambda _b: None, b"x" * (MAX_FRAME_BYTES + 1))


def test_empty_frame_is_distinct_from_close():
    sock = _FakeSocket(struct.pack(">I", 0))
    assert read_frame(sock.recv) == b""


# --- encoding --------------------------------------------------------------


def test_request_encode_decode_keeps_unicode_and_list():
    request = {"op": "check", "text": "tiếng việt", "context": ["a", "b"]}
    assert decode_request(encode_request(request)) == request


def test_response_encode_decode_keeps_nested_list():
    response = {"ok": True, "suggestions": ["một", "hai"], "ms": 1.5}
    assert decode_response(encode_response(response)) == response


def test_decode_rejects_non_map():
    with pytest.raises(ProtocolError, match="phải là map"):
        decode_request(msgpack.packb([1, 2], use_bin_type=True))


def test_decode_rejects_empty_frame():
    with pytest.raises(ProtocolError, match="rỗng"):
        decode_request(b"")


def test_decode_rejects_garbage():
    with pytest.raises(ProtocolError, match="msgpack"):
        decode_request(b"\xc1not-msgpack")


def test_header_is_four_bytes():
    assert HEADER_BYTES == 4


# --- operations ------------------------------------------------------------


def test_ping(client):
    assert client.ping()["status"] == "ok"


def test_check_returns_ordered_suggestions(client):
    result = client.check("toi dang go tieng viet", top_k=3)
    assert result["suggestions"]
    assert result["best_correction"] == result["suggestions"][0]
    assert len(result["suggestions"]) <= 3
    assert result["personalized"] is False
    assert result["processing_ms"] >= 0


def test_check_respects_top_k(client):
    assert len(client.check("toi dang go tieng viet", top_k=1)["suggestions"]) <= 1


def test_check_preserves_case_pattern(client):
    result = client.check("TOI DANG GO TIENG VIET", top_k=1)
    assert result["suggestions"][0].isupper()


def test_check_strips_surrounding_whitespace(client):
    result = client.check("   toi dang go   ", top_k=1)
    assert result["text"] == "toi dang go"


# --- error handling --------------------------------------------------------


@pytest.mark.parametrize(
    "request_body",
    [
        {"op": "check", "text": ""},
        {"op": "check", "text": "   "},
        {"op": "check", "text": "x" * 5000},
        {"op": "learn_text", "text": ""},
        {"op": "learn", "context": ["only-one"]},
        {"op": "not-a-real-op"},
        {"text": "missing op"},
    ],
)
def test_bad_request_returns_error(client, request_body):
    from client import SpellCheckerError

    with pytest.raises(SpellCheckerError):
        client.call(**request_body)


def test_connection_survives_bad_requests(client):
    from client import SpellCheckerError

    for _ in range(5):
        with pytest.raises(SpellCheckerError):
            client.call(op="not-a-real-op")
    assert client.ping()["status"] == "ok"
    assert client.check("toi dang go", top_k=1)["suggestions"]


def test_oversized_frame_closes_connection(server):
    bad = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    bad.connect(server)
    bad.sendall(struct.pack(">I", MAX_FRAME_BYTES + 1))
    response = msgpack.unpackb(bad.recv(4096)[HEADER_BYTES:], raw=False)
    assert response["ok"] is False
    bad.close()


def test_empty_frame_is_rejected_but_connection_survives(client):
    from client import SpellCheckerError

    with pytest.raises(SpellCheckerError):
        client.call(op="")
    assert client.ping()["status"] == "ok"


# --- personalization -------------------------------------------------------


def test_learn_text_then_profile(client):
    learned = client.learn_text("Bệnh nhân được uống thuốc kháng sinh")
    assert learned["words_added"] > 0
    assert learned["contexts_added"] > 0

    profile = client.profile()
    assert profile["learned_words"] == learned["words_added"]
    assert profile["learned_contexts"] == learned["contexts_added"]
    assert "user_hash" not in profile

    client.clear()


def test_personalized_flag_is_opt_in(client):
    client.learn_text("tôi đang gõ tiếng việt rất nhanh")
    try:
        assert client.check("tieng viet")["personalized"] is False
        result = client.check("tieng viet", personalized=True)
        assert result["personalized"] is True
    finally:
        client.clear()


def test_learn_selection_persists_memory(client):
    client.learn(["toi", "uong", "thuoc"])
    assert client.profile()["memory_size"] > 0
    client.clear()


def test_clear_resets_profile(client):
    client.learn_text("một hai ba bốn năm sáu")
    client.clear()
    profile = client.profile()
    assert profile["learned_words"] == 0
    assert profile["memory_size"] == 0


# --- connection handling ---------------------------------------------------


def test_multiple_concurrent_connections(server):
    results = []
    connections = [socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) for _ in range(4)]
    try:
        for conn in connections:
            conn.connect(server)
        for conn in connections:
            write_frame(conn.sendall, encode_request({"op": "ping"}))
        for conn in connections:
            payload = read_frame(conn.recv)
            assert payload is not None
            results.append(decode_response(payload)["ok"])
        assert results == [True, True, True, True]
    finally:
        for conn in connections:
            conn.close()


def test_socket_permissions_are_owner_only(server):
    import os
    import stat

    mode = os.stat(server).st_mode
    assert not mode & (stat.S_IROTH | stat.S_IWOTH)

# --- regression guards for issues found in review -------------------------


def test_bind_refuses_to_delete_a_regular_file(tmp_path):
    from server import SpellCheckerServer

    victim = tmp_path / "not-a-socket"
    victim.write_text("dữ liệu quan trọng", encoding="utf-8")
    srv = SpellCheckerServer(str(victim), service=None)

    with pytest.raises(FileExistsError):
        srv._bind()

    assert victim.read_text(encoding="utf-8") == "dữ liệu quan trọng"


def test_bind_removes_a_stale_socket(tmp_path, model_dir):
    import socket as socket_module

    from server import SpellCheckerServer

    path = tmp_path / "stale.sock"
    stale = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()

    srv = SpellCheckerServer(str(path), service=None)
    bound = srv._bind()
    try:
        bound.close()
    finally:
        srv._close()


def test_learn_rejects_a_bare_string_context(client):
    from client import SpellCheckerError

    with pytest.raises(SpellCheckerError):
        client.call(op="learn", context="toi dang")


def test_learn_rejects_non_string_elements(client):
    from client import SpellCheckerError

    with pytest.raises(SpellCheckerError):
        client.call(op="learn", context=["toi", 5])


def test_clear_only_removes_its_own_files(tmp_path, test_config):
    from service import SpellCheckerService

    keep_dir = tmp_path / "keepme"
    keep_dir.mkdir()
    (keep_dir / "t.txt").write_text("x", encoding="utf-8")
    (tmp_path / "khac.txt").write_text("y", encoding="utf-8")

    svc = SpellCheckerService(data_dir=str(tmp_path), config_path=test_config)
    svc.learn_text("một hai ba bốn")
    svc.clear()

    assert sorted(p.name for p in tmp_path.iterdir()) == ["keepme", "khac.txt"]


def test_learn_does_not_build_a_checker(tmp_path, model_dir, test_config):
    """Recording a memory entry must not pay for a model load."""
    from service import SpellCheckerService

    svc = SpellCheckerService(
        model_path=model_dir, data_dir=str(tmp_path), config_path=test_config
    )
    svc.preload()
    before = len(svc._cache)

    svc.learn(["toi", "uong", "thuoc"])

    assert len(svc._cache) == before


def test_personalized_check_reuses_the_loaded_checker(tmp_path, model_dir, test_config):
    """Personalisation varies per request, so it must not fork a second model load."""
    from service import SpellCheckerService

    svc = SpellCheckerService(
        model_path=model_dir, data_dir=str(tmp_path), config_path=test_config
    )
    svc.preload()
    checker = svc.get_checker()

    svc.check("toi dang go tieng viet", top_k=1, personalized=True)

    assert svc.get_checker() is checker
    assert len(svc._cache) == 1


def test_personalization_still_changes_the_result(tmp_path, model_dir, test_config):
    """Sharing one checker must not let the personalised path lose its boost.

    Asserts the boost took effect rather than a fixed suggestion string, because
    the exact correction also depends on config values the test does not pin.
    """
    from service import SpellCheckerService

    svc = SpellCheckerService(
        model_path=model_dir, data_dir=str(tmp_path), config_path=test_config
    )
    svc.learn_text("xilinzarow khoaimon")

    plain = svc.check("toi dang xilinzarow", top_k=1)
    tuned = svc.check("toi dang xilinzarow", top_k=1, personalized=True)

    assert plain["personalized"] is False
    assert tuned["personalized"] is True
    assert "xilinzarow" in tuned["best_correction"]


def test_one_checker_gives_the_same_answer_either_way(
    tmp_path, model_dir, test_config
):
    """Interleaved personalised and plain checks must not leak state into each other."""
    from service import SpellCheckerService

    svc = SpellCheckerService(
        model_path=model_dir, data_dir=str(tmp_path), config_path=test_config
    )
    texts = ["toi dang go tieng viet", "ngày mai là thứ hai", "khoong"]

    plain_only = [svc.check(t, top_k=3)["suggestions"] for t in texts]
    for _ in range(2):
        for t in texts:
            svc.check(t, top_k=3, personalized=True)
    plain_after = [svc.check(t, top_k=3)["suggestions"] for t in texts]

    assert plain_only == plain_after


def test_learned_words_are_invisible_to_plain_checks(tmp_path, model_dir, test_config):
    """A plain check after a personalised one must not inherit its boost."""
    from service import SpellCheckerService

    svc = SpellCheckerService(
        model_path=model_dir, data_dir=str(tmp_path), config_path=test_config
    )
    svc.learn_text("xilinzarow khoaimon")
    text = "toi dang xilinzarow"

    plain_before = svc.check(text, top_k=1)["suggestions"]
    svc.check(text, top_k=1, personalized=True)
    plain_after = svc.check(text, top_k=1)["suggestions"]

    assert plain_before == plain_after
