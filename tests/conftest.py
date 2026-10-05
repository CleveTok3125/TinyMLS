import contextlib
import io
import json
import os
import socket
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CORPUS = """\
tôi là người dùng
tôi là người mới
bệnh nhân uống thuốc bổ
gõ tiếng việt hàng ngày
ăn cơm ba bữa mỗi ngày
ngày mai là thứ hai
năm tháng năm qua nhanh
một người dùng mới
thuốc bổ cho bệnh nhân
đi chơi về học
học mỗi ngày
"""


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    (d / "test.txt").write_text(CORPUS, encoding="utf-8")
    return str(d)


@pytest.fixture(scope="session")
def model_dir(corpus_dir, tmp_path_factory):
    from builder import build_language_stats_from_folder

    output_dir = tmp_path_factory.mktemp("model")
    with contextlib.redirect_stdout(io.StringIO()):
        build_language_stats_from_folder(
            folder_path=corpus_dir,
            output_dir=str(output_dir),
            num_workers=1,
        )
    return str(output_dir)


@pytest.fixture(scope="session")
def test_config(tmp_path_factory):
    """A config file owned by the suite.

    config.json is gitignored, so a fresh checkout falls back to the dataclass
    defaults. Pinning the values keeps assertions independent of whether the
    working directory happens to carry a config file.
    """
    path = tmp_path_factory.mktemp("config") / "config.json"
    path.write_text(
        json.dumps(
            {
                "top_n": 25,
                "cutoff": 0.1,
                "sim_weight": 1,
                "context_weight": 1.0,
                "beam_width": 5,
                "lambda_3": 0.6,
                "lambda_2": 0.3,
                "lambda_1": 0.1,
            }
        ),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture(scope="session")
def socket_path(tmp_path_factory):
    return str(tmp_path_factory.mktemp("sock") / "tinymls.sock")


@pytest.fixture(scope="session")
def service(model_dir, tmp_path_factory, test_config):
    """Server service with an isolated model, data dir and config."""
    from service import SpellCheckerService

    data_root = tmp_path_factory.mktemp("xdg-data")
    svc = SpellCheckerService(
        model_path=model_dir,
        data_dir=str(data_root),
        config_path=test_config,
    )
    svc.preload()
    return svc


@pytest.fixture(scope="session")
def server(service, socket_path):
    from server import SpellCheckerServer

    srv = SpellCheckerServer(socket_path, service=service)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()

    deadline = 10.0
    step = 0.05
    while deadline > 0:
        if os.path.exists(socket_path):
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(socket_path)
                listening = True
            except OSError:
                listening = False
            finally:
                probe.close()
            if listening:
                break
        threading.Event().wait(step)
        deadline -= step
    else:
        pytest.fail("Server không lắng nghe trên socket")

    yield socket_path
    srv.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def client(server):
    from client import SpellCheckerClient

    with SpellCheckerClient(server, timeout=30) as conn:
        yield conn
