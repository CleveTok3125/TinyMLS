import contextlib
import io
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


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
)


@pytest.fixture(scope="session")
def socket_path(tmp_path_factory):
    return str(tmp_path_factory.mktemp("sock") / "tinymls.sock")


@pytest.fixture(scope="session")
def service(model_dir, tmp_path_factory):
    """Server service with isolated model, data dir and config.

    The config path is pinned so results do not shift with whatever
    config.json the developer happens to have in XDG_CONFIG_HOME.
    """
    from service import SpellCheckerService

    data_root = tmp_path_factory.mktemp("xdg-data")
    svc = SpellCheckerService(
        model_path=model_dir,
        data_dir=str(data_root),
        config_path=str(CONFIG_PATH),
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
