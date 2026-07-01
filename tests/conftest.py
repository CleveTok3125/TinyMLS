import contextlib
import io
import os
import shutil
import sys

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
def app(model_dir):
    from api import create_app
    return create_app(model_path=model_dir)


@pytest.fixture
def client(app):
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="session", autouse=True)
def auto_cleanup():
    yield
    shutil.rmtree("data/personalization", ignore_errors=True)
