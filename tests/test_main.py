"""Command line tests for main.py: parsing, dispatch, socket resolution."""

import pytest

import main
from service import SERVER_DATA_FOLDER


@pytest.fixture(autouse=True)
def clean_socket_env(monkeypatch):
    """Neither variable may leak in from the machine running the suite."""
    monkeypatch.delenv("TINYMLS_SOCKET", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)


def parse(argv):
    return main.build_parser().parse_args(argv)


def runtime_dir(tmp_path) -> str:
    """An existing directory standing in for XDG_RUNTIME_DIR."""
    target = tmp_path / "run"
    target.mkdir()
    return str(target)


# --- parsing ----------------------------------------------------------------


def test_serve_defaults():
    args = parse(["serve"])
    assert args.command == "serve"
    assert args.model is None
    assert args.socket is None


def test_serve_takes_a_positional_model():
    assert parse(["serve", "model.tinymls"]).model == "model.tinymls"


def test_socket_comes_from_environment(monkeypatch):
    monkeypatch.setenv("TINYMLS_SOCKET", "/run/user/example/from-env.sock")
    assert parse(["serve"]).socket == "/run/user/example/from-env.sock"


def test_build_defaults():
    args = parse(["build"])
    assert args.output_dir is None
    assert args.corpus == SERVER_DATA_FOLDER
    assert args.dict_path is None
    assert args.workers == 1
    assert args.recursive is False


def test_build_takes_every_option(tmp_path):
    args = parse(
        [
            "build", str(tmp_path),
            "--corpus", "corpus",
            "--dict", "wordlist.dic",
            "--workers", "4",
            "--recursive",
        ]
    )
    assert args.output_dir == str(tmp_path)
    assert args.corpus == "corpus"
    assert args.dict_path == "wordlist.dic"
    assert args.workers == 4
    assert args.recursive is True


def test_export_requires_an_output_path():
    with pytest.raises(SystemExit):
        parse(["export"])


def test_export_takes_dict_option():
    args = parse(["export", "model.tinymls", "--dict", "wordlist.dic"])
    assert args.output == "model.tinymls"
    assert args.dict_path == "wordlist.dic"


# --- options belong to one command ----------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["build", "--socket", "/tmp/example.sock"],
        ["serve", "--corpus", "corpus"],
        ["serve", "--dict", "wordlist.dic"],
        ["serve", "--workers", "4"],
        ["serve", "--recursive"],
        ["export", "model.tinymls", "--corpus", "corpus"],
        ["export", "model.tinymls", "--socket", "/tmp/example.sock"],
        ["export", "model.tinymls", "--workers", "4"],
        ["export", "model.tinymls", "--recursive"],
    ],
)
def test_option_is_rejected_on_the_wrong_command(argv):
    with pytest.raises(SystemExit):
        parse(argv)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["model.tinymls"],
        ["--socket", "/tmp/example.sock"],
        ["--build"],
        ["--export", "model.tinymls"],
        ["--corpus", "corpus"],
    ],
)
def test_flat_invocation_is_rejected(argv):
    with pytest.raises(SystemExit):
        parse(argv)


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        parse(["check", "toi dang go tieng viet"])


# --- dispatch ---------------------------------------------------------------


def test_build_forwards_its_options(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(main, "build_stats", lambda **kwargs: seen.update(kwargs))

    assert main.main(["build", str(tmp_path), "--workers", "3", "--recursive"]) == 0

    assert seen == {
        "output_dir": str(tmp_path),
        "dict_path": None,
        "workers": 3,
        "recursive": True,
        "corpus_dir": SERVER_DATA_FOLDER,
    }


def test_export_forwards_its_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        main,
        "export_stats",
        lambda output_path, dict_path=None: seen.update(
            {"output": output_path, "dict": dict_path}
        ),
    )

    assert main.main(["export", "model.tinymls", "--dict", "wordlist.dic"]) == 0

    assert seen == {"output": "model.tinymls", "dict": "wordlist.dic"}


def test_serve_uses_the_xdg_socket_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", runtime_dir(tmp_path))
    seen = {}
    monkeypatch.setattr(
        main,
        "serve",
        lambda socket_path, model_path=None: seen.update(
            {"socket": socket_path, "model": model_path}
        ),
    )

    assert main.main(["serve"]) == 0

    assert seen == {"socket": str(tmp_path / "run" / "tinymls.sock"), "model": None}


def test_serve_socket_option_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", runtime_dir(tmp_path))
    seen = {}
    monkeypatch.setattr(
        main,
        "serve",
        lambda socket_path, model_path=None: seen.update(
            {"socket": socket_path, "model": model_path}
        ),
    )

    assert main.main(["serve", "model.tinymls", "--socket", str(tmp_path / "s.sock")]) == 0

    assert seen == {"socket": str(tmp_path / "s.sock"), "model": "model.tinymls"}


# --- runtime directory is a serve-only concern ----------------------------


@pytest.mark.parametrize(
    ("argv", "entrypoint"),
    [("build", "build_stats"), ("export", "export_stats")],
)
def test_offline_commands_need_no_runtime_dir(monkeypatch, argv, entrypoint):
    monkeypatch.setattr(main, entrypoint, lambda *a, **k: None)
    assert main.main([argv, "model.tinymls"]) == 0


def test_serve_reports_a_missing_runtime_dir(capsys):
    with pytest.raises(SystemExit):
        main.main(["serve"])
    assert "XDG_RUNTIME_DIR chưa được đặt" in capsys.readouterr().err