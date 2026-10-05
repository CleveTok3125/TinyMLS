"""CLI tests: argument handling, output shape, exit codes."""

import json

import pytest

from cli import (
    EXIT_CONNECT_ERROR,
    EXIT_OK,
    EXIT_SERVER_ERROR,
    build_parser,
    main,
)


def runtime_dir(tmp_path) -> str:
    """An existing directory standing in for XDG_RUNTIME_DIR.

    paths.runtime_dir() rejects a path that is not an existing directory, so the
    tests need a real one and must not borrow the developer's /run/user/$UID.
    """
    target = tmp_path / "run"
    target.mkdir()
    return str(target)


def run(capsys, argv) -> tuple[int, str]:
    code = main(argv)
    return code, capsys.readouterr().out


# --- parsing ---------------------------------------------------------------


def test_global_flags_accepted_before_and_after_subcommand():
    parser = build_parser()
    before = parser.parse_args(["--json", "check", "abc"])
    after = parser.parse_args(["check", "abc", "--json"])
    assert before.json is after.json is True
    assert before.command == after.command == "check"


def test_socket_comes_from_environment(monkeypatch):
    monkeypatch.setenv("TINYMLS_SOCKET", "/run/user/example/from-env.sock")
    assert build_parser().parse_args(["ping"]).socket == "/run/user/example/from-env.sock"


def test_socket_defaults_to_xdg_runtime_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("TINYMLS_SOCKET", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", runtime_dir(tmp_path))
    assert main(["ping"]) == EXIT_CONNECT_ERROR


def test_socket_flag_wins_over_xdg_runtime_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("TINYMLS_SOCKET", raising=False)
    override = str(tmp_path / "custom.sock")
    assert build_parser().parse_args(["--socket", override, "ping"]).socket == override


def test_missing_xdg_runtime_dir_is_reported(monkeypatch, capsys):
    monkeypatch.delenv("TINYMLS_SOCKET", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    with pytest.raises(SystemExit):
        main(["ping"])
    assert "XDG_RUNTIME_DIR chưa được đặt" in capsys.readouterr().err


def test_top_k_defaults_to_5():
    assert build_parser().parse_args(["check", "abc"]).top_k == 5


def test_learn_context_is_positional():
    args = build_parser().parse_args(["learn", "da uong thuoc"])
    assert args.context == "da uong thuoc"


@pytest.mark.parametrize(
    "argv",
    [
        ["learn"],
        ["check"],
    ],
)
def test_missing_required_argument_is_rejected(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


# --- happy paths -----------------------------------------------------------


def test_ping(server, capsys):
    code, out = run(capsys, ["--socket", server, "ping"])
    assert code == EXIT_OK
    assert out.strip() == "ok"


def test_check_prints_numbered_suggestions(server, capsys):
    code, out = run(capsys, ["--socket", server, "check", "toi dang go tieng viet"])
    assert code == EXIT_OK
    first_line = out.splitlines()[0]
    assert first_line.startswith("1. ")
    assert len(first_line) > len("1. ")


def test_check_respects_top_k(server, capsys):
    code, out = run(
        capsys, ["--socket", server, "check", "toi dang go tieng viet", "--top-k", "2"]
    )
    assert code == EXIT_OK
    numbered = [l for l in out.splitlines() if l[:2] in {"1.", "2.", "3."}]
    assert len(numbered) == 2


def test_json_output_is_parsable(server, capsys):
    code, out = run(capsys, ["--socket", server, "--json", "check", "toi dang go"])
    assert code == EXIT_OK
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["op"] == "check"
    assert isinstance(payload["suggestions"], list)


def test_learn_text_profile_and_clear(server, capsys):
    base = ["--socket", server]
    code, out = run(
        capsys, base + ["learn-text", "Bệnh nhân uống thuốc kháng sinh"]
    )
    assert code == EXIT_OK
    assert "Đã học" in out

    code, out = run(capsys, base + ["profile"])
    assert code == EXIT_OK
    assert "learned_words" in out

    code, out = run(capsys, base + ["clear"])
    assert code == EXIT_OK

    code, out = run(capsys, base + ["--json", "profile"])
    assert json.loads(out)["learned_words"] == 0


def test_learn_records_selection(server, capsys):
    base = ["--socket", server]
    code, out = run(capsys, base + ["learn", "da uong thuoc"])
    assert code == EXIT_OK
    code, out = run(capsys, base + ["--json", "profile"])
    assert json.loads(out)["memory_size"] > 0
    run(capsys, base + ["clear"])


# --- exit codes ------------------------------------------------------------


def test_server_error_returns_exit_1(server, capsys):
    code, out = run(capsys, ["--socket", server, "check", ""])
    assert code == EXIT_SERVER_ERROR
    assert out == ""


def test_missing_socket_returns_exit_2(capsys):
    code, out = run(capsys, ["--socket", "/run/user/example/cli-absent.sock", "ping"])
    assert code == EXIT_CONNECT_ERROR
    assert out == ""


def test_error_message_goes_to_stderr(server, capsys):
    code = main(["--socket", server, "check", ""])
    assert code == EXIT_SERVER_ERROR
    assert "Lỗi từ server" in capsys.readouterr().err
