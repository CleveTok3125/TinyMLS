"""Command line client for the spellchecker socket.

Usage:
    python cli.py check "toi dang go tieng viet"
    python cli.py check "uong thuoc" --personalized --top-k 3
    python cli.py learn-text "$(cat tai-lieu.txt)"
    python cli.py learn "da uong thuoc"
    python cli.py profile
    python cli.py clear
    python cli.py ping

Add --json to any command to print the raw server response.
"""

import argparse
import json
import os
import sys

from client import SpellCheckerClient, SpellCheckerError
from paths import RuntimeDirUnavailable
from paths import socket_path as default_socket_path
from protocol import ProtocolError

EXIT_OK = 0
EXIT_SERVER_ERROR = 1
EXIT_CONNECT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--socket",
        default=os.getenv("TINYMLS_SOCKET"),
        help="Đường dẫn Unix socket (mặc định: $XDG_RUNTIME_DIR/tinymls.sock)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="In ra phản hồi nguyên bản từ server dạng JSON",
    )

    # Repeated on every subcommand with SUPPRESS defaults so that both
    # "cli.py --json check ..." and "cli.py check ... --json" work.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--socket", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    common.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "ping", parents=[common], help="Kiểm tra server còn sống không"
    )

    check = sub.add_parser(
        "check", parents=[common], help="Kiểm tra chính tả một đoạn văn bản"
    )
    check.add_argument("text", help="Văn bản cần kiểm tra")
    check.add_argument("--top-k", type=int, default=5, help="Số gợi ý (mặc định: 5)")
    check.add_argument(
        "--personalized", action="store_true",
        help="Dùng từ và thói quen đã học",
    )

    learn_text = sub.add_parser(
        "learn-text", parents=[common], help="Học từ và context trong văn bản"
    )
    learn_text.add_argument(
        "text", help='Văn bản nguồn; dùng "$(cat file.txt)" cho văn bản dài'
    )

    learn = sub.add_parser(
        "learn", parents=[common], help="Ghi nhận lựa chọn của người dùng"
    )
    learn.add_argument(
        "context", help="Chuỗi từ của gợi ý được chọn, cách nhau bằng dấu cách"
    )

    sub.add_parser(
        "profile", parents=[common], help="Xem dữ liệu cá nhân hóa hiện tại"
    )

    sub.add_parser(
        "clear", parents=[common], help="Xoá toàn bộ dữ liệu cá nhân hóa"
    )

    return parser


def _emit(args: argparse.Namespace, payload: dict, lines: list[str]) -> None:
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for line in lines:
        print(line)


def _run_check(args: argparse.Namespace, conn: SpellCheckerClient) -> tuple[dict, list[str]]:
    payload = conn.check(
        args.text, top_k=args.top_k, personalized=args.personalized
    )
    suggestions = payload["suggestions"]
    if not suggestions:
        return payload, ["Không có gợi ý."]
    lines = [f"{rank}. {text}" for rank, text in enumerate(suggestions, start=1)]
    lines.append(
        f"-- {payload['processing_ms']} ms"
        + (", đã cá nhân hóa" if payload["personalized"] else "")
    )
    return payload, lines


def _run_learn_text(
    args: argparse.Namespace, conn: SpellCheckerClient
) -> tuple[dict, list[str]]:
    payload = conn.learn_text(args.text)
    return payload, [
        f"Đã học {payload['words_added']} từ mới, {payload['contexts_added']} context.",
    ]


def _run_learn(args: argparse.Namespace, conn: SpellCheckerClient) -> tuple[dict, list[str]]:
    context = args.context.split()
    payload = conn.learn(context)
    return payload, ["Đã ghi nhận lựa chọn."]


def _run_profile(
    args: argparse.Namespace, conn: SpellCheckerClient
) -> tuple[dict, list[str]]:
    payload = conn.profile()
    return payload, [
        f"learned_words     : {payload['learned_words']}",
        f"learned_contexts  : {payload['learned_contexts']}",
        f"memory_size       : {payload['memory_size']}",
    ]


def _run_clear(
    args: argparse.Namespace, conn: SpellCheckerClient
) -> tuple[dict, list[str]]:
    payload = conn.clear()
    return payload, ["Đã xoá dữ liệu cá nhân hóa."]



_HANDLERS = {
    "ping": lambda _args, conn: (conn.ping(), ["ok"]),
    "check": _run_check,
    "learn-text": _run_learn_text,
    "learn": _run_learn,
    "profile": _run_profile,
    "clear": _run_clear,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    socket_path = args.socket
    if socket_path is None:
        try:
            socket_path = default_socket_path()
        except RuntimeDirUnavailable as exc:
            parser = build_parser()
            parser.error(str(exc))

    try:
        with SpellCheckerClient(socket_path) as conn:
            payload, lines = _HANDLERS[args.command](args, conn)
    except SpellCheckerError as exc:
        print(f"Lỗi từ server: {exc}", file=sys.stderr)
        return EXIT_SERVER_ERROR
    except ProtocolError as exc:
        print(f"Phản hồi không hợp lệ: {exc}", file=sys.stderr)
        return EXIT_SERVER_ERROR
    except (ConnectionRefusedError, FileNotFoundError) as exc:
        print(f"Không kết nối được socket {socket_path}: {exc}", file=sys.stderr)
        return EXIT_CONNECT_ERROR
    except OSError as exc:
        print(f"Lỗi socket: {exc}", file=sys.stderr)
        return EXIT_CONNECT_ERROR

    _emit(args, payload, lines)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
