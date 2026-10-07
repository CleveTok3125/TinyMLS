"""Command line entry point: run the server, build stats, export a model package."""

import argparse
import os

from paths import RuntimeDirUnavailable
from paths import socket_path as default_socket_path
from server import serve
from service import SERVER_DATA_FOLDER, build_stats, export_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py", description="TinyMLS Spell Checker"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser("serve", help="Chạy server trên Unix socket")
    serve_parser.add_argument(
        "model",
        nargs="?",
        metavar="MODEL",
        help="Đường dẫn model (.tinymls hoặc thư mục, mặc định: trained_model/)",
    )
    serve_parser.add_argument(
        "--socket",
        default=os.getenv("TINYMLS_SOCKET"),
        help="Đường dẫn Unix socket (mặc định: $XDG_RUNTIME_DIR/tinymls.sock)",
    )

    build = sub.add_parser(
        "build", help="Xây dựng lại thống kê N-gram từ corpus rồi thoát"
    )
    build.add_argument(
        "output_dir",
        nargs="?",
        metavar="OUTPUT_DIR",
        help="Thư mục ghi model (mặc định: stats_path trong config)",
    )
    build.add_argument(
        "--corpus",
        default=SERVER_DATA_FOLDER,
        help=f"Thư mục corpus (mặc định: {SERVER_DATA_FOLDER})",
    )
    build.add_argument(
        "--dict", dest="dict_path", default=None, help="File từ điển dùng khi build"
    )
    build.add_argument(
        "--workers", type=int, default=1, help="Số tiến trình (mặc định: 1)"
    )
    build.add_argument(
        "--recursive", action="store_true", help="Đọc đệ quy các thư mục con"
    )

    export = sub.add_parser(
        "export", help="Export model thành một file .tinymls rồi thoát"
    )
    export.add_argument("output", metavar="OUTPUT", help="File .tinymls cần ghi")
    export.add_argument(
        "--dict", dest="dict_path", default=None, help="File từ điển để đóng gói"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        build_stats(
            output_dir=args.output_dir,
            dict_path=args.dict_path,
            workers=args.workers,
            recursive=args.recursive,
            corpus_dir=args.corpus,
        )
        print("Đã build lại model. Khởi động lại server để nạp model mới.")
        return 0

    if args.command == "export":
        export_stats(args.output, dict_path=args.dict_path)
        return 0

    socket_path = args.socket
    if socket_path is None:
        try:
            socket_path = default_socket_path()
        except RuntimeDirUnavailable as exc:
            parser.error(str(exc))

    serve(socket_path, model_path=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())