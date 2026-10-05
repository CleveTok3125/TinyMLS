"""Command line entry point for the spellchecker server."""
import argparse
import os

from paths import RuntimeDirUnavailable
from paths import socket_path as default_socket_path
from server import serve
from service import SERVER_DATA_FOLDER, build_stats, export_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyMLS Spell Checker Server")
    parser.add_argument(
        "--socket",
        default=os.getenv("TINYMLS_SOCKET"),
        help="Đường dẫn Unix socket (mặc định: $XDG_RUNTIME_DIR/tinymls.sock)",
    )
    parser.add_argument(
        "model", nargs="?", metavar="MODEL",
        help="Đường dẫn model (.tinymls hoặc thư mục, mặc định: trained_model/)",
    )
    parser.add_argument(
        "--export", metavar="OUTPUT", nargs="?", const="model.tinymls",
        help="Export model thành file .tinymls rồi thoát",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Xây dựng lại thống kê N-gram từ corpus rồi thoát",
    )
    parser.add_argument(
        "--corpus", default=SERVER_DATA_FOLDER,
        help=f"Thư mục corpus cho --build (mặc định: {SERVER_DATA_FOLDER})",
    )
    parser.add_argument(
        "--dict", dest="dict_path", default=None,
        help="File từ điển dùng cho build và export",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Số tiến trình khi build thống kê (mặc định: 1)",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Đọc đệ quy các thư mục con khi build",
    )
    args = parser.parse_args()


    if args.build:
        build_stats(
            output_dir=args.model,
            dict_path=args.dict_path,
            workers=args.workers,
            recursive=args.recursive,
            corpus_dir=args.corpus,
        )
        print("Đã build lại model. Khởi động lại server để nạp model mới.")
        return

    if args.export:
        export_stats(args.export, dict_path=args.dict_path)
        return

    socket_path = args.socket
    if socket_path is None:
        try:
            socket_path = default_socket_path()
        except RuntimeDirUnavailable as exc:
            parser.error(str(exc))

    serve(socket_path, model_path=args.model)


if __name__ == "__main__":
    main()
