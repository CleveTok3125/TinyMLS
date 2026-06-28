import argparse
import os

from api import create_app, get_server_config
from model_pkg import export_model_package

app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyMLS Spell Check Server")
    parser.add_argument("--export", metavar="OUTPUT", nargs="?", const="model.tinymls",
                        help="Export trained model to .tinymls package and exit")
    args = parser.parse_args()

    if args.export:
        cfg = get_server_config()
        export_model_package(
            stats_path=cfg.stats_path,
            config_path="config.json",
            dict_path=cfg.dict_path,
            output_path=args.export,
        )
        return

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "").lower() in {"1", "true", "yes"}
    if debug:
        app.run(host=host, port=port, debug=True)
        return

    try:
        from waitress import serve
    except ImportError:
        app.run(host=host, port=port, debug=False)
        return

    serve(app, host=host, port=port)


if __name__ == "__main__":
    main()
