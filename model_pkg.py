import argparse
import json
import os
import shutil
import tempfile
import zipfile

import marisa_trie


def export_model_package(
    stats_path: str = "trained_model",
    config_path: str = "config.json",
    dict_path: str | None = None,
    output_path: str = "model.tinymls",
) -> str:
    components = {"config": False, "unigrams": False, "bigrams": False, "trigrams": False, "meta": False, "dictionary": False}

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(config_path):
            zf.write(config_path, "config.json")
            components["config"] = True

        meta_src = os.path.join(stats_path, "language_stats_meta.json")
        if os.path.exists(meta_src):
            zf.write(meta_src, "language_stats_meta.json")
            components["meta"] = True

        for name in ("unigrams", "bigrams", "trigrams"):
            trie_path = os.path.join(stats_path, f"{name}.trie")
            if os.path.exists(trie_path):
                zf.write(trie_path, f"{name}.trie")
                components[name] = True

        if dict_path and os.path.exists(dict_path):
            zf.write(dict_path, "dictionary.dic")
            components["dictionary"] = True

    included = [k for k, v in components.items() if v]
    print(f"Exported {len(included)} components ({', '.join(included)}) to {output_path}")
    return output_path


def extract_model_package(package_path: str, output_dir: str | None = None) -> str:
    if output_dir is None:
        base = os.path.splitext(os.path.basename(package_path))[0]
        output_dir = os.path.join(os.path.dirname(package_path) or ".", base)
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(package_path, "r") as zf:
        zf.extractall(output_dir)
    print(f"Extracted model to {output_dir}/")
    return output_dir


class ModelArchive:
    def __init__(self, path: str):
        self._zf = zipfile.ZipFile(path, "r")
        self._tmpdir: str | None = None

    def has(self, name: str) -> bool:
        try:
            self._zf.getinfo(name)
            return True
        except KeyError:
            return False

    def read_text(self, name: str) -> str:
        return self._zf.read(name).decode("utf-8")

    def read_json(self, name: str) -> dict:
        return json.loads(self._zf.read(name))

    def mmap_trie(self, name: str) -> marisa_trie.RecordTrie:
        data = self._zf.read(name)
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="tinymls_")
        tmp_path = os.path.join(self._tmpdir, name)
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(data)
        return marisa_trie.RecordTrie("<I").mmap(tmp_path)

    def extract_file(self, name: str) -> str:
        data = self._zf.read(name)
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="tinymls_")
        out = os.path.join(self._tmpdir, os.path.basename(name))
        with open(out, "wb") as f:
            f.write(data)
        return out

    def load_config(self) -> dict:
        if self.has("config.json"):
            return self.read_json("config.json")
        return {}

    def close(self) -> None:
        self._zf.close()
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyMLS Model Package Tool")
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export", help="Export model to .tinymls package")
    export_parser.add_argument("--stats", default="trained_model", help="Path to trained model directory")
    export_parser.add_argument("--config", default="config.json", help="Path to config file")
    export_parser.add_argument("--dict", default=None, help="Path to dictionary file to include in package")
    export_parser.add_argument("-o", "--output", default="model.tinymls", help="Output path")

    extract_parser = sub.add_parser("extract", help="Extract .tinymls package to directory")
    extract_parser.add_argument("package", help="Path to .tinymls package")
    extract_parser.add_argument("-o", "--output", default=None, help="Output directory")

    args = parser.parse_args()
    if args.command == "export":
        export_model_package(
            stats_path=args.stats,
            config_path=args.config,
            dict_path=args.dict,
            output_path=args.output,
        )
    elif args.command == "extract":
        extract_model_package(args.package, args.output)


if __name__ == "__main__":
    main()
