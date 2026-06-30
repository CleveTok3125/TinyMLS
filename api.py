import contextlib
import hashlib
import io
import os
import time
from threading import Condition, Lock
from typing import Any

from flask import Flask, jsonify, request

from builder import build_language_stats_from_folder
from checker import NGramSpellChecker
from config import SpellCheckerConfig
from model_pkg import export_model_package
from personalization import PersonalizationManager

CheckerCacheKey = tuple[str, str, str | None, bool, bool, str | None]
SERVER_CONFIG_PATH = "config.json"
SERVER_DATA_FOLDER = "data/corpus"
SERVER_MODEL_PATH: str | None = None
MAX_INPUT_CHARS = 2000


class SpellCheckerService:
    def __init__(self) -> None:
        self._cache: dict[CheckerCacheKey, NGramSpellChecker] = {}
        self._cache_lock = Lock()
        self._active_checker: NGramSpellChecker | None = None
        self._active_key: CheckerCacheKey | None = None
        self._state_lock = Lock()
        self._idle_condition = Condition(self._state_lock)
        self._active_requests = 0
        self._build_in_progress = False
        self._last_load_error: str | None = None
        self._personalization_cache: dict[str, PersonalizationManager] = {}
        self._personalization_cache_lock = Lock()

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()
            self._active_checker = None
            self._active_key = None

    def _build_config(
        self,
        config_path: str,
        stats_path: str | None,
        dict_path: str | None,
    ) -> SpellCheckerConfig:
        config = SpellCheckerConfig.from_json(config_path)
        if stats_path:
            config.stats_path = stats_path
        if dict_path:
            config.dict_path = dict_path
        return config

    def get_checker(
        self,
        config_path: str,
        stats_path: str | None,
        dict_path: str | None,
        debug: bool = False,
        detail: bool = False,
        personalization: PersonalizationManager | None = None,
    ) -> NGramSpellChecker:
        config = self._build_config(config_path, stats_path, dict_path)

        if not os.path.exists(config.stats_path):
            raise FileNotFoundError(
                f"Không tìm thấy thư mục dữ liệu thống kê tại '{config.stats_path}'."
            )

        pers_hash = personalization.user_hash if personalization else None
        cache_key = (
            config_path,
            config.stats_path,
            config.dict_path,
            debug,
            detail,
            pers_hash,
        )
        with self._cache_lock:
            checker = self._cache.get(cache_key)
            if checker is None:
                with contextlib.redirect_stdout(io.StringIO()):
                    checker = NGramSpellChecker(
                        config=config,
                        debug=debug,
                        detail_log=detail,
                        personalization=personalization,
                    )
                self._cache[cache_key] = checker

        return checker

    def preload(self) -> None:
        server_config = get_server_config()
        try:
            checker = self.get_checker(
                config_path=SERVER_CONFIG_PATH,
                stats_path=server_config.stats_path,
                dict_path=server_config.dict_path,
                debug=False,
                detail=False,
            )
        except Exception as exc:
            self._last_load_error = str(exc)
            raise

        cache_key: CheckerCacheKey = (
            SERVER_CONFIG_PATH,
            server_config.stats_path,
            server_config.dict_path,
            False,
            False,
            None,
        )
        with self._cache_lock:
            self._active_checker = checker
            self._active_key = cache_key
        self._last_load_error = None

    def get_active_checker(
        self,
        personalization: PersonalizationManager | None = None,
    ) -> NGramSpellChecker:
        if personalization:
            server_config = get_server_config()
            return self.get_checker(
                config_path=SERVER_CONFIG_PATH,
                stats_path=server_config.stats_path,
                dict_path=server_config.dict_path,
                personalization=personalization,
            )
        with self._cache_lock:
            checker = self._active_checker
        if checker is None:
            self.preload()
            with self._cache_lock:
                checker = self._active_checker
        if checker is None:
            raise RuntimeError("Spell checker chưa được khởi tạo.")
        return checker

    def get_personalization(self, passphrase: str) -> PersonalizationManager:
        cfg = get_server_config()
        h = hashlib.sha256(passphrase.encode()).hexdigest()[:32]
        with self._personalization_cache_lock:
            pm = self._personalization_cache.get(h)
            if pm is None:
                pm = PersonalizationManager(
                    passphrase=passphrase,
                    data_dir=cfg.personalization_dir,
                    max_memory_size=cfg.max_personal_memory_size,
                )
                self._personalization_cache[h] = pm
            return pm

    def begin_check(self) -> None:
        with self._idle_condition:
            while self._build_in_progress:
                self._idle_condition.wait()
            self._active_requests += 1

    def end_check(self) -> None:
        with self._idle_condition:
            self._active_requests = max(0, self._active_requests - 1)
            if self._active_requests == 0:
                self._idle_condition.notify_all()

    def begin_build(self) -> None:
        with self._idle_condition:
            while self._build_in_progress:
                self._idle_condition.wait()
            self._build_in_progress = True
            while self._active_requests > 0:
                self._idle_condition.wait()

    def end_build(self) -> None:
        with self._idle_condition:
            self._build_in_progress = False
            self._idle_condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._cache_lock:
            checker_loaded = self._active_checker is not None
        with self._state_lock:
            build_in_progress = self._build_in_progress
            active_requests = self._active_requests

        return {
            "checker_loaded": checker_loaded,
            "build_in_progress": build_in_progress,
            "active_requests": active_requests,
            "last_load_error": self._last_load_error,
        }


service = SpellCheckerService()


def get_server_config() -> SpellCheckerConfig:
    config = SpellCheckerConfig.from_json(SERVER_CONFIG_PATH)
    if SERVER_MODEL_PATH:
        config.stats_path = SERVER_MODEL_PATH
    return config


def normalize_logs(raw_logs: str) -> str:
    return raw_logs.replace("\r\n", "\n").replace("\r", "\n")


def add_cors_headers(response: Any) -> Any:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def health() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)
    status = service.status()
    return jsonify(
        {
            "status": "ok" if status["checker_loaded"] and not status["last_load_error"] else "degraded",
            **status,
        }
    )


def check_text() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Thiếu trường 'text'."}), 400
    if len(text) > MAX_INPUT_CHARS:
        return jsonify({"error": f"Văn bản vượt quá giới hạn {MAX_INPUT_CHARS} ký tự."}), 400

    top_k = int(payload.get("top_k", 5))
    passphrase = str(payload.get("passphrase", "")).strip() or None

    start_time = time.perf_counter()
    service.begin_check()
    try:
        personalization = None
        if passphrase:
            personalization = service.get_personalization(passphrase)
        checker = service.get_active_checker(personalization=personalization)
        suggestions = checker.correct_sentence(text, top_k=top_k)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Lỗi vận hành: {exc}"}), 500
    finally:
        service.end_check()

    return jsonify(
        {
            "text": text,
            "top_k": top_k,
            "best_correction": suggestions[0] if suggestions else text,
            "suggestions": suggestions,
            "personalized": personalization is not None,
            "processing_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }
    )


def learn_from_selection() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request.get_json(silent=True) or {}
    passphrase = str(payload.get("passphrase", "")).strip()
    if not passphrase:
        return jsonify({"error": "Thiếu 'passphrase'."}), 400
    context = payload.get("context")
    if not isinstance(context, list) or len(context) < 2:
        return jsonify({"error": "Cần ít nhất 2 phần tử trong 'context'."}), 400
    pm = service.get_personalization(passphrase)
    pm.learn_selection(context)
    return jsonify({"status": "ok"})


def learn_from_text() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request.get_json(silent=True) or {}
    passphrase = str(payload.get("passphrase", "")).strip()
    if not passphrase:
        return jsonify({"error": "Thiếu 'passphrase'."}), 400
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Thiếu 'text'."}), 400
    pm = service.get_personalization(passphrase)
    result = pm.learn_text(text)
    return jsonify({"status": "ok", **result})


def profile() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)
    passphrase = str(request.args.get("passphrase", "")).strip()
    if not passphrase:
        return jsonify({"error": "Thiếu 'passphrase'."}), 400
    pm = service.get_personalization(passphrase)
    return jsonify(pm.profile)


def clear_personalization() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request.get_json(silent=True) or {}
    passphrase = str(payload.get("passphrase", "")).strip()
    if not passphrase:
        return jsonify({"error": "Thiếu 'passphrase'."}), 400
    pm = service.get_personalization(passphrase)
    pm.clear_all()
    h = hashlib.sha256(passphrase.encode()).hexdigest()[:32]
    with service._personalization_cache_lock:
        service._personalization_cache.pop(h, None)
    with service._cache_lock:
        keys_to_remove = [k for k in service._cache if k[-1] == h]
        for k in keys_to_remove:
            service._cache.pop(k, None)
    return jsonify({"status": "ok"})


def export_stats() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    server_config = get_server_config()
    output = payload.get("output", "model.tinymls")

    stdout_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            export_model_package(
                stats_path=server_config.stats_path,
                config_path=SERVER_CONFIG_PATH,
                dict_path=server_config.dict_path,
                output_path=output,
            )
        return jsonify({
            "message": "Export hoàn tất.",
            "output": os.path.abspath(output),
            "logs": normalize_logs(stdout_buffer.getvalue()),
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "logs": normalize_logs(stdout_buffer.getvalue())}), 500


def build_stats() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    workers = int(payload.get("workers", 1))
    recursive = bool(payload.get("recursive", False))
    server_config = get_server_config()

    stdout_buffer = io.StringIO()
    response: Any

    service.begin_build()
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            build_language_stats_from_folder(
                folder_path=SERVER_DATA_FOLDER,
                output_dir=server_config.stats_path,
                external_dict_path=server_config.dict_path,
                num_workers=workers,
                recursive=recursive,
            )
        service.clear_cache()
        service.preload()
        response = jsonify(
            {
                "message": "Xây dựng thống kê hoàn tất.",
                "logs": normalize_logs(stdout_buffer.getvalue()),
            }
        )
    except FileNotFoundError as exc:
        response = (
            jsonify({"error": str(exc), "logs": normalize_logs(stdout_buffer.getvalue())}),
            404,
        )
    except ValueError as exc:
        response = (
            jsonify({"error": str(exc), "logs": normalize_logs(stdout_buffer.getvalue())}),
            400,
        )
    except Exception as exc:
        response = (
            jsonify(
                {
                    "error": f"Lỗi vận hành: {exc}",
                    "logs": normalize_logs(stdout_buffer.getvalue()),
                }
            ),
            500,
        )
    finally:
        service.end_build()

    return response


def create_app(model_path: str | None = None) -> Flask:
    global SERVER_MODEL_PATH
    SERVER_MODEL_PATH = model_path
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.after_request(add_cors_headers)
    app.add_url_rule("/api/health", view_func=health, methods=["GET", "OPTIONS"])
    app.add_url_rule("/api/check", view_func=check_text, methods=["POST", "OPTIONS"])
    app.add_url_rule("/api/build", view_func=build_stats, methods=["POST", "OPTIONS"])
    app.add_url_rule("/api/export", view_func=export_stats, methods=["POST", "OPTIONS"])
    app.add_url_rule("/api/learn", view_func=learn_from_selection, methods=["POST", "OPTIONS"])
    app.add_url_rule("/api/learn/text", view_func=learn_from_text, methods=["POST", "OPTIONS"])
    app.add_url_rule("/api/profile", view_func=profile, methods=["GET", "OPTIONS"])
    app.add_url_rule("/api/personalization", view_func=clear_personalization, methods=["DELETE", "OPTIONS"])
    try:
        service.preload()
    except Exception:
        pass
    return app
