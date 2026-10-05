"""Spellchecker operations behind the socket protocol.

Holds the checker and personalization caches. Requests arrive over a Unix
socket from a native caller, so nothing here depends on a web framework.
"""

import contextlib
import io
import os
import time
from threading import Lock, RLock
from typing import Any

from builder import build_language_stats_from_folder
from checker import NGramSpellChecker
from config import SpellCheckerConfig
from model_pkg import export_model_package
from paths import config_path as resolve_config_path
from paths import personalization_dir
from personalization import PersonalizationManager

SERVER_DATA_FOLDER = "data/corpus"
MAX_INPUT_CHARS = 2000
MIN_CONTEXT_LEN = 2

CheckerCacheKey = tuple[str, str | None, str | None, bool, bool]


class SpellCheckerService:
    def __init__(
        self,
        model_path: str | None = None,
        data_dir: str | None = None,
        config_path: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._data_dir = data_dir
        self._cache: dict[CheckerCacheKey, NGramSpellChecker] = {}
        self._cache_lock = Lock()
        self._config_path = config_path or resolve_config_path()
        self._config: SpellCheckerConfig | None = None
        self._personalization: PersonalizationManager | None = None
        self._init_lock = RLock()

    def server_config(self) -> SpellCheckerConfig:
        """Cached config with the model path applied. Does not change at runtime."""
        with self._init_lock:
            if self._config is None:
                config = SpellCheckerConfig.from_json(self._config_path)
                if self._model_path:
                    config.stats_path = self._model_path
                self._config = config
            return self._config

    def personalization(self) -> PersonalizationManager:
        """The single learned-vocabulary store for the local user."""
        with self._init_lock:
            if self._personalization is None:
                cfg = self.server_config()
                self._personalization = PersonalizationManager(
                    data_dir=self._data_dir or personalization_dir(),
                    max_memory_size=cfg.max_personal_memory_size,
                    priority_score=cfg.priority_score,
                    boost_factor=cfg.boost_factor,
                )
            return self._personalization

    def get_checker(self, debug: bool = False, detail: bool = False) -> NGramSpellChecker:
        config = self.server_config()
        if not os.path.exists(config.stats_path):
            raise FileNotFoundError(
                f"Không tìm thấy thư mục dữ liệu thống kê tại '{config.stats_path}'."
            )

        cache_key: CheckerCacheKey = (
            self._config_path,
            config.stats_path,
            config.dict_path,
            debug,
            detail,
        )
        with self._cache_lock:
            checker = self._cache.get(cache_key)
            if checker is None:
                with contextlib.redirect_stdout(io.StringIO()):
                    checker = NGramSpellChecker(
                        config=config, debug=debug, detail_log=detail
                    )
                self._cache[cache_key] = checker
        return checker

    def preload(self) -> None:
        self.get_checker()

    def check(self, text: str, top_k: int, personalized: bool = False) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Thiếu trường 'text'.")
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(f"Văn bản vượt quá giới hạn {MAX_INPUT_CHARS} ký tự.")
        top_k = max(1, int(top_k))

        start_time = time.perf_counter()
        personalization = self.personalization() if personalized else None
        checker = self.get_checker()
        suggestions = checker.correct_sentence(
            text, top_k=top_k, personalization=personalization
        )

        return {
            "text": text,
            "top_k": top_k,
            "best_correction": suggestions[0] if suggestions else text,
            "suggestions": suggestions,
            "personalized": personalization is not None,
            "processing_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }

    def learn(self, context: Any) -> dict:
        if not isinstance(context, list) or not all(
            isinstance(item, str) for item in context
        ):
            raise ValueError("'context' phải là danh sách các chuỗi.")
        if len(context) < MIN_CONTEXT_LEN:
            raise ValueError("'context' cần ít nhất 2 phần tử.")
        # Recording a memory entry reads only the manager, so no checker is
        # built. Going through get_checker would cost a model load.
        self.personalization().learn_selection(context)
        return {"status": "ok"}

    def learn_text(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Thiếu trường 'text'.")
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(f"Văn bản vượt quá giới hạn {MAX_INPUT_CHARS} ký tự.")
        result = self.personalization().learn_text(text)
        return {"status": "ok", **result}

    def profile(self) -> dict:
        return self.personalization().profile

    def clear(self) -> dict:
        # clear_all resets the manager in place, and the cached checkers read it
        # through that same object, so they stay valid. Dropping the checker
        # cache here would force a full model reload on the next requests.
        self.personalization().clear_all()
        return {"status": "ok"}


def build_stats(
    output_dir: str | None = None,
    dict_path: str | None = None,
    workers: int = 1,
    recursive: bool = False,
    corpus_dir: str = SERVER_DATA_FOLDER,
) -> None:
    build_language_stats_from_folder(
        folder_path=corpus_dir,
        output_dir=output_dir or SpellCheckerConfig.from_json(
            resolve_config_path()
        ).stats_path,
        external_dict_path=dict_path,
        num_workers=workers,
        recursive=recursive,
    )


def export_stats(output_path: str, dict_path: str | None = None) -> str:
    config_path = resolve_config_path()
    config = SpellCheckerConfig.from_json(config_path)
    export_model_package(
        stats_path=config.stats_path,
        config_path=config_path,
        dict_path=dict_path if dict_path is not None else config.dict_path,
        output_path=output_path,
    )
    return output_path
