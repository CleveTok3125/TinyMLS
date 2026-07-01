import hashlib
import json
import os
import shutil
from collections import OrderedDict
from threading import Lock
from typing import Dict, List, Set

from vietnamese import extract_ngrams, extract_sentences_with_words


class PersonalizationMemory:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._data: OrderedDict[str, int] = OrderedDict()
        self._lock = Lock()

    def get_total_boost(self, word: str, *context_words: str) -> int:
        total = 0
        if not context_words:
            return 0
        with self._lock:
            for n in range(1, len(context_words) + 1):
                ctx = " ".join(context_words[-n:])
                key = f"{ctx} {word}"
                if key in self._data:
                    self._data.move_to_end(key)
                    total += self._data[key]
        return total

    def learn(self, context: str, word: str) -> None:
        key = f"{context} {word}"
        with self._lock:
            if key in self._data:
                self._data[key] += 1
                self._data.move_to_end(key)
            else:
                if len(self._data) >= self.max_size:
                    self._data.popitem(last=False)
                self._data[key] = 1

    def to_dict(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._data)

    @classmethod
    def from_dict(cls, data: Dict[str, int], max_size: int = 10000):
        mem = cls(max_size=max_size)
        mem._data = OrderedDict(
            sorted(data.items(), key=lambda x: x[1], reverse=True)
        )
        return mem

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class PersonalizationManager:
    def __init__(
        self,
        passphrase: str,
        data_dir: str = "data/personalization",
        max_memory_size: int = 10000,
        priority_score: float = 5.0,
        boost_factor: float = 2.0,
    ):
        self._user_hash = self._passphrase_hash(passphrase)
        self._base_dir = os.path.join(data_dir, self._user_hash)
        self._max_memory_size = max_memory_size
        self._priority_score = priority_score
        self._boost_factor = boost_factor
        self._memory = PersonalizationMemory(max_size=max_memory_size)
        self._learned_words: Set[str] = set()
        self._learned_contexts: Set[str] = set()
        self._lock = Lock()

        os.makedirs(self._base_dir, exist_ok=True)
        self.load()

    @staticmethod
    def _passphrase_hash(passphrase: str) -> str:
        return hashlib.sha256(passphrase.encode()).hexdigest()[:32]

    @property
    def memory(self) -> PersonalizationMemory:
        return self._memory

    def _memory_path(self) -> str:
        return os.path.join(self._base_dir, "memory.json")

    def _learned_path(self) -> str:
        return os.path.join(self._base_dir, "learned.json")

    def load(self) -> None:
        mem_path = self._memory_path()
        if os.path.exists(mem_path):
            with open(mem_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._memory = PersonalizationMemory.from_dict(
                data, max_size=self._max_memory_size
            )
        learned_path = self._learned_path()
        if os.path.exists(learned_path):
            with open(learned_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._learned_words = set(data.get("words", []))
            self._learned_contexts = set(data.get("contexts", []))

    def save(self) -> None:
        mem_path = self._memory_path()
        os.makedirs(os.path.dirname(mem_path), exist_ok=True)
        with open(mem_path, "w", encoding="utf-8") as f:
            json.dump(self._memory.to_dict(), f, ensure_ascii=False, indent=2)
        learned_path = self._learned_path()
        with open(learned_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "words": sorted(self._learned_words),
                    "contexts": sorted(self._learned_contexts),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def learn_text(self, text: str) -> dict:
        sentences = extract_sentences_with_words(text)
        added_words = 0
        added_contexts = 0

        for words in sentences:
            for w in words:
                if w not in self._learned_words:
                    self._learned_words.add(w)
                    added_words += 1

            for key in extract_ngrams(words, 2, 3):
                if key not in self._learned_contexts:
                    self._learned_contexts.add(key)
                    added_contexts += 1

        total_words = sum(len(s) for s in sentences)
        self.save()
        return {
            "words_added": added_words,
            "contexts_added": added_contexts,
            "total_words": total_words,
        }

    def get_priority_words(self) -> Set[str]:
        return self._learned_words

    def compute_boost(
        self,
        candidate: str,
        prev_word: str | None,
        prev_prev_word: str | None,
    ) -> float:
        boost = 0.0
        if candidate in self._learned_words:
            boost += self._priority_score
        if prev_word:
            if f"{prev_word} {candidate}" in self._learned_contexts:
                boost += self._priority_score
            if prev_prev_word:
                if (
                    f"{prev_prev_word} {prev_word} {candidate}"
                    in self._learned_contexts
                ):
                    boost += self._priority_score

            mem_ctx = [prev_word]
            if prev_prev_word:
                mem_ctx.insert(0, prev_prev_word)
            count = self._memory.get_total_boost(candidate, *mem_ctx)
            if count > 0:
                boost += self._boost_factor * count
        return boost

    def learn_selection(self, context: List[str]) -> None:
        if len(context) < 2:
            return
        selected = context[-1]
        for i in range(len(context) - 2, -1, -1):
            ctx = " ".join(context[i:-1])
            self._memory.learn(ctx, selected)
        self.save()

    def clear_all(self) -> None:
        self._memory.clear()
        self._learned_words.clear()
        self._learned_contexts.clear()
        if os.path.exists(self._base_dir):
            shutil.rmtree(self._base_dir)
        os.makedirs(self._base_dir, exist_ok=True)

    def clear_memory(self) -> None:
        self._memory.clear()
        self.save()

    @property
    def user_hash(self) -> str:
        return self._user_hash

    @property
    def memory_size(self) -> int:
        return len(self._memory)

    @property
    def learned_word_count(self) -> int:
        return len(self._learned_words)

    @property
    def learned_context_count(self) -> int:
        return len(self._learned_contexts)

    @property
    def profile(self) -> dict:
        return {
            "user_hash": self._user_hash,
            "learned_words": self.learned_word_count,
            "learned_contexts": self.learned_context_count,
            "memory_size": self.memory_size,
        }
