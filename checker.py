import json
import math
import os
import unicodedata
from collections import OrderedDict, defaultdict
from enum import Enum
from functools import lru_cache

import marisa_trie
from rapidfuzz import fuzz, process

from builder import VOCAB_FILENAME
from config import SpellCheckerConfig
from keyboard import get_keyboard_coordinates, keyboard_matrix
from model_pkg import ModelArchive
from personalization import PersonalizationManager
from telex import to_standard_telex

_MIN_ERROR_LEN = 3
_MAX_LEN_DIFF = 3
_BONUS_RANGE_LEN = 2
_MIN_ANCHOR_LEN = 2
_MIN_CTX_LEN = 2
_GLOBAL_MATCH_CACHE_SIZE = 20000
_CONTEXT_INDEX_BUDGET = 250000
_KEEP_TABLES: dict[tuple[int, int], bytes] = {}
_EPS = 1e-8



class CasePattern(Enum):
    ALLCASE = "allcase"
    LOWER = "lower"
    INITCASE = "initcase"


def detect_case_pattern(word: str) -> CasePattern:
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return CasePattern.LOWER
    if all(c.isupper() for c in letters):
        return CasePattern.ALLCASE
    if all(c.islower() for c in letters):
        return CasePattern.LOWER
    return CasePattern.INITCASE


def apply_case_pattern(word: str, pattern: CasePattern) -> str:
    if pattern == CasePattern.ALLCASE:
        return word.upper()
    if pattern == CasePattern.LOWER:
        return word.lower()
    return word.capitalize()


class NGramSpellChecker:
    """Suggestion engine for one model.

    Personalization is a per-call argument rather than instance state: it is the
    only thing that varies between requests against the same model, so keeping
    it out of the constructor lets one instance serve both personalised and
    plain checks instead of paying a second full model load.
    """

    def __init__(
        self,
        config: SpellCheckerConfig,
        debug: bool = False,
        detail_log: bool = False,
    ) -> None:
        self.cfg = config
        self.debug = debug
        self.detail_log = detail_log
        self._archive: ModelArchive | None = None
        self._global_telex_cache: dict[tuple[str, int], tuple[str, ...]] = {}
        self._context_index_cache: OrderedDict[str, tuple[list[str], list[str], bytes]] = OrderedDict()
        self._context_index_size = 0

        # Resolved once: these were read via getattr on every score call.
        self._sim_weight = getattr(self.cfg, "sim_weight", 0.0)
        self._context_weight = getattr(self.cfg, "context_weight", 0.0)
        self._stutter_penalty = getattr(self.cfg, "stutter_penalty", 0.0)
        self._lambda_1 = getattr(self.cfg, "lambda_1", 0.0)
        self._lambda_2 = getattr(self.cfg, "lambda_2", 0.0)
        self._lambda_3 = getattr(self.cfg, "lambda_3", 0.0)
        self._top_n = self.cfg.top_n
        self._cutoff = self.cfg.cutoff

        if os.path.isdir(self.cfg.stats_path):
            self._load_from_dir()
        elif self.cfg.stats_path.endswith(
            ".tinymls"
        ) or self.cfg.stats_path.endswith(".zip"):
            self._load_from_archive()
        else:
            raise FileNotFoundError(f"Không tìm thấy model: {self.cfg.stats_path}")

    def _resolve_dict_path(self) -> str | None:
        if self.cfg.dict_path and os.path.exists(self.cfg.dict_path):
            return self.cfg.dict_path
        return None

    def _load_standard_dict(self, dict_path: str | None) -> None:
        self.standard_dict: set[str] = set()
        if not dict_path:
            return
        try:
            with open(dict_path, encoding="utf-8") as f:
                self.standard_dict = {
                    line.strip().lower() for line in f if line.strip()
                }
            print(f"Đã nạp {len(self.standard_dict)} từ chuẩn từ '{dict_path}'.")
        except FileNotFoundError:
            pass

    def _load_from_dir(self) -> None:
        stats_dir = self.cfg.stats_path
        print(f"Loading dữ liệu thống kê từ thư mục: {stats_dir}...")

        meta_path = os.path.join(stats_dir, "language_stats_meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        vocab_path = os.path.join(stats_dir, VOCAB_FILENAME)
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"Không tìm thấy từ điển '{vocab_path}'. "
                "Hãy build lại model để tạo file này."
            )
        with open(vocab_path, encoding="utf-8") as f:
            raw_vocab = f.read().split("\n")
        vocab_list = [unicodedata.normalize("NFC", w) for w in raw_vocab if w]
        self.total_unigrams = meta["total_unigrams"]

        self.unigrams = marisa_trie.RecordTrie("<I").mmap(
            os.path.join(stats_dir, "unigrams.trie")
        )
        self.bigrams = marisa_trie.RecordTrie("<I").mmap(
            os.path.join(stats_dir, "bigrams.trie")
        )
        tri_path = os.path.join(stats_dir, "trigrams.trie")
        self.trigrams = (
            marisa_trie.RecordTrie("<I").mmap(tri_path)
            if os.path.exists(tri_path) else None
        )

        self._init_from_vocab(vocab_list)
        self._load_standard_dict(self._resolve_dict_path())

    def _load_from_archive(self) -> None:
        self._archive = ModelArchive(self.cfg.stats_path)
        print(f"Loading dữ liệu thống kê từ package: {self.cfg.stats_path}...")

        meta = self._archive.read_json("language_stats_meta.json")
        if not self._archive.has(VOCAB_FILENAME):
            raise FileNotFoundError(
                f"Package '{self.cfg.stats_path}' không chứa '{VOCAB_FILENAME}'. "
                "Hãy export lại model từ thư mục đã build."
            )
        raw_vocab = self._archive.read_text(VOCAB_FILENAME).split("\n")
        vocab_list = [unicodedata.normalize("NFC", w) for w in raw_vocab if w]
        self.total_unigrams = meta["total_unigrams"]

        self.unigrams = self._archive.mmap_trie("unigrams.trie")
        self.bigrams = self._archive.mmap_trie("bigrams.trie")
        self.trigrams = (
            self._archive.mmap_trie("trigrams.trie")
            if self._archive.has("trigrams.trie") else None
        )

        self._init_from_vocab(vocab_list)

        dict_path = self._resolve_dict_path()
        if not dict_path and self._archive.has("dictionary.dic"):
            dict_path = self._archive.extract_file("dictionary.dic")
        self._load_standard_dict(dict_path)

    def close(self) -> None:
        if self._archive:
            self._archive.close()
            self._archive = None

    def _init_from_vocab(self, vocab_list: list[str]) -> None:
        self.telex_to_vocab: dict[str, str | list[str]] = {}
        # 97.8% of telex keys map to exactly one surface form, so a bare str is
        # stored for those instead of a one-element list (~140k fewer objects).
        for w in vocab_list:
            t = to_standard_telex(w)
            existing = self.telex_to_vocab.get(t)
            if existing is None:
                self.telex_to_vocab[t] = w
            elif isinstance(existing, str):
                self.telex_to_vocab[t] = [existing, w]
            else:
                existing.append(w)

        telex_vocab_list: list[str] = list(self.telex_to_vocab.keys())
        self.telex_by_length = defaultdict(list)
        for t in telex_vocab_list:
            self.telex_by_length[len(t)].append(t)

        sorted_unigrams = sorted(
            self.unigrams.items(), key=lambda item: item[1], reverse=True
        )
        self.unigram_rankings: dict[str, int] = {
            word: rank for rank, (word, _) in enumerate(sorted_unigrams, start=1)
        }
        self.total_ranked_unigrams = len(sorted_unigrams)
        top_k = getattr(self.cfg, "auto_ambiguous_top_k", 50)
        self.dynamic_ambiguous_words: set[str] = {
            word for word, _ in sorted_unigrams[:top_k]
        }
        if self.debug:
            n_ambig = len(self.dynamic_ambiguous_words)
            top10 = list(self.dynamic_ambiguous_words)[:10]
            print(
                f"Tự động cấm Anchor {n_ambig} từ phổ biến: {top10}..."
            )

        self.kb_coords = get_keyboard_coordinates(keyboard_matrix=keyboard_matrix)
        print(f"Done! The dictionary has {len(vocab_list)} words.")

    def get_trie_count(self, trie, key: str) -> int:
        res = trie.get(unicodedata.normalize("NFC", key))
        return res[0][0] if res else 0

    def allowed_length_window(self, error_len: int) -> tuple[int, int]:
        """Inclusive candidate telex length bounds accepted for error_len."""
        if error_len >= _MIN_ERROR_LEN:
            low = max(_MIN_ERROR_LEN, error_len - _MAX_LEN_DIFF)
        else:
            low = 1
        return low, error_len + _MAX_LEN_DIFF

    @staticmethod
    def length_keep_table(low: int, high: int) -> bytes:
        """Byte table mapping lengths in [low, high] to 1, everything else to 0.

        Lets one bytes.translate call filter a whole batch of lengths.
        """
        table = _KEEP_TABLES.get((low, high))
        if table is None:
            table = bytes(1 if low <= i <= high else 0 for i in range(256))
            _KEEP_TABLES[(low, high)] = table
        return table

    @staticmethod
    def kept_positions(lengths: bytes, low: int, high: int) -> list[int]:
        """Indices of accepted lengths, ascending."""
        mask = lengths.translate(NGramSpellChecker.length_keep_table(low, high))
        found = []
        pos = mask.find(1)
        while pos >= 0:
            found.append(pos)
            pos = mask.find(1, pos + 1)
        return found

    def get_fast_close_matches(
        self, target: str, possibilities: list[str], n: int, cutoff: float
    ) -> list[str]:
        results = process.extract(
            target, possibilities, scorer=fuzz.ratio, limit=n, score_cutoff=cutoff * 100
        )

        return [r[0] for r in results]

    def get_global_telex_matches(self, error_word: str, error_len: int) -> tuple[str, ...]:
        """Fuzzy-match result for a token against the whole vocabulary.

        Independent of the preceding word, but Viterbi calls get_candidates once
        per beam path, so this ~92k-word scan otherwise repeats up to
        beam_width times for the same token. Cached per instance rather than
        with a module-level lru_cache so that discarded checkers are collectable.
        """
        cache_key = (error_word, error_len)
        cached = self._global_telex_cache.get(cache_key)
        if cached is not None:
            return cached

        filtered_global_telex = []
        min_len = (
            max(_MIN_ERROR_LEN, error_len - _MAX_LEN_DIFF)
            if error_len >= _MIN_ERROR_LEN else 1
        )
        max_len = error_len + _MAX_LEN_DIFF

        for length in range(min_len, max_len + 1):
            if length in self.telex_by_length:
                filtered_global_telex.extend(self.telex_by_length[length])

        matches = tuple(
            self.get_fast_close_matches(
                to_standard_telex(error_word),
                filtered_global_telex,
                n=self._top_n,
                cutoff=self._cutoff,
            )
        )

        if len(self._global_telex_cache) >= _GLOBAL_MATCH_CACHE_SIZE:
            self._global_telex_cache.clear()
        self._global_telex_cache[cache_key] = matches
        return matches

    def context_index(self, prev_word: str) -> tuple[list[str], list[str], bytes]:
        """Successors of prev_word as parallel telex/word lists plus a length array.

        Depends only on prev_word, so it is cached. The length array lets the
        per-call filter run as one batch pass instead of a Python loop over every
        successor, which for a common preceding word can be several hundred.
        """
        cached = self._context_index_cache.get(prev_word)
        if cached is not None:
            self._context_index_cache.move_to_end(prev_word)
            return cached

        prefix = f"{prev_word} "
        telexes: list[str] = []
        words: list[str] = []
        for key in self.bigrams.keys(prefix):
            cw = key[len(prefix) :]
            if not cw:
                continue
            telexes.append(to_standard_telex(cw))
            words.append(cw)
        # One byte per successor. Telex forms are far shorter than 256, so the
        # length fits in a byte; the clamp keeps a pathological entry from
        # corrupting the batch filter.
        lengths = bytes(min(len(t), 255) for t in telexes)

        # Budgeted by total successors, not entry count: one common preceding
        # word can carry hundreds of them. Evicts least-recently-used entries so
        # a single large entry cannot flush the whole cache.
        self._context_index_size += len(telexes)
        while (
            self._context_index_cache
            and self._context_index_size > _CONTEXT_INDEX_BUDGET
        ):
            _evicted, old = self._context_index_cache.popitem(last=False)
            self._context_index_size -= len(old[0])
        entry = (telexes, words, lengths)
        self._context_index_cache[prev_word] = entry
        return entry

    def get_candidates(  # noqa: C901, PLR0912
        self,
        error_word: str,
        prev_word: str | None = None,
        personalization: PersonalizationManager | None = None,
    ) -> list[str]:
        candidates: list[str] = []

        error_len = len(to_standard_telex(error_word))

        if prev_word:
            telexes, words, lengths = self.context_index(prev_word)

            if telexes:
                # Vectorised length filter; only survivors reach the dict, and
                # flatnonzero preserves ascending index order so the telex list
                # keeps the same order as a plain Python loop. The accepted
                # lengths form a contiguous range, so a range mask beats isin.
                low, high = self.allowed_length_window(error_len)
                selected = self.kept_positions(lengths, low, high)
                context_telex_to_word: dict[str, list[str]] = {}
                for i in selected:
                    ct = telexes[i]
                    if ct in context_telex_to_word:
                        context_telex_to_word[ct].append(words[i])
                    else:
                        context_telex_to_word[ct] = [words[i]]
            else:
                context_telex_to_word = {}

            if context_telex_to_word:
                context_telex_matches: list[str] = self.get_fast_close_matches(
                    to_standard_telex(error_word),
                    list(context_telex_to_word),
                    n=self._top_n,
                    cutoff=self._cutoff,
                )

                for ctm in context_telex_matches:
                    for raw_word in context_telex_to_word[ctm]:
                        rw = unicodedata.normalize("NFC", raw_word)
                        if rw not in candidates:
                            candidates.append(rw)

        if len(candidates) < self._top_n:
            # telex_to_vocab values are NFC already (normalised in
            # _init_from_vocab), so re-normalising here is a no-op.
            for gtm in self.get_global_telex_matches(error_word, error_len):
                forms = self.telex_to_vocab[gtm]
                if isinstance(forms, str):
                    if forms not in candidates:
                        candidates.append(forms)
                    continue
                for word in forms:
                    if word not in candidates:
                        candidates.append(word)

        candidates = candidates[: self._top_n]

        if personalization:
            err_first = (
                unicodedata.normalize("NFC", error_word)[0] if error_word else ""
            )
            low_len, high_len = self.allowed_length_window(error_len)
            # Only learned words sharing the token's first character can ever be
            # accepted below, so ask for that slice instead of every learned word.
            for pw_norm in personalization.get_priority_words(err_first):
                if pw_norm in candidates:
                    continue
                pw_len = len(to_standard_telex(pw_norm))
                if low_len <= pw_len <= high_len:
                    if len(candidates) >= self._top_n:
                        candidates.pop()
                    candidates.append(pw_norm)

        return candidates

    def get_kb_cost(self, char1: str, char2: str) -> float:
        if char1 == char2:
            return 0.0

        if char1 not in self.kb_coords or char2 not in self.kb_coords:
            return self.cfg.unknown_char_penalty

        x1, y1 = self.kb_coords[char1]
        x2, y2 = self.kb_coords[char2]
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        return min(dist / self.cfg.max_kb_distance, 1.0)

    @lru_cache(maxsize=100000)
    def keyboard_aware_similarity(self, word1: str, word2: str) -> float:
        m, n = len(word1), len(word2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = float(i)
        for j in range(n + 1):
            dp[0][j] = float(j)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = self.get_kb_cost(word1[i - 1], word2[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,
                    dp[i][j - 1] + 1.0,
                    dp[i - 1][j - 1] + cost,
                )

                if (
                    i > 1
                    and j > 1
                    and word1[i - 1] == word2[j - 2]
                    and word1[i - 2] == word2[j - 1]
                ):
                    dp[i][j] = min(
                        dp[i][j], dp[i - 2][j - 2] + self.cfg.transposition_cost
                    )

        max_len = max(m, n)
        if max_len == 0:
            return 1.0

        distance = dp[m][n]
        sim = math.exp(-distance / max_len)
        final_sim = max(0.0, sim)

        return max(0.0, final_sim)

    def context_base_counts(self, w1: str | None, w2: str) -> tuple[int, int]:
        """Counts that depend only on the preceding words, not on the candidate.

        bigram(w1 w2) and unigram(w2) are looked up once per surviving path
        instead of once per (candidate, path) pair.
        """
        bigram_w1_w2 = (
            self.get_trie_count(self.bigrams, f"{w1} {w2}") if w1 else 0
        )
        return bigram_w1_w2, self.get_trie_count(self.unigrams, w2)

    def context_prob_from_base(
        self,
        w1: str | None,
        w2: str,
        w3: str,
        bigram_w1_w2_count: int,
        unigram_w2_count: int,
        unigram_w3_count: int,
    ) -> float:
        """Context probability using pre-fetched counts; see context_base_counts."""
        p_tri = 0.0
        if w1:
            trigram_count = (
                self.get_trie_count(self.trigrams, f"{w1} {w2} {w3}")
                if self.trigrams
                else 0
            )
            p_tri = (
                trigram_count / bigram_w1_w2_count if bigram_w1_w2_count > 0 else 0.0
            )

        bigram_w2_w3_count = self.get_trie_count(self.bigrams, f"{w2} {w3}")
        p_bi = bigram_w2_w3_count / unigram_w2_count if unigram_w2_count > 0 else 0.0

        p_uni = (
            unigram_w3_count / self.total_unigrams if self.total_unigrams > 0 else 0.0
        )

        l3 = self._lambda_3
        l2 = self._lambda_2
        l1 = self._lambda_1

        if not w1:
            total_l = l2 + l1
            return (l2 * p_bi + l1 * p_uni) / total_l if total_l > 0 else 0.0

        return (l3 * p_tri) + (l2 * p_bi) + (l1 * p_uni)

    def calculate_context_prob(self, w1: str | None, w2: str, w3: str) -> float:
        bigram_w1_w2_count, unigram_w2_count = self.context_base_counts(w1, w2)
        return self.context_prob_from_base(
            w1, w2, w3, bigram_w1_w2_count, unigram_w2_count,
            self.get_trie_count(self.unigrams, w3),
        )

    def candidate_base_score(
        self, candidate: str, error_word: str
    ) -> tuple[float, float, float, str, str, int]:
        """Scoring inputs that depend only on (candidate, error_word).

        Viterbi scores every candidate once per surviving path, so these are
        hoisted out of the path loop and computed once per candidate.
        Returns (sim_feat, p_uni, exact_match_bonus, candidate, error_word,
        unigram_count); the count is reused instead of re-read by the context
        probability, which needs unigram(candidate).
        """
        candidate = unicodedata.normalize("NFC", candidate)
        error_word = unicodedata.normalize("NFC", error_word)

        cand_telex = to_standard_telex(candidate)
        err_telex = to_standard_telex(error_word)

        sim = self.keyboard_aware_similarity(err_telex, cand_telex)
        sim_feat = math.log(sim + _EPS)

        count = self.get_trie_count(self.unigrams, candidate)
        p_uni = count / self.total_unigrams if self.total_unigrams > 0 else 0.0
        exact_bonus = self.calculate_exact_match_bonus(candidate, error_word)

        return sim_feat, p_uni, exact_bonus, candidate, error_word, count

    def score_from_base(
        self,
        candidate: str,
        prev_word: str | None,
        prev_prev_word: str | None,
        sim_feat: float,
        p_uni: float,
        exact_bonus: float,
        context_base: tuple[int, int] = (0, 0),
        unigram_count: int = 0,
        personalization: PersonalizationManager | None = None,
    ) -> float:
        """Path-dependent half of the score; see candidate_base_score."""
        if prev_word:
            p_ctx = self.context_prob_from_base(
                prev_prev_word, prev_word, candidate,
                context_base[0], context_base[1], unigram_count,
            )

            if candidate == prev_word:
                p_ctx *= self._stutter_penalty
        else:
            p_ctx = max(p_uni, _EPS)

        ctx_feat = (math.log(p_ctx + _EPS) + 10) / 10

        score = (self._sim_weight * sim_feat) + (self._context_weight * ctx_feat)
        score += exact_bonus

        if personalization:
            score += personalization.compute_boost(
                candidate, prev_word, prev_prev_word
            )

        return score

    def calculate_score(
        self,
        candidate: str,
        error_word: str,
        prev_word: str | None,
        prev_prev_word: str | None = None,
        personalization: PersonalizationManager | None = None,
    ) -> float:
        sim_feat, p_uni, exact_bonus, candidate, error_word, uni_count = (
            self.candidate_base_score(candidate, error_word)
        )
        score = self.score_from_base(
            candidate, prev_word, prev_prev_word, sim_feat, p_uni, exact_bonus,
            self.context_base_counts(prev_prev_word, prev_word) if prev_word else (0, 0),
            uni_count, personalization,
        )

        if self.debug and self.detail_log:
            prev_str = prev_word if prev_word else "[START]"
            print("ERROR:", to_standard_telex(error_word))
            print("CAND :", to_standard_telex(candidate))
            print(f"      ➜ '{prev_str}' -> '{candidate}' (error: '{error_word}')")
            print(f"         sim : {sim_feat:.4f} * {self._sim_weight}")
            print(f"         score: {score:.4f}")

        return score

    def calculate_exact_match_bonus(self, candidate: str, error_word: str) -> float:
        if candidate != error_word:
            return 0.0

        rank = self.unigram_rankings.get(candidate)
        if rank is None:
            return 0.0

        bonus_range = getattr(self.cfg, "exact_match_bonus", [0.0, 0.0])
        if not isinstance(bonus_range, list) or len(bonus_range) != _BONUS_RANGE_LEN:
            return 0.0

        min_bonus, max_bonus = bonus_range
        if self.total_ranked_unigrams <= 1:
            return float(max_bonus)

        rank_ratio = (rank - 1) / (self.total_ranked_unigrams - 1)
        return float(max_bonus - ((max_bonus - min_bonus) * rank_ratio))

    def is_delayed_anchor(self, w: str, next_w: str | None) -> bool:
        if not hasattr(self, "standard_dict") or w not in self.standard_dict:
            return False
        if len(w) < _MIN_ANCHOR_LEN or w in getattr(
            self, "dynamic_ambiguous_words", set()
        ):
            return False
        if next_w is None:
            return True

        if f"{w} {next_w}" in self.bigrams:
            return True
        return False

    def correct_sentence(
        self,
        sentence: str,
        top_k: int = 5,
        personalization: PersonalizationManager | None = None,
    ) -> list[str]:  # noqa: C901, PLR0912, PLR0915
        original_words: list[str] = sentence.split()
        case_patterns = [detect_case_pattern(w) for w in original_words]
        words: list[str] = [w.lower() for w in original_words]
        if not words:
            return []

        paths: dict[str, tuple[float, list[str], str]] = {}

        reset_context_next_step = True

        for i, current_word in enumerate(words):
            new_paths: dict[str, tuple[float, list[str], str]] = {}
            is_garbage = False

            if self.debug:
                print(f"\n[VITERBI] Từ thứ {i + 1}: '{current_word}'")

            next_word = words[i + 1] if i + 1 < len(words) else None

            if self.is_delayed_anchor(current_word, next_word):
                candidates = [current_word]
                if self.debug:
                    print(f"  ➜ Đã Neo cứng (Delayed Anchor Passed): '{current_word}'")
            else:
                candidates_set = {}
                if not reset_context_next_step and paths:
                    for _, _, prev_cand in paths.values():
                        for c in self.get_candidates(
                            current_word,
                            prev_word=prev_cand,
                            personalization=personalization,
                        ):
                            candidates_set[c] = None

                for c in self.get_candidates(
                    current_word, prev_word=None,
                    personalization=personalization,
                ):
                    candidates_set[c] = None
                candidates = list(candidates_set.keys())

                if candidates:
                    best_match = process.extractOne(
                        current_word,
                        candidates,
                        scorer=fuzz.ratio,
                        score_cutoff=self._cutoff * 100,
                    )

                    if best_match is None:
                        candidates = [current_word]
                        is_garbage = True
                        if self.debug:
                            cutoff = self._cutoff
                            print(
                                "  ➜ Bỏ cuộc: Rác/Từ lạ"
                                f" (Không có từ nào >= {cutoff})."
                            )
                else:
                    candidates = [current_word]

            step_log_data: list[dict] = []

            # Per-path counts fetched once, not once per (candidate, path).
            path_context_bases: dict[str | None, tuple[int, int]] = {}
            if paths and not reset_context_next_step:
                for _s, _p, prev_c in paths.values():
                    pp = _p[-2] if len(_p) >= _MIN_CTX_LEN else None
                    if prev_c not in path_context_bases:
                        path_context_bases[prev_c] = self.context_base_counts(pp, prev_c)

            for curr_cand in candidates:
                sim_feat, p_uni, exact_bonus, norm_cand, _err, uni_count = (
                    self.candidate_base_score(curr_cand, current_word)
                )

                if reset_context_next_step or not paths:
                    step_score = self.score_from_base(
                        norm_cand, None, None, sim_feat, p_uni, exact_bonus,
                        personalization=personalization,
                    )

                    best_history = []
                    if paths:
                        best_past = max(paths.items(), key=lambda kv: kv[1][0])
                        best_history = best_past[1][1]
                        best_history_key = best_past[0]
                    else:
                        best_history = []
                        best_history_key = ""

                    total_score = step_score
                    new_path = best_history + [curr_cand]

                    # Every key in `paths` is exactly " ".join(its path), so
                    # appending by concatenation is equivalent to rejoining the
                    # whole path and avoids that work per candidate x path.
                    new_path_key = (
                        best_history_key + " " + curr_cand if best_history_key else curr_cand
                    )
                    new_paths[new_path_key] = (total_score, new_path, curr_cand)

                    if self.debug:
                        step_log_data.append(
                            {
                                "cand": curr_cand,
                                "path": f"[MỚI] -> {curr_cand}",
                                "calc_str": f"(Khởi tạo: {step_score:.4f})",
                                "score": total_score,
                            }
                        )

                else:
                    for prev_key, (prev_score, prev_path, prev_cand) in paths.items():
                        prev_prev_cand = (
                            prev_path[-2] if len(prev_path) >= _MIN_CTX_LEN
                            else None
                        )

                        step_score = self.score_from_base(
                            norm_cand, prev_cand, prev_prev_cand,
                            sim_feat, p_uni, exact_bonus,
                            path_context_bases[prev_cand], uni_count,
                            personalization,
                        )

                        total_score = prev_score + step_score

                        new_path = prev_path + [curr_cand]
                        new_path_key = prev_key + " " + curr_cand
                        new_paths[new_path_key] = (total_score, new_path, curr_cand)

                        if self.debug:
                            step_log_data.append(
                                {
                                    "cand": curr_cand,
                                    "path": f"{prev_cand} -> {curr_cand}",
                                    "calc_str": (
                                        f"({prev_score:.4f}"
                                        f" + {step_score:.4f})"
                                    ),
                                    "score": total_score,
                                }
                            )

            paths = new_paths

            if is_garbage:
                reset_context_next_step = True
            else:
                reset_context_next_step = False

            if len(paths) > getattr(self.cfg, "beam_width", 5):
                sorted_paths = sorted(
                    paths.items(), key=lambda item: item[1][0], reverse=True
                )
                paths = dict(sorted_paths[: self.cfg.beam_width])

            if self.debug and step_log_data:
                step_log_data.sort(key=lambda x: x["score"], reverse=True)

                print("-" * 82)
                header = (
                    f"| {'ỨNG VIÊN':<12} | {'TUYẾN TỐT NHẤT':<20}"
                    f" | {'LỊCH SỬ + ĐIỂM BƯỚC NHẢY':<25}"
                    f" | {'TỔNG ĐIỂM':<12} |"
                )
                print(header)
                print("-" * 82)
                for item in step_log_data[:20]:
                    row = (
                        f"| {item['cand']:<12} | {item['path']:<20}"
                        f" | {item['calc_str']:<25}"
                        f" | {item['score']:<12.4f} |"
                    )
                    print(row)
                print("-" * 82)

        if paths:
            sorted_paths = sorted(
                paths.values(), key=lambda item: item[0], reverse=True
            )

            results = []
            for _, best_sentence, _ in sorted_paths[:top_k]:
                restored = [
                    apply_case_pattern(best_sentence[i], case_patterns[i])
                    if i < len(case_patterns) else best_sentence[i]
                    for i in range(len(best_sentence))
                ]
                results.append(" ".join(restored))

            return results

        return []
