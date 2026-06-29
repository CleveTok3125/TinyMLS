import json
import math
import os
import unicodedata
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Set, Tuple

import marisa_trie
from rapidfuzz import fuzz, process

from config import SpellCheckerConfig
from keyboard import get_keyboard_coordinates, keyboard_matrix
from model_pkg import ModelArchive
from telex import to_standard_telex


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

        if os.path.isdir(self.cfg.stats_path):
            self._load_from_dir()
        elif self.cfg.stats_path.endswith(".tinymls") or self.cfg.stats_path.endswith(".zip"):
            self._load_from_archive()
        else:
            raise FileNotFoundError(f"Không tìm thấy model: {self.cfg.stats_path}")

    def _resolve_dict_path(self) -> str | None:
        if self.cfg.dict_path and os.path.exists(self.cfg.dict_path):
            return self.cfg.dict_path
        return None

    def _load_standard_dict(self, dict_path: str | None) -> None:
        self.standard_dict: Set[str] = set()
        if not dict_path:
            return
        try:
            with open(dict_path, "r", encoding="utf-8") as f:
                self.standard_dict = {
                    line.strip().lower() for line in f if line.strip()
                }
            print(f"Đã nạp {len(self.standard_dict)} từ chuẩn từ '{dict_path}'.")
        except FileNotFoundError:
            pass

    def _load_from_dir(self) -> None:
        stats_dir = self.cfg.stats_path
        print(f"Loading dữ liệu thống kê từ thư mục: {stats_dir}...")

        with open(os.path.join(stats_dir, "language_stats_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)

        vocab_list = [unicodedata.normalize("NFC", w) for w in meta["vocab"]]
        self.total_unigrams = meta["total_unigrams"]

        self.unigrams = marisa_trie.RecordTrie("<I").mmap(
            os.path.join(stats_dir, "unigrams.trie")
        )
        self.bigrams = marisa_trie.RecordTrie("<I").mmap(
            os.path.join(stats_dir, "bigrams.trie")
        )
        tri_path = os.path.join(stats_dir, "trigrams.trie")
        self.trigrams = marisa_trie.RecordTrie("<I").mmap(tri_path) if os.path.exists(tri_path) else None

        self._init_from_vocab(vocab_list)
        self._load_standard_dict(self._resolve_dict_path())

    def _load_from_archive(self) -> None:
        self._archive = ModelArchive(self.cfg.stats_path)
        print(f"Loading dữ liệu thống kê từ package: {self.cfg.stats_path}...")

        meta = self._archive.read_json("language_stats_meta.json")
        vocab_list = [unicodedata.normalize("NFC", w) for w in meta["vocab"]]
        self.total_unigrams = meta["total_unigrams"]

        self.unigrams = self._archive.mmap_trie("unigrams.trie")
        self.bigrams = self._archive.mmap_trie("bigrams.trie")
        self.trigrams = self._archive.mmap_trie("trigrams.trie") if self._archive.has("trigrams.trie") else None

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
        vocab: Set[str] = set(vocab_list)

        self.telex_to_vocab: Dict[str, List[str]] = {}
        for w in vocab_list:
            t = to_standard_telex(w)
            if t not in self.telex_to_vocab:
                self.telex_to_vocab[t] = []
            self.telex_to_vocab[t].append(w)

        telex_vocab_list: list[str] = list(self.telex_to_vocab.keys())
        self.telex_by_length = defaultdict(list)
        for t in telex_vocab_list:
            self.telex_by_length[len(t)].append(t)

        sorted_unigrams = sorted(
            self.unigrams.items(), key=lambda item: item[1], reverse=True
        )
        self.unigram_rankings: Dict[str, int] = {
            word: rank for rank, (word, _) in enumerate(sorted_unigrams, start=1)
        }
        self.total_ranked_unigrams = len(sorted_unigrams)
        top_k = getattr(self.cfg, "auto_ambiguous_top_k", 50)
        self.dynamic_ambiguous_words: Set[str] = {
            word for word, _ in sorted_unigrams[:top_k]
        }
        if self.debug:
            print(
                f"Tự động cấm Anchor {len(self.dynamic_ambiguous_words)} từ phổ biến: {list(self.dynamic_ambiguous_words)[:10]}..."
            )

        self.kb_coords = get_keyboard_coordinates(keyboard_matrix=keyboard_matrix)
        print(f"Done! The dictionary has {len(vocab)} words.")

    def get_trie_count(self, trie, key: str) -> int:
        normalized_key = unicodedata.normalize("NFC", key)
        res = trie.get(normalized_key)
        return res[0][0] if res else 0

    def is_valid_length(self, cand_telex: str, error_len: int) -> bool:
        cand_len = len(cand_telex)
        if error_len >= 3 and cand_len < 3:
            return False
        if abs(cand_len - error_len) > 3:
            return False
        return True

    def get_fast_close_matches(
        self, target: str, possibilities: List[str], n: int, cutoff: float
    ) -> List[str]:
        results = process.extract(
            target, possibilities, scorer=fuzz.ratio, limit=n, score_cutoff=cutoff * 100
        )

        return [r[0] for r in results]

    def get_candidates(
        self, error_word: str, prev_word: str | None = None
    ) -> List[str]:
        candidates: List[str] = []

        error_telex = to_standard_telex(error_word)
        error_len = len(error_telex)

        if prev_word:
            prefix = f"{prev_word} "
            context_words: List[str] = [
                key[len(prefix) :] for key in self.bigrams.keys(prefix)
            ]

            if context_words:
                context_telex_to_word: Dict[str, List[str]] = {}
                for cw in context_words:
                    ct = to_standard_telex(cw)
                    if ct not in context_telex_to_word:
                        context_telex_to_word[ct] = []
                    context_telex_to_word[ct].append(cw)

                context_telex_list = [
                    t
                    for t in context_telex_to_word.keys()
                    if self.is_valid_length(t, error_len)
                ]

                context_telex_matches: List[str] = self.get_fast_close_matches(
                    error_telex,
                    context_telex_list,
                    n=self.cfg.top_n,
                    cutoff=self.cfg.cutoff,
                )

                for ctm in context_telex_matches:
                    for real_word in context_telex_to_word[ctm]:
                        real_word = unicodedata.normalize("NFC", real_word)
                        if real_word not in candidates:
                            candidates.append(real_word)

        if len(candidates) < self.cfg.top_n:
            filtered_global_telex = []
            min_len = 3 if error_len >= 3 else max(1, error_len - 3)
            max_len = error_len + 3

            min_len = max(min_len, error_len - 3)

            for length in range(min_len, max_len + 1):
                if length in self.telex_by_length:
                    filtered_global_telex.extend(self.telex_by_length[length])

            general_telex_matches: List[str] = self.get_fast_close_matches(
                error_telex,
                filtered_global_telex,
                n=self.cfg.top_n,
                cutoff=self.cfg.cutoff,
            )
            for gtm in general_telex_matches:
                for real_word in self.telex_to_vocab[gtm]:
                    real_word = unicodedata.normalize("NFC", real_word)
                    if real_word not in candidates:
                        candidates.append(real_word)

        return candidates[: self.cfg.top_n]

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

    def calculate_context_prob(self, w1: str | None, w2: str, w3: str) -> float:
        p_tri = 0.0
        if w1:
            bigram_w1_w2_count = self.get_trie_count(self.bigrams, f"{w1} {w2}")
            trigram_count = (
                self.get_trie_count(self.trigrams, f"{w1} {w2} {w3}")
                if self.trigrams
                else 0
            )
            p_tri = (
                trigram_count / bigram_w1_w2_count if bigram_w1_w2_count > 0 else 0.0
            )

        unigram_w2_count = self.get_trie_count(self.unigrams, w2)
        bigram_w2_w3_count = self.get_trie_count(self.bigrams, f"{w2} {w3}")
        p_bi = bigram_w2_w3_count / unigram_w2_count if unigram_w2_count > 0 else 0.0

        p_uni = (
            self.get_trie_count(self.unigrams, w3) / self.total_unigrams
            if self.total_unigrams > 0
            else 0.0
        )

        l3 = getattr(self.cfg, "lambda_3", 0.0)
        l2 = getattr(self.cfg, "lambda_2", 0.0)
        l1 = getattr(self.cfg, "lambda_1", 0.0)

        if not w1:
            total_l = l2 + l1
            return (l2 * p_bi + l1 * p_uni) / total_l if total_l > 0 else 0.0

        return (l3 * p_tri) + (l2 * p_bi) + (l1 * p_uni)

    def calculate_score(
        self,
        candidate: str,
        error_word: str,
        prev_word: str | None,
        prev_prev_word: str | None = None,
    ) -> float:
        candidate = unicodedata.normalize("NFC", candidate)
        error_word = unicodedata.normalize("NFC", error_word)

        cand_telex = to_standard_telex(candidate)
        err_telex = to_standard_telex(error_word)

        eps = 1e-8

        sim = self.keyboard_aware_similarity(err_telex, cand_telex)
        sim_feat = math.log(sim + eps)

        count = self.get_trie_count(self.unigrams, candidate)
        p_uni = count / self.total_unigrams if self.total_unigrams > 0 else 0.0

        if prev_word:
            p_ctx = self.calculate_context_prob(prev_prev_word, prev_word, candidate)

            if candidate == prev_word:
                p_ctx *= getattr(self.cfg, "stutter_penalty", 0.0)
        else:
            p_ctx = max(p_uni, eps)

        ctx_feat = math.log(p_ctx + eps)
        ctx_feat = (ctx_feat + 10) / 10

        w_sim = getattr(self.cfg, "sim_weight", 0.0)
        w_ctx = getattr(self.cfg, "context_weight", 0.0)

        score = (w_sim * sim_feat) + (w_ctx * ctx_feat)

        score += self.calculate_exact_match_bonus(candidate, error_word)

        if self.debug and self.detail_log:
            prev_str = prev_word if prev_word else "[START]"
            print("ERROR:", err_telex)
            print("CAND :", cand_telex)
            print(f"      ➜ '{prev_str}' -> '{candidate}' (error: '{error_word}')")
            print(f"         sim : {sim_feat:.4f} * {w_sim}")
            print(f"         ctx : {ctx_feat:.4f} * {w_ctx}")
            print(f"         => SCORE: {score:.4f}")

        return score

    def calculate_exact_match_bonus(self, candidate: str, error_word: str) -> float:
        if candidate != error_word:
            return 0.0

        rank = self.unigram_rankings.get(candidate)
        if rank is None:
            return 0.0

        bonus_range = getattr(self.cfg, "exact_match_bonus", [0.0, 0.0])
        if not isinstance(bonus_range, list) or len(bonus_range) != 2:
            return 0.0

        min_bonus, max_bonus = bonus_range
        if self.total_ranked_unigrams <= 1:
            return float(max_bonus)

        rank_ratio = (rank - 1) / (self.total_ranked_unigrams - 1)
        return float(max_bonus - ((max_bonus - min_bonus) * rank_ratio))

    def is_delayed_anchor(self, w: str, next_w: str | None) -> bool:
        if not hasattr(self, "standard_dict") or w not in self.standard_dict:
            return False
        if len(w) < 2 or w in getattr(self, "dynamic_ambiguous_words", set()):
            return False
        if next_w is None:
            return True

        if f"{w} {next_w}" in self.bigrams:
            return True
        return False

    def correct_sentence(self, sentence: str, top_k: int = 5) -> List[str]:
        original_words: List[str] = sentence.split()
        case_patterns = [detect_case_pattern(w) for w in original_words]
        words: List[str] = [w.lower() for w in original_words]
        if not words:
            return []

        paths: Dict[str, Tuple[float, List[str], str]] = {}

        reset_context_next_step = True

        for i, current_word in enumerate(words):
            new_paths: Dict[str, Tuple[float, List[str], str]] = {}
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
                        for c in self.get_candidates(current_word, prev_word=prev_cand):
                            candidates_set[c] = None

                for c in self.get_candidates(current_word, prev_word=None):
                    candidates_set[c] = None
                candidates = list(candidates_set.keys())

                if candidates:
                    best_match = process.extractOne(
                        current_word,
                        candidates,
                        scorer=fuzz.ratio,
                        score_cutoff=self.cfg.cutoff * 100,
                    )

                    if best_match is None:
                        candidates = [current_word]
                        is_garbage = True
                        if self.debug:
                            print(
                                f"  ➜ Bỏ cuộc: Rác/Từ lạ (Không có từ nào >= {self.cfg.cutoff})."
                            )
                else:
                    candidates = [current_word]

            step_log_data: List[Dict] = []

            for curr_cand in candidates:
                if reset_context_next_step or not paths:
                    step_score = self.calculate_score(
                        curr_cand,
                        current_word,
                        prev_word=None,
                        prev_prev_word=None,
                    )

                    best_history = []
                    if paths:
                        best_past_key = max(paths.keys(), key=lambda k: paths[k][0])
                        best_history = paths[best_past_key][1]

                    total_score = step_score
                    new_path = best_history + [curr_cand]

                    new_path_key = " ".join(new_path)
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
                    for prev_score, prev_path, prev_cand in paths.values():
                        prev_prev_cand = prev_path[-2] if len(prev_path) >= 2 else None

                        step_score = self.calculate_score(
                            curr_cand, current_word, prev_cand, prev_prev_cand
                        )

                        total_score = prev_score + step_score

                        new_path = prev_path + [curr_cand]
                        new_path_key = " ".join(new_path)
                        new_paths[new_path_key] = (total_score, new_path, curr_cand)

                        if self.debug:
                            step_log_data.append(
                                {
                                    "cand": curr_cand,
                                    "path": f"{prev_cand} -> {curr_cand}",
                                    "calc_str": f"({prev_score:.4f} + {step_score:.4f})",
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
                print(
                    f"| {'ỨNG VIÊN':<12} | {'TUYẾN TỐT NHẤT':<20} | {'LỊCH SỬ + ĐIỂM BƯỚC NHẢY':<25} | {'TỔNG ĐIỂM':<12} |"
                )
                print("-" * 82)
                for item in step_log_data[:20]:
                    print(
                        f"| {item['cand']:<12} | {item['path']:<20} | {item['calc_str']:<25} | {item['score']:<12.4f} |"
                    )
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
