import json
import os
import re
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Iterator, List, Set

import marisa_trie

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

VOWELS = "aeiouyàáãạảăắằẵặẳâấầẫậẩèéẽẹẻêếềễệểìíĩịỉòóõọỏôốồỗộổơớờỡợởùúũụủưứừữựửỳýỹỵỷ"

TONED_VOWELS = "àáãạảắằẵặẳấầẫậẩèéẽẹẻếềễệểìíĩịỉòóõọỏốồỗộổớờỡợởùúũụủứừữựửỳýỹỵỷ"

INITIALS = r"(ch|gh|gi|kh|ngh|ng|nh|ph|qu|th|tr|b|c|d|đ|g|h|k|l|m|n|p|r|s|t|v|x)"

FINALS = r"(ch|ng|nh|c|m|n|p|t)"

SYLLABLE_PATTERN = re.compile(rf"^{INITIALS}?([{VOWELS}]+){FINALS}?$")


def _get_toned_variations(base_char: str) -> str:
    variations = {
        "a": "àáảãạ",
        "ă": "ằắẳẵặ",
        "â": "ầấẩẫậ",
        "e": "èéẻẽẹ",
        "ê": "ềếểễệ",
        "i": "ìíỉĩị",
        "o": "òóỏõọ",
        "ô": "ồốổỗộ",
        "ơ": "ờớởỡợ",
        "u": "ùúủũụ",
        "ư": "ừứửữự",
        "y": "ỳýỷỹỵ",
    }
    return variations.get(base_char, "")


def is_valid_vietnamese_word(word: str) -> bool:
    word = unicodedata.normalize("NFC", word)

    if not (1 <= len(word) <= 7):
        return False

    tone_count = sum(1 for char in word if char in TONED_VOWELS)
    if tone_count > 1:
        return False

    match = SYLLABLE_PATTERN.match(word)
    if not match:
        return False

    initial = match.group(1) or ""
    vowel_part = match.group(2)

    if len(vowel_part) > 3:
        return False

    front_vowel_base = "eêiy"

    first_v_char = vowel_part[0]
    is_front_vowel = any(
        first_v_char in _get_toned_variations(base) for base in front_vowel_base
    )

    if initial in ["gh", "ngh", "k"] and not is_front_vowel:
        return False

    if initial in ["g", "ng", "c"] and is_front_vowel:
        return False

    return True


def is_valid_word(word: str) -> bool:
    word = unicodedata.normalize("NFC", word)
    return word.isalpha() and 1 <= len(word) <= 15


def extract_valid_sequences(raw_text: str) -> List[List[str]]:
    text = raw_text.lower()

    text = re.sub(r'[.,!?;:()\[\]{}""\'\n\r\t\-]', " | ", text)

    raw_words = text.split()

    sequences: List[List[str]] = []
    current_seq: List[str] = []

    for w in raw_words:
        if w == "|":
            if current_seq:
                sequences.append(current_seq)
                current_seq = []
            continue

        if is_valid_word(w):
            current_seq.append(w)
        else:
            if current_seq:
                sequences.append(current_seq)
                current_seq = []

    if current_seq:
        sequences.append(current_seq)

    return sequences


def iter_valid_sequences(raw_text: str) -> Iterator[List[str]]:
    text = raw_text.lower()
    text = re.sub(r'[.,!?;:()\[\]{}""\'\n\r\t\-]', " | ", text)

    current_seq: List[str] = []
    for w in text.split():
        if w == "|":
            if current_seq:
                yield current_seq
                current_seq = []
            continue

        if is_valid_word(w):
            current_seq.append(w)
        elif current_seq:
            yield current_seq
            current_seq = []

    if current_seq:
        yield current_seq


def _update_ngram_counts_from_sequences(
    sequences: Iterable[List[str]],
    unigram_counts: Counter[str],
    bigram_counts: Counter[str],
    trigram_counts: Counter[str],
    vocab_set: Set[str],
) -> int:
    sequence_count = 0

    for seq in sequences:
        sequence_count += 1
        len_seq = len(seq)

        unigram_counts.update(seq)
        vocab_set.update(seq)

        if len_seq >= 2:
            bigram_counts.update(f"{w1} {w2}" for w1, w2 in zip(seq, seq[1:]))

        if len_seq >= 3:
            trigram_counts.update(
                f"{w1} {w2} {w3}" for w1, w2, w3 in zip(seq, seq[1:], seq[2:])
            )

    return sequence_count


def _merge_partial_stats(
    unigram_counts: Counter[str],
    bigram_counts: Counter[str],
    trigram_counts: Counter[str],
    vocab_set: Set[str],
    partial_stats: tuple[Counter[str], Counter[str], Counter[str], Set[str], int],
) -> int:
    (
        partial_unigrams,
        partial_bigrams,
        partial_trigrams,
        partial_vocab,
        partial_sequence_count,
    ) = partial_stats

    unigram_counts.update(partial_unigrams)
    bigram_counts.update(partial_bigrams)
    trigram_counts.update(partial_trigrams)
    vocab_set.update(partial_vocab)

    return partial_sequence_count


def _save_language_stats(
    unigram_counts: Counter[str],
    bigram_counts: Counter[str],
    trigram_counts: Counter[str],
    vocab_set: Set[str],
    output_dir: str,
) -> None:
    print("Building Marisa Tries...")
    unigram_trie = marisa_trie.RecordTrie(
        "<I", ((k, (v,)) for k, v in unigram_counts.items())
    )
    bigram_trie = marisa_trie.RecordTrie(
        "<I", ((k, (v,)) for k, v in bigram_counts.items())
    )
    trigram_trie = marisa_trie.RecordTrie(
        "<I", ((k, (v,)) for k, v in trigram_counts.items())
    )

    os.makedirs(output_dir, exist_ok=True)

    unigram_trie.save(os.path.join(output_dir, "unigrams.trie"))
    bigram_trie.save(os.path.join(output_dir, "bigrams.trie"))
    trigram_trie.save(os.path.join(output_dir, "trigrams.trie"))

    metadata = {
        "vocab": sorted(vocab_set),
        "total_unigrams": sum(unigram_counts.values()),
    }

    with open(
        os.path.join(output_dir, "language_stats_meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"Done! Toàn bộ dữ liệu thống kê đã được lưu tại: {output_dir}/")


def _load_external_vocab(vocab_set: Set[str], external_dict_path: str | None) -> None:
    if not external_dict_path:
        return

    print(f"Đang nạp từ điển ngoài: '{external_dict_path}'...")
    try:
        with open(external_dict_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if is_valid_word(w):
                    vocab_set.add(w)
        print(f"-> Đã nạp thêm từ vựng. Tổng Vocab hiện tại: {len(vocab_set)} từ.")
    except FileNotFoundError:
        print(f"File not found: '{external_dict_path}'. Skip this step.")


def _progress(iterable, total: int | None = None, desc: str = "", unit: str = "file"):
    if tqdm is None:
        return iterable

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
    )


def _process_corpus_file(
    file_path: str,
    show_progress: bool = False,
) -> tuple[Counter[str], Counter[str], Counter[str], Set[str], int]:
    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()
    trigram_counts: Counter[str] = Counter()
    vocab_set: Set[str] = set()
    sequence_count = 0

    file_size = os.path.getsize(file_path)
    progress_bar = None
    if show_progress and tqdm is not None:
        progress_bar = tqdm(
            total=file_size,
            desc=os.path.basename(file_path),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            leave=False,
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                sequence_count += _update_ngram_counts_from_sequences(
                    iter_valid_sequences(line),
                    unigram_counts,
                    bigram_counts,
                    trigram_counts,
                    vocab_set,
                )
                if progress_bar is not None:
                    progress_bar.update(len(line.encode("utf-8")))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return (
        unigram_counts,
        bigram_counts,
        trigram_counts,
        vocab_set,
        sequence_count,
    )


def iter_corpus_files(folder_path: str, recursive: bool = False) -> Iterator[str]:
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"File not found: {folder_path}")

    entries = sorted(os.listdir(folder_path))
    for entry in entries:
        full_path = os.path.join(folder_path, entry)
        if recursive and os.path.isdir(full_path):
            yield from iter_corpus_files(full_path, recursive=True)
        elif entry.endswith(".txt"):
            yield full_path


def build_language_stats_from_folder(
    folder_path: str,
    output_dir="trained_model",
    external_dict_path: str | None = None,
    num_workers: int = 1,
    recursive: bool = False,
) -> None:
    print("Xây dựng thống kê N-gram từ corpus...")

    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()
    trigram_counts: Counter[str] = Counter()
    vocab_set: Set[str] = set()
    sequence_count = 0

    file_paths = list(iter_corpus_files(folder_path, recursive=recursive))
    if not file_paths:
        raise ValueError("Không có dữ liệu để xử lý.")

    print("Đang tách câu và phân tích ngữ cảnh...")
    max_workers = max(1, min(num_workers, len(file_paths)))

    if max_workers == 1:
        for file_path in file_paths:
            print(f"  + Processing: {os.path.basename(file_path)}")
            sequence_count += _merge_partial_stats(
                unigram_counts,
                bigram_counts,
                trigram_counts,
                vocab_set,
                _process_corpus_file(file_path, show_progress=True),
            )
    else:
        print(f"  + Parallel workers: {max_workers}")
        print("  + Progress chi tiết theo từng file chỉ hiển thị khi --workers 1")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(_process_corpus_file, file_paths)
            progress_bar = _progress(
                results, total=len(file_paths), desc="Building statistics"
            )
            for file_path, partial_stats in zip(file_paths, progress_bar):
                sequence_count += _merge_partial_stats(
                    unigram_counts,
                    bigram_counts,
                    trigram_counts,
                    vocab_set,
                    partial_stats,
                )

    if sequence_count == 0:
        raise ValueError("Không có dữ liệu hợp lệ để xây dựng thống kê.")

    print(f"-> Vocab từ Corpus: {len(vocab_set)} từ.")
    _load_external_vocab(vocab_set, external_dict_path)
    _save_language_stats(
        unigram_counts, bigram_counts, trigram_counts, vocab_set, output_dir
    )
