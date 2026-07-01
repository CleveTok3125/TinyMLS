import dataclasses
import json
import os
from dataclasses import dataclass, field


@dataclass
class SpellCheckerConfig:
    stats_path: str = "trained_model"
    dict_path: str | None = None

    top_n: int = 25

    cutoff: float = 0.4

    sim_weight: int = 5

    context_weight: float = 1.0

    max_kb_distance: float = 4.0

    unknown_char_penalty: float = 1.0

    transposition_cost: float = 1.0

    stutter_penalty: float = 0.01

    exact_match_bonus: list[float] = field(default_factory=lambda: [0.5, 1.5])

    auto_ambiguous_top_k: int = 20

    beam_width: int = 5

    lambda_3: float = 0.6
    lambda_2: float = 0.3
    lambda_1: float = 0.1

    max_personal_memory_size: int = 10000
    priority_score: float = 5.0
    boost_factor: float = 2.0
    personalization_dir: str = "data/personalization"

    @classmethod
    def from_json(cls, json_path: str) -> "SpellCheckerConfig":
        if os.path.exists(json_path):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)

                valid_keys = {f.name for f in dataclasses.fields(cls)}
                filtered_data = {k: v for k, v in data.items() if k in valid_keys}

                return cls(**filtered_data)
            except Exception as e:
                msg = (
                    f"Lỗi khi đọc file config {json_path}: {e}."
                    " Đang dùng config mặc định."
                )
                print(msg)

        return cls()
