"""Match format for the tennis scoring engine.

Port of src/scoreboard/shared/scoring.mjs's ``defaultFormat()`` — kept as a
plain dataclass (mirrors pipeline/rally_reel/config.py's ReelConfig) so it
serializes trivially to/from the tags.json ``format`` object.
"""

from dataclasses import dataclass, asdict
from typing import Literal

Player = Literal["A", "B"]
FinalSet = Literal["tiebreak", "advantage", "super"]


@dataclass
class MatchFormat:
    best_of: int = 3            # 3 or 5
    set_to: int = 6             # games to win a set (win by 2); 4 = short/Fast4 set
    no_ad: bool = False         # sudden-death deuce
    final_set: FinalSet = "tiebreak"
    tiebreak_to: int = 7
    super_to: int = 10
    first_server: Player = "A"

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MatchFormat":
        """Accepts either this dataclass's snake_case fields or the
        reference app's camelCase tags.json fields (``bestOf``, ``setTo``,
        ``noAd``, ``finalSet``, ``tiebreakTo``, ``superTo``, ``firstServer``),
        so tags.json files exported by src/scoreboard load unchanged.
        """
        if not d:
            return cls()
        alias = {
            "bestOf": "best_of", "setTo": "set_to", "noAd": "no_ad",
            "finalSet": "final_set", "tiebreakTo": "tiebreak_to",
            "superTo": "super_to", "firstServer": "first_server",
        }
        norm = {alias.get(k, k): v for k, v in d.items()}
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in norm.items() if k in fields})


def default_format() -> MatchFormat:
    return MatchFormat()
