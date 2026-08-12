"""Tennis scoring engine — faithful port of
src/scoreboard/shared/scoring.mjs (a separate, standalone project this one
project's App.tsx/Scoreboard.tsx/render.mjs are ported from).

Given an ordered list of point winners ("A" | "B") and a MatchFormat, replays
the match and produces a score snapshot *before* each point, plus the final
state. Handles: 0/15/30/40, deuce/advantage, no-ad, games to N (win by 2),
tiebreaks (regular/super), best-of-3/5, and configurable final-set behavior.

Pure Python, no Qt/ffmpeg imports — this is the single scoring authority used
by both the desktop live preview (scoreboard_widget.ScoreboardPreview) and
the ffmpeg burn-in (render.py), exactly as scoring.mjs is shared by
Scoreboard.tsx and render.mjs in the reference app.

A snapshot is a plain dict (not a dataclass) so it serializes/consumes the
same way the reference app's plain JS objects do:

    {
      "setsWon": {"A": int, "B": int},
      "completedSets": [{"A": int, "B": int, "tb": {"A": int, "B": int} | None}, ...],
      "games": {"A": int, "B": int},
      "points": {"A": int, "B": int},
      "inTiebreak": bool,
      "server": "A" | "B",
      "matchOver": bool,
      "winner": "A" | "B" | None,
      "pointLabels": {"A": str, "B": str},
    }
"""

from typing import Dict, List, Optional, Sequence

from .config import MatchFormat, default_format

POINT_LABELS = ["0", "15", "30", "40"]

Snapshot = Dict
Player = str  # "A" | "B"


class _MatchState:
    """Internal mutable replay state — mirrors scoring.mjs's closure-local
    `state` object. Not exposed outside this module; callers only see the
    plain-dict snapshots `replay_match` returns.
    """

    def __init__(self, fmt: MatchFormat):
        self.f = fmt
        self.sets_needed = 3 if fmt.best_of >= 5 else 2
        self.sets_won = {"A": 0, "B": 0}
        self.completed_sets: List[dict] = []  # {A, B, tb?: {A,B}}
        self.games = {"A": 0, "B": 0}
        self.points = {"A": 0, "B": 0}
        self.in_tiebreak = False
        self.server: Player = "B" if fmt.first_server == "B" else "A"
        self.match_over = False
        self.winner: Optional[Player] = None

    def is_final_set(self) -> bool:
        return len(self.completed_sets) == self.f.best_of - 1

    def switch_server(self):
        self.server = "B" if self.server == "A" else "A"

    def game_point_labels(self) -> Dict[str, str]:
        if self.in_tiebreak:
            return {"A": str(self.points["A"]), "B": str(self.points["B"])}
        a, b = self.points["A"], self.points["B"]
        if not self.f.no_ad and a >= 3 and b >= 3:
            if a == b:
                return {"A": "40", "B": "40"}  # deuce
            return {"A": "AD", "B": "40"} if a > b else {"A": "40", "B": "AD"}
        return {
            "A": POINT_LABELS[min(a, 3)],
            "B": POINT_LABELS[min(b, 3)],
        }

    def snapshot(self) -> Snapshot:
        return {
            "setsWon": dict(self.sets_won),
            "completedSets": [dict(s) for s in self.completed_sets],
            "games": dict(self.games),
            "points": dict(self.points),
            "inTiebreak": self.in_tiebreak,
            "server": self.server,
            "matchOver": self.match_over,
            "winner": self.winner,
            "pointLabels": self.game_point_labels(),
        }

    def win_set(self, w: Player, tb_points: Optional[Dict[str, int]]):
        super_set = self.is_final_set() and self.f.final_set == "super"
        if super_set and tb_points:
            entry = {"A": tb_points["A"], "B": tb_points["B"]}
        else:
            entry = {"A": self.games["A"], "B": self.games["B"], "tb": tb_points or None}
        self.completed_sets.append(entry)
        self.sets_won[w] += 1
        self.games = {"A": 0, "B": 0}
        self.points = {"A": 0, "B": 0}
        self.in_tiebreak = False
        self.switch_server()
        if self.sets_won[w] >= self.sets_needed:
            self.match_over = True
            self.winner = w
            return
        # A super-tiebreak final set starts immediately as a tiebreak.
        if self.is_final_set() and self.f.final_set == "super":
            self.in_tiebreak = True

    def win_game(self, w: Player, was_tiebreak: bool = False):
        tb_points = dict(self.points) if was_tiebreak else None
        self.games[w] += 1
        self.points = {"A": 0, "B": 0}
        if was_tiebreak:
            self.in_tiebreak = False

        g = self.games
        set_to = self.f.set_to if self.f.set_to >= 1 else 6
        set_won = was_tiebreak
        if not set_won and (g["A"] >= set_to or g["B"] >= set_to) and abs(g["A"] - g["B"]) >= 2:
            set_won = True
        if not set_won and g["A"] == set_to and g["B"] == set_to:
            advantage_set = self.is_final_set() and self.f.final_set == "advantage"
            if not advantage_set:
                self.in_tiebreak = True  # start tiebreak at setTo-setTo

        if set_won:
            self.win_set(w, tb_points)
        else:
            self.switch_server()

    def award_point(self, w: Player):
        self.points[w] += 1
        a, b = self.points["A"], self.points["B"]

        if self.in_tiebreak:
            to = self.f.super_to if (self.is_final_set() and self.f.final_set == "super") else self.f.tiebreak_to
            if (a >= to or b >= to) and abs(a - b) >= 2:
                self.win_game(w, was_tiebreak=True)
            return

        if self.f.no_ad:
            # Game to 4 points; at 3-3 the next point decides.
            if a >= 4 and a > b:
                self.win_game("A")
            elif b >= 4 and b > a:
                self.win_game("B")
        else:
            if a >= 4 and a - b >= 2:
                self.win_game("A")
            elif b >= 4 and b - a >= 2:
                self.win_game("B")


class ReplayResult:
    def __init__(self, snapshots: List[Snapshot], final_state: Snapshot, fmt: MatchFormat):
        self.snapshots = snapshots
        self.final_state = final_state
        self.format = fmt


def replay_match(winners: Sequence[str], fmt: Optional[MatchFormat] = None) -> ReplayResult:
    """Replay a full sequence of point winners.

    `winners` is an ordered list of "A"/"B" (any other value is treated as
    "A", matching scoring.mjs's `raw === 'B' ? 'B' : 'A'` coercion).
    """
    f = fmt or default_format()
    state = _MatchState(f)

    snapshots: List[Snapshot] = []
    for raw in winners:
        w = "B" if raw == "B" else "A"
        if state.match_over:
            break
        snapshots.append(state.snapshot())  # score standing as the point begins
        state.award_point(w)

    return ReplayResult(snapshots, state.snapshot(), f)


def display_columns(snap: Snapshot) -> List[dict]:
    """One column per completed set, plus the in-progress set (unless the
    match is over). Each column: {A, B, tb?, current?}.
    """
    cols = [{"A": s["A"], "B": s["B"], "tb": s.get("tb")} for s in snap["completedSets"]]
    if not snap["matchOver"]:
        cols.append({"A": snap["games"]["A"], "B": snap["games"]["B"], "current": True})
    return cols


def describe_score(snap: Snapshot, names: Optional[Dict[str, str]] = None) -> str:
    """A short human-readable description of the standing in a snapshot."""
    names = names or {"a": "A", "b": "B"}
    sets = ", ".join(f"{s['A']}-{s['B']}" for s in snap["completedSets"])
    if snap["matchOver"]:
        winner_name = names["a"] if snap["winner"] == "A" else names["b"]
        return f"{winner_name} wins ({sets})"
    cur = f" | games {snap['games']['A']}-{snap['games']['B']}"
    pts = f" | {snap['pointLabels']['A']}-{snap['pointLabels']['B']}"
    sets_part = f" [{sets}]" if sets else ""
    return f"Sets {snap['setsWon']['A']}-{snap['setsWon']['B']}{sets_part}{cur}{pts}"
