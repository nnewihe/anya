"""scoreboard_widget.py — live in-app scoreboard preview.

Port of src/scoreboard/src/Scoreboard.tsx (a separate, standalone reference
project) from a React DOM component to a QWidget/QPainter paint routine.
Mirrors the same layout the ffmpeg burn-in draws (pipeline/scoreboard_reel/
render.py's `_scoreboard_filters`) — two rows (Player A/B), serve-indicator
dot, name, per-set game/tiebreak columns, current-game point label — so what
the user sees while tagging matches what the rendered video will show.
Reskinned to Anya's black/yellow brand instead of the reference app's
forest-green/Montserrat.
"""

from typing import Dict, Optional

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from pipeline.scoreboard_reel import display_columns
from theme import BLACK, SKY, TEXT_DIM, WHITE, YELLOW

_ROW_H = 40
_NAME_W = 130
_SET_W = 28
_POINT_W = 50
_DOT_D = 9


class ScoreboardPreview(QWidget):
    """Two-row scoreboard bug. Call `set_snapshot(snap, names)` whenever the
    tagging state changes; repaints itself.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._snap: Optional[dict] = None
        self._names: Dict[str, str] = {"a": "Player A", "b": "Player B"}
        self.setMinimumHeight(2 * _ROW_H + 1)
        self.setStyleSheet(f"background: {BLACK};")

    def set_snapshot(self, snap: Optional[dict], names: Optional[Dict[str, str]] = None):
        self._snap = snap
        if names:
            self._names = names
        self.update()

    def sizeHint(self):
        from PyQt6.QtCore import QSize
        return QSize(420, 2 * _ROW_H + 1)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(BLACK))

        if not self._snap:
            painter.setPen(QColor(TEXT_DIM))
            painter.setFont(QFont("Helvetica Neue", 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No points tagged yet")
            painter.end()
            return

        snap = self._snap
        cols = display_columns(snap)
        width = self.width()
        col_block_w = len(cols) * _SET_W

        self._paint_row(painter, 0, "A", snap, cols, width, col_block_w)
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
        painter.drawLine(0, _ROW_H, width, _ROW_H)
        self._paint_row(painter, _ROW_H, "B", snap, cols, width, col_block_w)
        painter.end()

    def _paint_row(self, painter: QPainter, y0: int, player: str, snap: dict,
                    cols, width: int, col_block_w: int):
        serving = not snap["matchOver"] and snap["server"] == player
        is_winner = snap["matchOver"] and snap["winner"] == player
        cy = y0 + _ROW_H / 2

        if is_winner:
            painter.fillRect(QRectF(0, y0, width, _ROW_H), QColor(232, 255, 61, 30))

        # Serve dot.
        dot_x = 10
        if serving:
            painter.setBrush(QColor(SKY))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(
                QRectF(dot_x, cy - _DOT_D / 2, _DOT_D, _DOT_D)
            )

        # Name.
        painter.setPen(QColor(WHITE))
        painter.setFont(QFont("Helvetica Neue", 12, QFont.Weight.Bold))
        name_x = dot_x + _DOT_D + 10
        name = self._names.get("a" if player == "A" else "b", f"Player {player}")
        painter.drawText(
            QRectF(name_x, y0, _NAME_W, _ROW_H),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            name,
        )

        # Set/game columns, right-aligned before the point box.
        point_x = width - _POINT_W
        cols_x = point_x - col_block_w
        painter.setFont(QFont("Helvetica Neue", 11, QFont.Weight.DemiBold))
        for i, c in enumerate(cols):
            cx = cols_x + i * _SET_W
            val = c["A"] if player == "A" else c["B"]
            color = QColor(WHITE) if c.get("current") else QColor(TEXT_DIM)
            painter.setPen(color)
            painter.drawText(
                QRectF(cx, y0, _SET_W, _ROW_H),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter,
                str(val),
            )

        # Current-game point box.
        painter.fillRect(QRectF(point_x, y0, _POINT_W, _ROW_H), QColor(232, 255, 61, 35))
        painter.setPen(QColor(YELLOW))
        painter.setFont(QFont("Helvetica Neue", 13, QFont.Weight.Bold))
        label = snap["pointLabels"]["A"] if player == "A" else snap["pointLabels"]["B"]
        painter.drawText(
            QRectF(point_x, y0, _POINT_W, _ROW_H),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter,
            label,
        )
