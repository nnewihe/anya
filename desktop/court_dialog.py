"""
court_dialog.py — court corner calibration as a native Qt dialog.

Replaces the OpenCV HighGUI window that ``pipeline.utilities.init_court``
opens. That window is fine from the CLI, where nothing else owns the process,
but the desktop app calls it from a Qt slot — so PyQt6's event loop sits on
the stack while OpenCV spins its own ``while True: cv2.waitKey(20)`` loop and
waits for its window to receive native mouse messages. Two GUI toolkits
contending for one thread's message queue.

On macOS that happens to work. On Windows it does not: the window paints
(drawing is a direct call) and then ignores every click (events need
dispatch), which left calibration — the very first thing a tester does —
dead in the water on the Windows build.

Rather than hunt for the OpenCV incantation that survives both platforms,
this does the picking in the toolkit that already owns the event loop. The
whole class of problem goes away, and focus, z-order, modality, HiDPI and
theming come out right for free instead of being fought for.

``init_court`` is deliberately left alone: it is still what the pipeline CLI
and stage 0 use. Both read and write the same cache file through
``load_court_cache``/``save_court_cache``, so whichever picker ran, stage 0
finds the corners and opens nothing.
"""

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from pipeline.utilities import (
    COURT_CORNER_ORDER, COURT_CORNER_TAGS, get_reference_frame,
    load_court_cache, save_court_cache,
)
from theme import BLACK, TEXT_DIM, WHITE, YELLOW, ghost_btn_css, primary_btn_css

# Plain-language corner names. COURT_CORNER_ORDER is written for the
# homography code ("bottom-left" is a vertex index, not a place on screen);
# a tester needs to know where to actually click.
_PROMPTS = [
    "Click the NEAR-LEFT corner  (baseline closest to you, left side)",
    "Click the NEAR-RIGHT corner  (baseline closest to you, right side)",
    "Click the FAR-RIGHT corner  (baseline furthest away, right side)",
    "Click the FAR-LEFT corner  (baseline furthest away, left side)",
]


class _FrameCanvas(QLabel):
    """The reference frame, with the clicked corners drawn over it.

    Clicks are recorded in IMAGE coordinates, not widget coordinates. The
    frame is scaled to fit the dialog, so the two differ by the scale factor —
    storing widget coordinates would silently hand the homography corners that
    are off by however much the window happened to be resized, which is the
    kind of bug that produces a plausible-looking but wrong reel.
    """

    def __init__(self, bgr, on_change):
        super().__init__()
        self._on_change = on_change
        self.points = []

        h, w = bgr.shape[:2]
        self._img_w, self._img_h = w, h
        # cv2 is BGR and QImage wants RGB. `.copy()` because QImage does not
        # take ownership of the buffer: without it the underlying numpy array
        # can be freed while Qt is still painting from it.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._base = QPixmap.fromImage(
            QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        )
        self.setPixmap(self._base)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumSize(1, 1)
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # ── coordinate mapping ────────────────────────────────────────────────

    def _geometry(self):
        """Scale factor and letterbox offset of the drawn image inside us."""
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return 1.0, 0, 0
        scale = min(self.width() / self._img_w, self.height() / self._img_h)
        drawn_w, drawn_h = self._img_w * scale, self._img_h * scale
        return scale, (self.width() - drawn_w) / 2, (self.height() - drawn_h) / 2

    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if len(self.points) >= len(COURT_CORNER_ORDER):
            return
        scale, off_x, off_y = self._geometry()
        if scale <= 0:
            return
        pos = ev.position()
        x = (pos.x() - off_x) / scale
        y = (pos.y() - off_y) / scale
        # Clicks in the letterbox margin aren't on the court at all.
        if not (0 <= x < self._img_w and 0 <= y < self._img_h):
            return
        self.points.append((x, y))
        self._repaint()
        self._on_change()

    # ── painting ──────────────────────────────────────────────────────────

    def reset(self):
        self.points.clear()
        self._repaint()
        self._on_change()

    def undo(self):
        if self.points:
            self.points.pop()
            self._repaint()
            self._on_change()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._repaint()

    def _repaint(self):
        scale = min(self.width() / self._img_w, self.height() / self._img_h) if self.width() else 1.0
        scaled = self._base.scaled(
            max(1, int(self._img_w * scale)), max(1, int(self._img_h * scale)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter = QPainter(scaled)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        yellow = QColor(YELLOW)

        # The quadrilateral so far, so a mis-click is obvious before you commit.
        if len(self.points) > 1:
            pen = QPen(yellow, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            for a, b in zip(self.points, self.points[1:]):
                painter.drawLine(
                    QPoint(int(a[0] * scale), int(a[1] * scale)),
                    QPoint(int(b[0] * scale), int(b[1] * scale)),
                )
            if len(self.points) == len(COURT_CORNER_ORDER):
                painter.drawLine(
                    QPoint(int(self.points[-1][0] * scale), int(self.points[-1][1] * scale)),
                    QPoint(int(self.points[0][0] * scale), int(self.points[0][1] * scale)),
                )

        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        painter.setFont(font)
        for i, (x, y) in enumerate(self.points):
            px, py = int(x * scale), int(y * scale)
            painter.setPen(QPen(QColor(0, 0, 0, 180), 4))
            painter.drawEllipse(QPoint(px, py), 7, 7)
            painter.setPen(QPen(yellow, 2))
            painter.setBrush(yellow)
            painter.drawEllipse(QPoint(px, py), 5, 5)
            # Outlined text so a label stays readable over pale court lines.
            painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
            painter.drawText(px + 11, py - 9, COURT_CORNER_TAGS[i])
            painter.setPen(QPen(yellow, 1))
            painter.drawText(px + 10, py - 10, COURT_CORNER_TAGS[i])
        painter.end()
        self.setPixmap(scaled)


class CourtCornerDialog(QDialog):
    """Modal four-corner picker over a reference frame from the video."""

    def __init__(self, parent, bgr):
        super().__init__(parent)
        self.setWindowTitle("Court calibration")
        self.setModal(True)
        self.setStyleSheet(f"QDialog {{ background: {BLACK}; }}")

        self._canvas = _FrameCanvas(bgr, self._sync)

        self._prompt = QLabel()
        self._prompt.setStyleSheet(f"color: {YELLOW}; font-size: 15px; font-weight: 700;")
        self._hint = QLabel(
            "Click the four corners of the court in the order prompted. "
            "Backspace undoes the last point."
        )
        self._hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")

        self._undo_btn = QPushButton("UNDO")
        self._undo_btn.setStyleSheet(ghost_btn_css())
        self._undo_btn.clicked.connect(self._canvas.undo)

        self._reset_btn = QPushButton("RESET")
        self._reset_btn.setStyleSheet(ghost_btn_css())
        self._reset_btn.clicked.connect(self._canvas.reset)

        self._cancel_btn = QPushButton("CANCEL")
        self._cancel_btn.setStyleSheet(ghost_btn_css())
        self._cancel_btn.clicked.connect(self.reject)

        self._ok_btn = QPushButton("USE THESE CORNERS")
        self._ok_btn.clicked.connect(self.accept)

        buttons = QHBoxLayout()
        buttons.addWidget(self._undo_btn)
        buttons.addWidget(self._reset_btn)
        buttons.addStretch()
        buttons.addWidget(self._cancel_btn)
        buttons.addWidget(self._ok_btn)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(10)
        lay.addWidget(self._prompt)
        lay.addWidget(self._hint)
        lay.addWidget(self._canvas, 1)
        lay.addLayout(buttons)

        self.resize(1100, 760)
        self._sync()

    def _sync(self):
        n = len(self._canvas.points)
        total = len(COURT_CORNER_ORDER)
        done = n >= total
        self._prompt.setText(
            f"All four corners placed — check the outline, then confirm."
            if done else f"{n + 1} of {total}:  {_PROMPTS[n]}"
        )
        self._ok_btn.setEnabled(done)
        self._ok_btn.setStyleSheet(primary_btn_css(enabled=done))
        self._undo_btn.setEnabled(n > 0)
        self._reset_btn.setEnabled(n > 0)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Backspace:
            self._canvas.undo()
            return
        # Enter must not accept a half-finished quad via QDialog's default
        # button handling.
        if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if len(self._canvas.points) == len(COURT_CORNER_ORDER):
                self.accept()
            return
        super().keyPressEvent(ev)

    def points(self):
        return list(self._canvas.points)


def pick_court_corners(parent, video_path: str, target_idx: int = 300,
                       analysis_size: tuple = None):
    """``init_court``'s contract, in Qt: return ``(points, frame_shape)``.

    Returns the cached corners without showing anything when they exist, so
    repeat runs on the same video behave exactly as before. Raises
    RuntimeError if the tester cancels — the caller already treats that as
    "calibration cancelled" and aborts the job.
    """
    cached = load_court_cache(video_path, analysis_size)
    if cached is not None:
        return cached

    frame = get_reference_frame(video_path, target_idx=target_idx)
    if analysis_size is not None:
        frame = cv2.resize(frame, analysis_size, interpolation=cv2.INTER_AREA)

    dlg = CourtCornerDialog(parent, frame)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        raise RuntimeError("Court calibration cancelled.")

    pts = [(float(x), float(y)) for x, y in dlg.points()]
    shape = frame.shape
    save_court_cache(video_path, pts, shape, analysis_size)
    return pts, shape
