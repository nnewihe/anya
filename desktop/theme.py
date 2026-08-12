"""Anya Tennis brand palette (mirrors mobile/lib/theme.dart): black court,
yellow ball, sky-blue as the secondary accent.

Shared by every desktop UI module (app.py's tab shell, highlight_tab.py,
scoreboard_tab.py, scoreboard_widget.py) so the brand only needs a single
source of truth.
"""

BLACK        = "#000000"
SURFACE      = "#141412"
SURFACE_ALT  = "#1F1F1B"
YELLOW       = "#E8FF3D"
YELLOW_HOVER = "#F1FF6B"
YELLOW_PRESS = "#C8E020"
SKY          = "#49C5F1"
OUTLINE      = "#3A3A36"
TEXT_DIM     = "#A0A099"
WHITE        = "#FFFFFF"


def ghost_btn_css():
    """Shared "secondary" button style used across tabs (Browse, Open Folder,
    Undo, Export, etc.) — kept here so every tab's ghost buttons stay visually
    identical without copy-pasting the QSS.
    """
    return f"""
        QPushButton {{
            background: transparent;
            border: 1px solid rgba(255,255,255,0.28);
            border-radius: 6px;
            color: rgba(255,255,255,0.75);
            padding: 6px 12px;
            font-size: 12px;
        }}
        QPushButton:hover {{ border-color: {YELLOW}; color: {YELLOW}; }}
        QPushButton:disabled {{ border-color: rgba(255,255,255,0.12); color: rgba(255,255,255,0.30); }}
    """


def primary_btn_css(enabled=True):
    """Shared "primary action" button style (Build Rally Reel, Render, etc.)."""
    if enabled:
        return f"""
            QPushButton {{
                background: {YELLOW}; color: {BLACK};
                font-size: 14px; font-weight: 700; letter-spacing: 0.07em;
                border-radius: 8px; border: none;
            }}
            QPushButton:hover  {{ background: {YELLOW_HOVER}; }}
            QPushButton:pressed {{ background: {YELLOW_PRESS}; }}
        """
    return f"""
        QPushButton {{
            background: rgba(232,255,61,0.18); color: rgba(232,255,61,0.40);
            font-size: 14px; font-weight: 700; letter-spacing: 0.07em;
            border-radius: 8px; border: none;
        }}
    """


def label_css():
    return "color: rgba(255,255,255,0.60); font-size: 10px; font-weight: 700; letter-spacing: 0.08em;"


def line_edit_css():
    return f"""
        QLineEdit {{
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 6px;
            color: {WHITE};
            padding: 9px 12px;
            font-size: 13px;
        }}
        QLineEdit:focus {{ border: 1px solid {YELLOW}; }}
    """
