"""
web_app.py
==========
Gradio web interface for the Anya Rally Detector.

Flow
----
1. Upload a tennis video.
2. A reference frame is extracted and displayed.
3. Click 4 court corners on the image (Near-Left/BL, Near-Right/BR,
   Far-Right/TR, Far-Left/TL).
4. Choose Auto Active Zone (computed from court corners) or click 8 manual
   points to define a custom active zone polygon.
5. Click "Run Detection" — the rally segments are detected and combined into
   a downloadable MP4.

Environment variables
---------------------
BALL_MODEL_PATH    — path to the custom ball-detection YOLO weights
PLAYER_MODEL_PATH  — path to the player-detection YOLO weights
TROPHY_MODEL_PATH  — path to the trophy-pose YOLO weights (optional; only
                     used in ARMED state which this pipeline bypasses)
"""

import os
import tempfile

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rally_detector import detect_rallies

# ── Model paths from env (override defaults in anya_base.py) ──────────────────
BALL_MODEL_PATH   = os.getenv("BALL_MODEL_PATH")
PLAYER_MODEL_PATH = os.getenv("PLAYER_MODEL_PATH")
TROPHY_MODEL_PATH = os.getenv("TROPHY_MODEL_PATH")   # optional

# ── Point labels and colours ──────────────────────────────────────────────────
COURT_LABELS  = ["BL (near-left)", "BR (near-right)", "TR (far-right)", "TL (far-left)"]
COURT_COLOURS = [(255, 80,  80),  (80, 200,  80),  (80, 120, 255), (255, 200,  40)]
ZONE_COLOUR   = (255, 160,  40)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_points(base_np: np.ndarray, court_pts, zone_pts) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(base_np, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    for i, pt in enumerate(court_pts):
        x, y = int(pt[0]), int(pt[1])
        r = 9
        col = COURT_COLOURS[i]
        draw.ellipse([x-r, y-r, x+r, y+r], fill=col, outline="white", width=2)
        draw.text((x + 12, y - 10), COURT_LABELS[i].split()[0], fill=col)

    if len(court_pts) == 4:
        pts = [(int(p[0]), int(p[1])) for p in court_pts]
        # Draw court quadrilateral
        for j in range(4):
            draw.line([pts[j], pts[(j + 1) % 4]], fill=(200, 200, 200), width=1)

    if len(zone_pts) >= 2:
        zpts = [(int(p[0]), int(p[1])) for p in zone_pts]
        for j in range(len(zpts) - 1):
            draw.line([zpts[j], zpts[j+1]], fill=ZONE_COLOUR, width=2)
        if len(zone_pts) == 8:
            draw.line([zpts[-1], zpts[0]], fill=ZONE_COLOUR, width=2)

    for i, pt in enumerate(zone_pts):
        x, y = int(pt[0]), int(pt[1])
        draw.ellipse([x-6, y-6, x+6, y+6], fill=ZONE_COLOUR, outline="white", width=1)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _compute_auto_zone(court_pts):
    """6-vertex active-zone polygon derived from the 4 court corners."""
    BL, BR, TR, TL = court_pts
    return [
        [BL[0], BL[1]],
        [BR[0], BR[1]],
        [BR[0], BR[1] - 150],
        [TR[0], TR[1] - 200],
        [TL[0], TL[1] - 200],
        [BL[0], BL[1] - 150],
    ]


def _extract_reference_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = min(300, max(0, total // 2))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read a reference frame from the video.")
    return cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)


# ── Gradio app ────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Anya Rally Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Anya Rally Detector\nUpload a tennis video, mark the court, and download a clip of only the active rallies.")

        # ── Persistent state ──────────────────────────────────────────────
        base_frame_state  = gr.State(None)   # np.ndarray (BGR)
        court_pts_state   = gr.State([])     # list of [x, y]
        zone_pts_state    = gr.State([])     # list of [x, y]
        phase_state       = gr.State("court")  # "court" | "zone" | "ready"
        video_path_state  = gr.State(None)

        # ── Step 1: Upload ────────────────────────────────────────────────
        with gr.Row():
            video_input = gr.Video(label="Upload tennis video", height=200)
            extract_btn = gr.Button("Extract reference frame →", variant="primary")

        status_box = gr.Textbox(label="Status", value="Upload a video and click Extract.", interactive=False)

        # ── Step 2: Annotate frame ────────────────────────────────────────
        annotated_img = gr.Image(
            label="Click to place court corners, then active-zone points",
            type="numpy",
            interactive=False,
            height=400,
        )

        with gr.Row():
            reset_court_btn = gr.Button("Reset court corners", variant="secondary")
            auto_zone_btn   = gr.Button("Auto active zone (from corners)", variant="secondary")
            reset_zone_btn  = gr.Button("Reset zone points", variant="secondary")

        # ── Step 3: Run ───────────────────────────────────────────────────
        run_btn      = gr.Button("Run Rally Detection", variant="primary", interactive=False)
        output_video = gr.Video(label="Output — rally highlights")

        # ── Callbacks ────────────────────────────────────────────────────

        def on_extract(video_file, base_frame, video_path_s):
            if video_file is None:
                return base_frame, [], [], "court", video_path_s, None, "Upload a video first.", gr.update(interactive=False)
            path = video_file if isinstance(video_file, str) else video_file.name
            frame = _extract_reference_frame(path)
            disp  = _draw_points(frame, [], [])
            return (
                frame, [], [], "court", path,
                disp,
                "Step 1: Click the 4 court corners in this order: Near-Left (BL), Near-Right (BR), Far-Right (TR), Far-Left (TL).",
                gr.update(interactive=False),
            )

        extract_btn.click(
            on_extract,
            inputs=[video_input, base_frame_state, video_path_state],
            outputs=[base_frame_state, court_pts_state, zone_pts_state, phase_state, video_path_state,
                     annotated_img, status_box, run_btn],
        )

        def on_image_click(base_frame, court_pts, zone_pts, phase, evt: gr.SelectData):
            if base_frame is None:
                return base_frame, court_pts, zone_pts, phase, None, "Extract a frame first.", gr.update(interactive=False)

            x, y = evt.index

            if phase == "court":
                if len(court_pts) < 4:
                    court_pts = court_pts + [[x, y]]
                    remaining = 4 - len(court_pts)
                    if remaining > 0:
                        msg = f"Click {COURT_LABELS[len(court_pts)].split()[0]}... ({remaining} more)"
                        new_phase = "court"
                        run_active = False
                    else:
                        msg = "Court corners set! Click 'Auto active zone' or click 8 zone points manually."
                        new_phase = "zone"
                        run_active = False
                else:
                    msg = "Court already has 4 corners. Reset if you want to redo."
                    new_phase = "zone"
                    run_active = len(zone_pts) >= 4

            elif phase == "zone":
                if len(zone_pts) < 8:
                    zone_pts = zone_pts + [[x, y]]
                    remaining = 8 - len(zone_pts)
                    if remaining > 0:
                        msg = f"Zone point {len(zone_pts)}/8 placed. ({remaining} more)"
                        new_phase = "zone"
                        run_active = False
                    else:
                        msg = "Active zone defined! Click 'Run Rally Detection'."
                        new_phase = "ready"
                        run_active = True
                else:
                    msg = "Zone already has 8 points. Reset zone if you want to redo."
                    new_phase = "ready"
                    run_active = True
            else:
                msg = "Reset court or zone to make changes."
                new_phase = phase
                run_active = True

            disp = _draw_points(base_frame, court_pts, zone_pts)
            return base_frame, court_pts, zone_pts, new_phase, disp, msg, gr.update(interactive=run_active)

        annotated_img.select(
            on_image_click,
            inputs=[base_frame_state, court_pts_state, zone_pts_state, phase_state],
            outputs=[base_frame_state, court_pts_state, zone_pts_state, phase_state,
                     annotated_img, status_box, run_btn],
        )

        def on_reset_court(base_frame):
            if base_frame is None:
                return [], [], "court", None, "Extract a frame first.", gr.update(interactive=False)
            disp = _draw_points(base_frame, [], [])
            return [], [], "court", disp, "Court reset. Click Near-Left (BL) first.", gr.update(interactive=False)

        reset_court_btn.click(
            on_reset_court,
            inputs=[base_frame_state],
            outputs=[court_pts_state, zone_pts_state, phase_state, annotated_img, status_box, run_btn],
        )

        def on_auto_zone(base_frame, court_pts, zone_pts):
            if len(court_pts) < 4:
                return zone_pts, "zone", None, "Place all 4 court corners first.", gr.update(interactive=False)
            auto = _compute_auto_zone(court_pts)
            disp = _draw_points(base_frame, court_pts, auto)
            return auto, "ready", disp, "Auto active zone applied. Click 'Run Rally Detection'.", gr.update(interactive=True)

        auto_zone_btn.click(
            on_auto_zone,
            inputs=[base_frame_state, court_pts_state, zone_pts_state],
            outputs=[zone_pts_state, phase_state, annotated_img, status_box, run_btn],
        )

        def on_reset_zone(base_frame, court_pts):
            if base_frame is None:
                return [], "zone" if len(court_pts) == 4 else "court", None, "Extract a frame first.", gr.update(interactive=False)
            disp = _draw_points(base_frame, court_pts, [])
            return [], "zone", disp, "Zone reset. Click 8 points or use Auto active zone.", gr.update(interactive=False)

        reset_zone_btn.click(
            on_reset_zone,
            inputs=[base_frame_state, court_pts_state],
            outputs=[zone_pts_state, phase_state, annotated_img, status_box, run_btn],
        )

        def on_run(video_path, court_pts, zone_pts, progress=gr.Progress()):
            if not video_path:
                raise gr.Error("No video loaded.")
            if len(court_pts) < 4:
                raise gr.Error("Place all 4 court corners first.")
            if len(zone_pts) < 4:
                raise gr.Error("Define the active zone first (auto or manual).")

            progress(0.0, desc="Initialising models and exclusion zones…")

            out_dir  = tempfile.mkdtemp(prefix="anya_out_")
            out_path = os.path.join(out_dir, "rallies.mp4")

            def _cb(frac, msg):
                progress(0.1 + frac * 0.85, desc=msg)

            detect_rallies(
                video_path,
                output_path=out_path,
                headless=True,
                court_vertices=court_pts,
                active_zone_polygon=zone_pts,
                ball_model_path=BALL_MODEL_PATH,
                player_model_path=PLAYER_MODEL_PATH,
                # Rally detector forces ACTIVE state; trophy model (ARMED only)
                # and exclusion zone scan are both skipped to cut startup time.
                trophy_model_path=None,
                skip_exclusion_zones=True,
                progress_callback=_cb,
            )

            progress(1.0, desc="Done!")

            if not os.path.isfile(out_path):
                raise gr.Error("Detection finished but no output file was produced. Check that rallies were detected.")

            return out_path

        run_btn.click(
            on_run,
            inputs=[video_path_state, court_pts_state, zone_pts_state],
            outputs=[output_video],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(max_size=4).launch(server_name="0.0.0.0", server_port=7860, share=False)
