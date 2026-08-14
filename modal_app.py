"""
modal_app.py
============
Deploys web_app.py as a GPU-accelerated web service on Modal.

Setup (one-time)
----------------
1. Install Modal:
       pip install modal
       modal setup               # authenticate

2. Create a Modal Volume to hold the custom YOLO weights:
       modal volume create anya-weights

3. Upload your custom weights into the volume:
       modal volume put anya-weights /path/to/ball/best.pt    /weights/ball.pt
       modal volume put anya-weights /path/to/player/best.pt  /weights/player.pt
       modal volume put anya-weights /path/to/trophy/best.pt  /weights/trophy.pt

   If your player model is a standard Ultralytics name (e.g. yolov8n.pt), it
   will be downloaded automatically; skip that upload.

4. Deploy:
       modal deploy modal_app.py

5. The app URL is printed after deploy — open it in a browser.

Serve locally for testing (no GPU charge):
       modal serve modal_app.py
"""

import pathlib

import modal

# ── Volume for custom YOLO weights ────────────────────────────────────────────
weights_volume = modal.Volume.from_name("anya-weights", create_if_missing=True)
WEIGHTS_DIR    = "/weights"

_SRC_DIR = pathlib.Path(__file__).parent

# ── Container image (source files baked in via add_local_dir) ────────────────
# Modal no longer has modal.Mount; the modern API copies local files into the
# image at build time with .add_local_dir() / .add_local_file().
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.0",
        "opencv-python-headless>=4.8",
        "filterpy>=1.4",
        "gradio>=4.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "Pillow>=10.0",
    )
    # Copy only the top-level .py source files we need (skip archive/).
    .add_local_file(str(_SRC_DIR / "web_app.py"),        remote_path="/app/web_app.py")
    .add_local_file(str(_SRC_DIR / "rally_detector.py"), remote_path="/app/rally_detector.py")
    .add_local_file(str(_SRC_DIR / "anya_base.py"),      remote_path="/app/anya_base.py")
    .add_local_file(str(_SRC_DIR / "ball_tracker.py"),   remote_path="/app/ball_tracker.py")
    .add_local_file(str(_SRC_DIR / "utilities.py"),      remote_path="/app/utilities.py")
    .add_local_file(str(_SRC_DIR / "anya_transitions.py"), remote_path="/app/anya_transitions.py")
)

app = modal.App("anya-rally-detector", image=image)

# ── Modal function ─────────────────────────────────────────────────────────────
@app.function(
    gpu="T4",
    timeout=3600,          # up to 1 h per job
    volumes={WEIGHTS_DIR: weights_volume},
    memory=8192,
    max_containers=1,      # one active job at a time per container
)
@modal.web_server(7860, startup_timeout=120)
def web_ui():
    import os
    import subprocess

    env = os.environ.copy()
    env["BALL_MODEL_PATH"]   = f"{WEIGHTS_DIR}/ball.pt"
    env["PLAYER_MODEL_PATH"] = f"{WEIGHTS_DIR}/player.pt"
    # Trophy model intentionally omitted — not used by rally detector.

    # web_app.py's __main__ block launches Gradio on 0.0.0.0:7860.
    subprocess.Popen(["python", "/app/web_app.py"], env=env)


# ── Local entrypoint for quick CLI smoke-test ─────────────────────────────────
@app.local_entrypoint()
def main():
    print("Deploy with:  modal deploy modal_app.py")
    print("Serve locally:  modal serve modal_app.py")
