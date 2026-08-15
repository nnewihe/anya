"""Isolate why court-calibration clicks don't register on Windows.

Runs 4 combinations of (window-title encoding) x (PyQt6 loaded or not).
Click once inside each window that appears. Press q to skip a test.

    python cvclick_diag.py
"""
import sys
import cv2
import numpy as np

ASCII_TITLE = "TEST - click me"
EMDASH_TITLE = "TEST — click me"   # U+2014, same char as utilities.py:361

results = []


def run(label, title):
    img = np.full((540, 960, 3), 40, np.uint8)
    cv2.putText(img, label, (40, 260), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "click anywhere  (q to skip)", (40, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)

    state = {"hit": None}

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["hit"] = (x, y)
            cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
            cv2.imshow(title, img)

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, img)
    cv2.setMouseCallback(title, cb, state)

    print(f"\n>>> {label}: click in the window...")
    for _ in range(750):                      # ~15s at 20ms
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
        if state["hit"]:
            cv2.waitKey(300)
            break
    cv2.destroyWindow(title)
    cv2.waitKey(1)

    ok = state["hit"] is not None
    print(f"    {'CLICK REGISTERED at ' + str(state['hit']) if ok else 'NO CLICK'}")
    results.append((label, ok))


print("OpenCV:", cv2.__version__)
print("Python:", sys.version)

run("1. ASCII title, no Qt", ASCII_TITLE)
run("2. EM-DASH title, no Qt", EMDASH_TITLE)

from PyQt6.QtWidgets import QApplication
_app = QApplication(sys.argv)
print("\n[PyQt6 QApplication created]")

run("3. ASCII title, Qt loaded", ASCII_TITLE)
run("4. EM-DASH title, Qt loaded", EMDASH_TITLE)

print("\n===== SUMMARY =====")
for label, ok in results:
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
