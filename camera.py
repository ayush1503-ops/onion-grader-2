#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 camera.py - LIVE WEBCAM MODE (laptop camera, or a video file)
 Smart India Hackathon 2026 - Problem SIH26031
=====================================================================

WHAT IT DOES (simple words):
    Opens your laptop webcam and shows a LIVE preview.
    - Press SPACE (or c)  -> take a snapshot -> grader.py analyses it,
      the annotated result pops up in a second window and is saved
      in outputs/.
    - Press q (or ESC)    -> quit.

It can also play a VIDEO FILE and you can snapshot any frame
(useful for a "conveyor belt" demo video).

HOW TO RUN:
    python camera.py                    # live webcam
    python camera.py --source 1         # a 2nd camera (USB webcam)
    python camera.py --source my.mp4    # a video file instead
    python camera.py --coin-mm 23       # tell it the coin size
    python camera.py --auto 2           # AUTO mode: re-analyze every
                                        # 2 seconds - no keys needed!

NOTE: no coin menu here - put a real coin near the onions and pass
its size with --coin-mm (27 for Rs.10/Rs.2, 23 for Rs.5, 22 for Rs.1).
"""

import argparse
import time

import cv2

import grader


def overlay(frame, fps, message="", auto_left=None):
    """Small helper: draw FPS + help text on the live frame."""
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 56), (frame.shape[1], h), (30, 30, 30), -1)
    if auto_left is not None:
        help_line = (f"FPS {fps:4.1f}   AUTO every {auto_left:.0f}s   "
                     "[SPACE]=analyze now  [q]=quit")
    else:
        help_line = f"FPS {fps:4.1f}   [SPACE/c]=snapshot  [q]=quit"
    cv2.putText(frame, help_line, (8, h - 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if message:
        cv2.putText(frame, message, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (80, 200, 255), 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser(description="Live onion grading camera")
    ap.add_argument("--source", default="0",
                    help="0 = webcam (default), 1 = second camera, "
                         "or a video file path like belt.mp4")
    ap.add_argument("--coin-mm", type=float, default=27.0,
                    help="diameter of the coin in the scene, in mm "
                         "(only used when a coin is actually detected)")
    ap.add_argument("--assume-mm", type=float, default=None,
                    help="no coin in the scene? assume this standard "
                         "onion diameter (default 55) - sizes are estimates")
    ap.add_argument("--auto", type=float, default=0,
                    help="AUTO mode: analyze every N seconds without "
                         "pressing any key (e.g. --auto 2). 0 = off")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    # webcam number or video file?
    src = int(args.source) if args.source.isdigit() else args.source
    cam = cv2.VideoCapture(src)
    if not cam.isOpened():
        raise SystemExit(f"Could not open camera/video source: {args.source!r}. "
                         "Try --source 1 or check the cable/permissions.")

    print("Live mode started. SPACE/c = snapshot + analyze, q = quit.")
    prev, fps, message = time.time(), 0.0, ""
    result_shown = None          # the annotated image currently on screen
    last_auto = time.time()      # for --auto mode

    def show_counts(rep):
        if not rep.get("onion_detected", rep.get("onion_count", 0) > 0):
            return "ERROR: No onion detected"
        p = rep["grade_percent"]
        return (f"{rep['onion_count']} onions | ~{rep['estimated_weight_kg']} kg | "
                f"A {p['A']}% URS {p['URS']}% REJ {p['REJECT']}%")

    while True:
        ok, frame = cam.read()
        if not ok:                        # video file ended (or camera lost)
            print("Stream ended.")
            break
        frame = cv2.flip(frame, 1)        # mirror view feels natural

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        # AUTO mode: analyze every N seconds WITHOUT pressing any key
        auto_left = None
        if args.auto > 0:
            auto_left = args.auto - (now - last_auto)
            if auto_left <= 0:
                last_auto = now
                try:
                    # in-memory live path: no files, draws boxes itself
                    rep = grader.analyze_frame(frame.copy(),
                                               coin_mm=args.coin_mm,
                                               assume_mm=args.assume_mm)
                    result_shown = grader.make_annotated(rep, frame, None, None)
                    message = show_counts(rep)
                    print("auto:", message)
                except Exception as exc:
                    message = f"analysis failed: {exc}"
                    print("ERROR:", exc)

        live = overlay(frame.copy(), fps, message, auto_left)
        cv2.imshow("Onion Grader - LIVE (SPACE=snapshot, q=quit)", live)
        if result_shown is not None:
            cv2.imshow("RESULT (press any key to close)", result_shown)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):         # q or ESC -> exit
            break
        if key in (ord(" "), ord("c")):   # SPACE or c -> snapshot!
            stamp = time.strftime("%Y%m%d-%H%M%S")
            snap_path = f"snapshot_{stamp}.jpg"
            cv2.imwrite(snap_path, frame)
            print(f"\nSnapshot saved: {snap_path}  -> analyzing...")
            try:
                rep = grader.analyze(snap_path, coin_mm=args.coin_mm,
                                     out_dir=args.out_dir,
                                     assume_mm=args.assume_mm)
                grader.print_report(rep)
                result_shown = cv2.imread(rep["files"][0])  # annotated image
                message = (f"#{stamp}: " + show_counts(rep))
            except Exception as exc:
                message = f"analysis failed: {exc}"
                print("ERROR:", exc)
        if result_shown is not None:
            # any key on the RESULT window closes it
            if cv2.getWindowProperty("RESULT (press any key to close)",
                                     cv2.WND_PROP_VISIBLE) < 1:
                result_shown = None

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
