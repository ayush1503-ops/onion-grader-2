#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracker.py - conveyor-belt counting: track onions across VIDEO frames.

Why tracking? On a moving belt the same onion appears in many frames.
Naive "count boxes per frame" counts every onion 20+ times. Tracking
gives each onion a persistent ID, so we count each onion ONCE - that's
the difference between a toy and a real belt system.

How it works (simple + honest):
 1. per frame: Otsu mask -> contours -> boxes (same CV as the grader)
 2. IoU tracker: match boxes frame-to-frame by overlap; new box with no
    match = new ID; a track missing for MAX_AGE frames = onion left
 3. a track counts only after MIN_HITS frames (no noise blobs)

Honest limits: assumes onions move roughly smoothly between frames
(belt speed reasonable); touching onions on a belt merge into one box
(same limit as the photo grader).

Run:
    python tracker.py 0                       # webcam
    python tracker.py belt.mp4                # a video file
    python tracker.py belt.mp4 --no-gui       # headless (prints count only)
    python tracker.py belt.mp4 --save out.mp4 # save annotated video
"""

import argparse

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. the tracker (pure logic - easy to test without a camera)
# ---------------------------------------------------------------------------
def iou(a, b):
    """Intersection-over-union of two boxes (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


class IoUTracker:
    """Greedy IoU matcher - the classic simple baseline."""

    def __init__(self, iou_thr=0.30, max_age=8, min_hits=3):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = {}        # id -> {"box":..., "age":..., "hits":...}
        self.next_id = 1
        self.finished = []      # confirmed IDs that have left the frame

    def update(self, boxes):
        """Feed this frame's boxes; returns list of active (id, box)."""
        unmatched = list(range(len(boxes)))
        # try to match every live track to its best-overlapping box
        for tid in list(self.tracks):
            t = self.tracks[tid]
            best, best_iou = None, self.iou_thr
            for bi in unmatched:
                v = iou(t["box"], boxes[bi])
                if v > best_iou:
                    best, best_iou = bi, v
            if best is not None:
                t["box"] = boxes[best]
                t["age"] = 0
                t["hits"] += 1
                unmatched.remove(best)
            else:
                t["age"] += 1
                if t["age"] > self.max_age:      # gone for good
                    if t["hits"] >= self.min_hits:
                        self.finished.append(tid)
                    del self.tracks[tid]

        for bi in unmatched:                     # leftovers = new onions
            self.tracks[self.next_id] = {"box": boxes[bi],
                                         "age": 0, "hits": 1}
            self.next_id += 1

        return [(tid, t["box"]) for tid, t in self.tracks.items()
                if t["hits"] >= self.min_hits]

    def total_count(self):
        """Confirmed onions so far (left + still on screen)."""
        live = sum(1 for t in self.tracks.values()
                   if t["hits"] >= self.min_hits)
        return len(self.finished) + live


# ---------------------------------------------------------------------------
# 2. per-frame detection (lightweight, reuses the grader's CV ideas)
# ---------------------------------------------------------------------------
def detect_boxes(frame):
    """Find onion-like blobs in one frame -> list of (x, y, w, h).
    Tries BOTH Otsu polarities (belt can be lighter or darker than the
    onions) and keeps whichever set looks more plausible."""
    scale = 640 / max(frame.shape[:2])
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    area_img = small.shape[0] * small.shape[1]

    def blobs(mask):
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < area_img * 0.004 or a > area_img * 0.35:
                continue                       # dust speck / whole-frame blob
            x, y, w, h = cv2.boundingRect(c)
            if w < 12 or h < 12:
                continue
            out.append((int(x / scale), int(y / scale),
                        int(w / scale), int(h / scale)))
        return out

    cand = blobs(otsu)                          # bright onions on dark belt
    inv = blobs(cv2.bitwise_not(otsu))          # dark onions on light belt
    return cand if len(cand) >= len(inv) else inv


# ---------------------------------------------------------------------------
# 3. main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Onion belt tracker")
    ap.add_argument("source", help="video file path, or 0 for webcam")
    ap.add_argument("--no-gui", action="store_true",
                    help="headless: process everything, print the count")
    ap.add_argument("--save", help="write annotated video to this file")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, size)

    tracker = IoUTracker()
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        boxes = detect_boxes(frame)
        active = tracker.update(boxes)

        if writer:
            for tid, (x, y, w, h) in active:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 82, 255), 2)
                cv2.putText(frame, f"#{tid}", (x, max(14, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 82, 255), 2)
            cv2.putText(frame, f"unique: {tracker.total_count()}",
                        (14, 30), cv2.FONT_HERSHEY_SIMPLEX, .9, (20, 120, 40), 2)
            writer.write(frame)

        if not args.no_gui:
            disp = frame if writer else frame.copy()
            for tid, (x, y, w, h) in active:
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 82, 255), 2)
                cv2.putText(disp, f"#{tid}", (x, max(14, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 82, 255), 2)
            cv2.putText(disp, f"unique: {tracker.total_count()}  "
                        f"(frame {frame_no})", (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .9, (20, 120, 40), 2)
            cv2.imshow("tracker - Q to quit", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    if not args.no_gui:
        cv2.destroyAllWindows()
    print(f"frames processed : {frame_no}")
    print(f"UNIQUE ONIONS    : {tracker.total_count()}")


if __name__ == "__main__":
    main()
