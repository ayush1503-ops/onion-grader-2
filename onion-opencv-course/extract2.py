import cv2, os, numpy as np

vdir = "/home/user/onion-opencv-course/videos"
fdir = "/home/user/onion-opencv-course/frames"
cdir = "/home/user/onion-opencv-course/contacts"
os.makedirs(fdir, exist_ok=True); os.makedirs(cdir, exist_ok=True)

videos = [f"part{i:02d}.h264.mp4" for i in range(1, 14)]

for v in videos:
    cap = cv2.VideoCapture(os.path.join(vdir, v))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps
    n = 12
    times = [2.0 + (dur - 4.0) * i / (n - 1) for i in range(n)]
    frames = []
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if ok:
            out = os.path.join(fdir, f"{v[:6]}_{i:02d}.jpg")
            cv2.imwrite(out, fr)
            frames.append((t, fr))
    cap.release()
    # build contact sheet 4 cols
    cols = 4
    thumb_w = 420
    rows = []
    for r in range(0, len(frames), cols):
        row = []
        for t, fr in frames[r:r+cols]:
            h, w = fr.shape[:2]
            tw = thumb_w; th = int(h * thumb_w / w)
            th_fr = cv2.resize(fr, (tw, th))
            cv2.putText(th_fr, f"{int(t)}s", (8, th - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            row.append(th_fr)
        while len(row) < cols:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    cv2.imwrite(os.path.join(cdir, f"{v[:6]}_contact.jpg"), sheet)
    print(v, f"dur={dur:.0f}s frames={len(frames)}")
print("DONE")
