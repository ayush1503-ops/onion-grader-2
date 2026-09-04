import cv2, os

videos = [f"part{i:02d}.mp4" for i in range(1, 14)]
vdir = "/home/user/onion-opencv-course/videos"
fdir = "/home/user/onion-opencv-course/frames"
os.makedirs(fdir, exist_ok=True)

for v in videos:
    path = os.path.join(vdir, v)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(v, "FAILED to open"); continue
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps
    # sample every ~12 seconds
    step = max(12.0, dur / 8)
    t = 3.0  # skip intro
    idx = 0
    while t < dur - 1:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            out = os.path.join(fdir, f"{v[:-4]}_{idx:02d}.jpg")
            cv2.imwrite(out, frame)
            idx += 1
        t += step
    cap.release()
    print(v, f"{dur:.0f}s, {fps:.0f}fps -> {idx} frames")

print("DONE")
