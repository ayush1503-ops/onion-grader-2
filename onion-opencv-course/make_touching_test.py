"""Synthesize a deterministic test: 5 'onions' (two touching) + a coin.
Used to verify the counter / watershed split."""
import cv2
import numpy as np

W = H = 900
img = np.full((H, W, 3), 255, np.uint8)  # white background

# onion-ish blobs (ellipses): (cx, cy, rx, ry, color BGR)
onions = [
    (200, 220, 90, 85, (60, 130, 200)),    # healthy golden
    (430, 220, 85, 80, (60, 130, 200)),    # healthy
    # two TOUCHING onions (tangent: centers exactly 2*radius apart -> a real
    # contact seam survives preprocessing, exactly like real onions touching)
    (250, 520, 80, 85, (70, 140, 210)),
    (410, 520, 80, 85, (70, 140, 210)),
    (620, 500, 90, 95, (40, 60, 90)),      # darker (rotten-ish)
]
for (cx, cy, rx, ry, col) in onions:
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, col, -1)

# coin: small solid dark circle
cv2.circle(img, (700, 150), 32, (95, 95, 95), -1)
cv2.circle(img, (700, 150), 32, (120, 120, 120), 3)

# texture: light radial shading so distance transform behaves like real onions
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
shade = cv2.GaussianBlur(gray, (0, 0), 60)
img = np.clip(img * (0.85 + 0.3 * (shade / 255.0))[..., None], 0, 255).astype(np.uint8)

cv2.imwrite("/home/user/onion-opencv-course/images/test_touching.jpg", img)
print("saved test_touching.jpg (5 onions, 2 touching, 1 coin)")
