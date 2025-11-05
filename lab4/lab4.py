import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = "me.jpg"
img = cv.imread(IMG_PATH)

# =================== TASK 1 ===================
"""
CANNY EDGE DETECTION WITH DIFFERENT THRESHOLDS
"""

# Convert to grayscale and apply Gaussian blur
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 1.0)

# Apply Canny edge detection with different thresholds
edges1 = cv.Canny(blur, 50, 150)
edges2 = cv.Canny(blur, 100, 200)

# Display the results
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1);
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB));
plt.title("Original");
plt.axis("off")

plt.subplot(1, 3, 2);
plt.imshow(edges1, cmap="gray");
plt.title("Canny 50/150");

plt.axis("off")
plt.subplot(1, 3, 3);
plt.imshow(edges2, cmap="gray");
plt.title("Canny 100/200");
plt.axis("off")

plt.show()

# ==============================================

# =================== TASK 2 ===================
"""
CORNER AND SALIENT POINT DETECTION USING HARRIS AND SHI-TOMASI METHODS
"""

# Convert to float32 for Harris corner detection
gray_f = np.float32(gray)

# Harris Corner Detection
harris = cv.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
# Dilate to mark the corners
harris = cv.dilate(harris, None)
harris_img = img.copy()
# Mark corners in the image in red
harris_img[harris > 0.01 * harris.max()] = [0, 0, 255]

# Shi–Tomasi (Tomasi–Kanade)
shi_img = img.copy()
# Marks corners using Shi-Tomasi method
pts = cv.goodFeaturesToTrack(gray, maxCorners=150, qualityLevel=0.01, minDistance=8)

# If corners are found, draw them
if pts is not None:
    pts = np.int32(pts)
    # For each corner point, draw a circle on the image in green
    for p in pts:
        x, y = p.ravel()
        cv.circle(shi_img, (x, y), 3, (0, 255, 0), -1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original")

plt.axis("off")
plt.subplot(1, 3, 2);
plt.imshow(cv.cvtColor(harris_img, cv.COLOR_BGR2RGB))
plt.title("Harris")
plt.axis("off")

plt.subplot(1, 3, 3);
plt.imshow(cv.cvtColor(shi_img, cv.COLOR_BGR2RGB))
plt.title("Shi–Tomasi")
plt.axis("off")
plt.show()

# ==============================================

# =================== TASK 3 ===================
"""
IMAGE REGISTRATION USING SIFT AND HOMOGRAPHY
"""

IMG_PATH = "lena.png"
img1 = cv.imread(IMG_PATH)

# Rotate and scale the image to create a second image
h, w = img1.shape[:2]
M = cv.getRotationMatrix2D((w / 2, h / 2), 15, 1.1)  # rotate + scale

# Warp the image using the transformation matrix
img2 = cv.warpAffine(img1, M, (w, h))
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Use SIFT to detect and compute keypoints and descriptors
det = cv.SIFT_create()
k1, d1 = det.detectAndCompute(gray1, None)
k2, d2 = det.detectAndCompute(gray2, None)

# Nearest neighbor matching with ratio test
norm = cv.NORM_L2
bf = cv.BFMatcher(norm)
matches = bf.knnMatch(d1, d2, k=2)

# Good matches based on Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# Draw matches
match_vis = cv.drawMatches(img1, k1, img2, k2, good[:60], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# If enough good matches are found, compute homography
if len(good) >= 4:
    # Homography computation
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Find homography using RANSAC
    H, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)

    # If homography is found, warp img1 to img2's perspective
    if H is not None:
        warped = cv.warpPerspective(img1, H, (w, h))
        overlay = cv.addWeighted(img2, 0.5, warped, 0.5, 0)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1);
plt.imshow(cv.cvtColor(match_vis, cv.COLOR_BGR2RGB));
plt.title("Matches");
plt.axis("off")
plt.subplot(1, 2, 2)
plt.axis("off")
plt.tight_layout()
plt.show()

# ==============================================

# =================== TASK 4 ===================
"""
LINE DETECTION USING HOUGH TRANSFORM
"""

IMG_PATH = "Lena.png"
img = cv.imread(IMG_PATH)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 1.0)

# Canny edge detection and Hough Line Transform
edges = cv.Canny(blur, 50, 150)
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                       threshold=80, minLineLength=40, maxLineGap=10)
out = img.copy()
if lines is not None:
    for l in lines:
        # Get line endpoints
        x1, y1, x2, y2 = l[0]
        cv.line(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1);
plt.imshow(edges, cmap="gray");
plt.title("Canny");
plt.axis("off")

plt.subplot(1, 2, 2);
plt.imshow(cv.cvtColor(out, cv.COLOR_BGR2RGB));
plt.title("Hough Lines");
plt.axis("off")
plt.show()

# ==============================================
