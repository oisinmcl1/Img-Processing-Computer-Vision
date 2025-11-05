import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = "flowers.jpg"
img = cv.imread(IMG_PATH)

# =================== TASK 1 ===================

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
plt.subplot(1, 3, 1);
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB));
plt.title("Original");

plt.axis("off")
plt.subplot(1, 3, 2);
plt.imshow(cv.cvtColor(harris_img, cv.COLOR_BGR2RGB));
plt.title("Harris");
plt.axis("off")

plt.subplot(1, 3, 3);
plt.imshow(cv.cvtColor(shi_img, cv.COLOR_BGR2RGB));
plt.title("Shi–Tomasi");
plt.axis("off")
plt.show()

# ==============================================
