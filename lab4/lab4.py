import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = "flowers.jpg"
img = cv.imread(IMG_PATH)

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
