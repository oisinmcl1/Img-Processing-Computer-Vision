# Task 1: Medical Img analysis
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Import img
img = cv.imread('medical.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

# Change BGR to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Split the channels
r, g, b = cv.split(img)

# Show original img
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# Green channel
plt.subplot(1, 2, 2)
plt.imshow(g, cmap='gray')
plt.title('Green Channel')
plt.axis('off')

plt.show()


# Histogram equalization
hist_og, bins = np.histogram(g.flatten(), 256, [0, 256])
cdf = hist_og.cumsum()
cdf_normalized = cdf * float(hist_og.max()) / cdf.max()

# Img equalization
img_equalized = cv.equalizeHist(g)

# Histogram after equalization
hist_eq, bins = np.histogram(img_equalized.flatten(), 256, [0, 256])
cdf_eq = hist_eq.cumsum()
cdf_eq_normalized = cdf_eq * float(hist_eq.max()) / cdf_eq.max()

# Plot CDF and histogram before and after equalization
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(cdf_normalized, color='b')
plt.hist(g.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.title('Before Equalization')

plt.subplot(2, 2, 2)
plt.plot(cdf_eq_normalized, color='b')
plt.hist(img_equalized.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.title('After Equalization')

plt.subplot(2, 2, 3)
plt.imshow(g, cmap='gray')
plt.title('Green Channel Before Equalization')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_equalized, cmap='gray')
plt.title('Green Channel After Equalization')
plt.axis('off')

plt.tight_layout()
plt.show()
