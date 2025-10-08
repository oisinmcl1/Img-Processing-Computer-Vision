import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

### SECTION 1 ###

# Read img in grayscale
# img = cv.imread('apple.jpg', cv.IMREAD_GRAYSCALE)
img = cv.imread('flowers.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# Convert 2d img array to 1d array
# 256 bins for 256 pixel values
# Range of pixel values is [0, 256]
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# CDF shows the cumulative distribution of pixel values (brightness levels)
# Normalize CDF to fit in the same range as histogram
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

# Plot CDF and histogram
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

# Using OpenCV for histogram equalization
equ = cv.equalizeHist(img)
res = np.hstack((img, equ)) # Put images side by side
cv.imwrite('res.png', res)  # Save the result