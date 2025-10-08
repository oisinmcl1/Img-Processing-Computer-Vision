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


### SECTION 2 ###

# Different types of thresholding
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY) # Simple binary thresholding - dark pixels become black, bright pixels become white
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV) # Inverse binary thresholding - dark pixels become white, bright pixels become black
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC) # Truncate thresholding - bright pixels become the threshold value, dark pixels remain unchanged
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO) # To zero thresholding - dark pixels become black, bright pixels remain unchanged
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV) # Inverse to zero thresholding - dark pixels remain unchanged, bright pixels become black

# titles for the plots
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']

# images to be plotted
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# Plot all the images
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
