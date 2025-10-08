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


### SECTION 3 ###

sigma = 1.0
blurred_image = cv.GaussianBlur(img, (5, 5), sigma)

# Apply Laplacian
laplacian_image = cv.Laplacian(blurred_image, cv.CV_64F)
laplacian_image = cv.convertScaleAbs(laplacian_image)

# Plot original and Laplacian images
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Laplacian of Gaussian (LoG)')
plt.show()


### SECTION 4 ###

# Compute the 2D FFT of the image
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Compute the magnitude spectrum and use logarithmic scaling for better visualization
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Plot the original image and its magnitude spectrum
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


### SECTION 5 ###

# Create a high-pass filter mask, you can also design low-pass filter or any filter of your choice
rows, cols = img.shape
crow, ccol = rows//2, cols//2

# Create a mask with a square of 1s in the center (low frequencies) and 0s elsewhere (high frequencies)
fshift[crow-30:crow+31, ccol-30:ccol+31] = 0

# Inverse FFT to get the filtered image back
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)

# Plot the original image and the filtered image
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
plt.show()

