import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Task1-1.png')
assert img is not None, "file could not be read, check with os.path.exists()"

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()

# equalise histogram
hist_og, bins = np.histogram(gray.flatten(), 256, [0,256])
cdf_og = hist_og.cumsum()
cdf_og_normalized = cdf_og * hist_og.max() / cdf_og.max()

img_eq = cv.equalizeHist(gray)

hist_eq, bins = np.histogram(img_eq.flatten(), 256, [0,256])
cdf_eq = hist_eq.cumsum()
cdf_eq_normalized = cdf_eq * hist_eq.max() / cdf_eq.max()

# plot
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(img_eq, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.show()


# thresholding
thresholds = [80, 100, 120, 140, 160, 180, 200, 220, 240]

plt.figure(figsize=(15, 10))
for i, threshold in enumerate(thresholds):
    ret, thresh_img = cv.threshold(img_eq, threshold, 255, cv.THRESH_BINARY)
    plt.subplot(3, 3, i + 1)
    plt.imshow(thresh_img, cmap='gray')
    plt.title(f'Threshold: {threshold}', fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()

ret, thresholded_img = cv.threshold(img_eq, 100, 255, cv.THRESH_BINARY)


# remove nosie
kernel_size = 3
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

# Try different operations
eroded_img = cv.erode(thresholded_img, kernel, iterations=1)
dilated_img = cv.dilate(thresholded_img, kernel, iterations=1)
opened_img = cv.morphologyEx(thresholded_img, cv.MORPH_OPEN, kernel)
closed_img = cv.morphologyEx(thresholded_img, cv.MORPH_CLOSE, kernel)

# Plot morphological operations
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(thresholded_img, cmap='gray')
plt.title('Original', fontsize=11, fontweight='bold')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(eroded_img, cmap='gray')
plt.title(f'Erosion ({kernel_size}×{kernel_size})', fontsize=11, fontweight='bold')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(dilated_img, cmap='gray')
plt.title(f'Dilation ({kernel_size}×{kernel_size})', fontsize=11, fontweight='bold')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(opened_img, cmap='gray')
plt.title(f'Opening ({kernel_size}×{kernel_size})', fontsize=11, fontweight='bold')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(closed_img, cmap='gray')
plt.title(f'Closing ({kernel_size}×{kernel_size})', fontsize=11, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()


denoised_img = opened_img.copy()

# gaussian blur
blurred_img_spatial = cv.GaussianBlur(denoised_img, (5, 5), 1)

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.imshow(denoised_img, cmap='gray')
plt.title('Denoised Image')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(blurred_img_spatial, cmap='gray')
plt.title('Spatial Domain Gaussian Blur')
plt.axis('off')

plt.show()


# thinning here


