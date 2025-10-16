import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Task2.png')
assert img is not None, "file could not be read, check with os.path.exists()"

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()

# denoise img in freq domain
kernel_size = 5
sigma = 1

blurred_img_spatial = cv.GaussianBlur(gray, (kernel_size, kernel_size), sigma)

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(blurred_img_spatial, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.show()

kernel = cv.getGaussianKernel(kernel_size, sigma)
kernel = kernel * kernel.T

# pad kernel to image size
img_height, img_width = gray.shape
kernel_padded = np.zeros((img_height, img_width))
kh, kw = kernel.shape
kernel_padded[:kh, :kw] = kernel

# Calculate 2D FFT and shift
kernel_fft = np.fft.fft2(kernel_padded)
kernel_fft_shifted = np.fft.fftshift(kernel_fft)

# gausing filter in freq domain
gaussian_filter = np.abs(kernel_fft_shifted)
gaussian_filter = gaussian_filter / gaussian_filter.max()

# denoise using suitable low pass filter
f_gray = np.fft.fft2(gray)
f_gray_shifted = np.fft.fftshift(f_gray)
magnitude_original = 20 * np.log(np.abs(f_gray_shifted) + 1)

f_gray_filtered = f_gray_shifted * gaussian_filter
magnitude_filtered = 20 * np.log(np.abs(f_gray_filtered) + 1)

# original img converted back to spatial domain
f_ishift = np.fft.ifftshift(f_gray_shifted)
img_back_original = np.fft.ifft2(f_ishift)
img_back_original = np.abs(img_back_original)

# filtered img converted back to spatial domain
f_ishift_filtered = np.fft.ifftshift(f_gray_filtered)
img_back_filtered = np.fft.ifft2(f_ishift_filtered)
img_back_filtered = np.abs(img_back_filtered)
img_back_filtered = np.clip(img_back_filtered, 0, 255).astype(np.uint8)

# plot results
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(magnitude_original, cmap='gray')
plt.title('Magnitude Spectrum (Original)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(img_back_filtered, cmap='gray')
plt.title('Filtered Image (Spatial Domain)')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(magnitude_filtered, cmap='gray')
plt.title('Magnitude Spectrum (Filtered)')

plt.axis('off')
plt.show()