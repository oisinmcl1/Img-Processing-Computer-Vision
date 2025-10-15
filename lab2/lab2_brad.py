# Task 2: Brad Pitt Derinkles
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('brad.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

# Convert BGR to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.figure(figsize=[10,8])
plt.imshow(img)
plt.axis('off')
plt.title("Original Image")
plt.tight_layout()
plt.show()

# gaussian blur
kernel_size = (11, 11)
sigma = 1.5

blurred_img = cv.GaussianBlur(img, kernel_size, sigma)

# Comparison
plt.figure(figsize=[10,8])
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis('off')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(blurred_img)
plt.axis('off')
plt.title("Blurred Image")

plt.tight_layout()
plt.show()
