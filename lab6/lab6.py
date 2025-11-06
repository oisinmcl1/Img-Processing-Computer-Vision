import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio

# Oisin Mc Laughlin
# 22441106

# TASK 1

img = cv.imread("img_road.jpg")

# Convert BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# Define HSV range for yellow
# Hue for yellow ≈ 20–30, Saturation and Value should be high
lower_yellow = np.array([20, 100, 40])
upper_yellow = np.array([30, 255, 255])
# Create mask
mask = cv.inRange(hsv, lower_yellow, upper_yellow)
# Extract yellow parts
Yellow_region = cv.bitwise_and(img, img, mask=mask)

# gray = cv.cvtColor(Yellow_region, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(Yellow_region, (5, 5), 3.0)

# Detect yellow lines in blue in image
edges = cv.Canny(blur, 50, 150)
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                       threshold=80, minLineLength=40, maxLineGap=10)
out = img.copy()
if lines is not None:
    for l in lines:
        # Get line endpoints
        x1, y1, x2, y2 = l[0]
        cv.line(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.imshow(out)
plt.show()



# TASK 2

DATA_DIR = 'att_faces'  # folder with s1 ... s40 subfolders, each with 1..10.pgm
NUM_SUBJECTS = 40
TRAIN_PER_SUBJECT = 9  # 9 train, 1 test (the 10th)
TEST_PER_SUBJECT = 1
IMAGE_SIZE = (112, 92)  # (H, W)
K = 200  # number of eigenfaces to use
GIVEN_INDEX = 3  # an example index for reconstruction (1-based in MATLAB)
# Using save file from lab5
USE_SAVED_DATA = True  # Set False to rebuild A and mean face
SAVE_FILE = 'face_data_eigenfaces_orl.npz'  # where A and mean_image are saved

def load_training_matrix(data_dir, num_subjects, train_per_subject, image_size):
    """Build A (pixels x num_train) from ORL/AT&T training images."""
    H, W = image_size
    num_train = num_subjects * train_per_subject
    A = np.zeros((H * W, num_train), dtype=np.float64)
    idx = 0
    for subject in range(1, num_subjects + 1):
        subject_dir = os.path.join(data_dir, f's{subject}')
        for img_num in range(1, train_per_subject + 1):
            img_path = os.path.join(subject_dir, f'{img_num}.pgm')
            A[:, idx] = read_pgm(img_path)
            idx += 1
    return A

# ----------------------- Utilities ------------------------
def read_pgm(path):
    """Read .pgm as float64 vectorized (H*W,)."""
    img = imageio.imread(path)
    # imageio loads greyscale as (H, W); ensure dtype float64
    return img.astype(np.float64).reshape(-1)


if USE_SAVED_DATA and os.path.exists(SAVE_FILE):
    data = np.load(SAVE_FILE)
    A = data['A']
    mean_image = data['mean_image']
else:
    A = load_training_matrix(DATA_DIR, NUM_SUBJECTS, TRAIN_PER_SUBJECT, IMAGE_SIZE)
    mean_image = A.mean(axis=0)  # WRONG! (kept for clarity below)

C = A.T @ A

eigvals, eigvecs_C = np.linalg.eigh(C)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs_C = eigvecs_C[:, order]

total_variance = np.sum(eigvals)

k_values = [1, 3, 5, 10, 20]
for k in k_values:
    variance_captured = np.sum(eigvals[:k])
    percentage = (variance_captured / total_variance) * 100
    print(f"k = {k}: {percentage:.2f}% variance captured")
