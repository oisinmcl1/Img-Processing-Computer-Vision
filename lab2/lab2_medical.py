# Task 1: Medical Image Analysis (COMPLETE VERSION)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

print("=" * 70)
print("TASK 1: MORPHOLOGICAL IMAGE PROCESSING PIPELINE")
print("=" * 70)

# Import img
img = cv.imread('medical.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

# Change BGR to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

print(f"\nImage loaded: {img.shape[1]} × {img.shape[0]} pixels")

### STEP 1.1: CHANNEL SELECTION ###
print("\n--- STEP 1.1: Channel Selection ---")

# Split the channels
r, g, b = cv.split(img)

# Show original img and selected channel
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(g, cmap='gray')
plt.title('Green Channel (Selected)', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('task1_step1.1_green_channel.jpg', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Selected: Green channel (best contrast)")
cv.imwrite('task1_step1.1_green_channel_only.jpg', g)

### STEP 1.2: HISTOGRAM EQUALIZATION ###
print("\n--- STEP 1.2: Histogram Equalization ---")

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
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.plot(cdf_normalized, color='b')
plt.hist(g.flatten(), 256, [0, 256], color='r', alpha=0.7)
plt.xlim([0, 256])
plt.legend(('CDF', 'Histogram'), loc='upper left')
plt.title('Before Equalization', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(cdf_eq_normalized, color='b')
plt.hist(img_equalized.flatten(), 256, [0, 256], color='r', alpha=0.7)
plt.xlim([0, 256])
plt.legend(('CDF', 'Histogram'), loc='upper left')
plt.title('After Equalization', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.imshow(g, cmap='gray')
plt.title('Green Channel (Original)', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_equalized, cmap='gray')
plt.title('Histogram Equalized', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('task1_step1.2_histogram_equalization.jpg', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Method: Histogram Equalization")
print("✓ Contrast enhanced")
cv.imwrite('task1_step1.2_equalized.jpg', img_equalized)

### STEP 1.3: THRESHOLDING ###
print("\n--- STEP 1.3: Thresholding ---")

# Show multiple thresholds for selection
thresholds = [80, 100, 120, 140, 160, 180, 200, 220, 240]

plt.figure(figsize=(15, 10))
for i, threshold in enumerate(thresholds):
    ret, thresh_img = cv.threshold(img_equalized, threshold, 255, cv.THRESH_BINARY)
    plt.subplot(3, 3, i + 1)
    plt.imshow(thresh_img, cmap='gray')
    plt.title(f'Threshold: {threshold}', fontsize=10)
    plt.axis('off')

plt.suptitle('Exploring Different Threshold Values', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Selected threshold
selected_threshold = 200
ret, threshold_img = cv.threshold(img_equalized, selected_threshold, 255, cv.THRESH_BINARY)

print(f"✓ Selected threshold: {selected_threshold}")
print(f"  White pixels: {np.sum(threshold_img == 255):,}")
print(f"  Black pixels: {np.sum(threshold_img == 0):,}")

# Plot selected thresholded image
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_equalized, cmap='gray')
plt.title('Equalized Image', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(threshold_img, cmap='gray')
plt.title(f'Thresholded Image (T={selected_threshold})', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('task1_step1.3_thresholded.jpg', dpi=150, bbox_inches='tight')
plt.show()

cv.imwrite('task1_step1.3_binary.jpg', threshold_img)

### STEP 1.4: NOISE REMOVAL ###
print("\n--- STEP 1.4: Morphological Noise Removal ---")

# Create structuring element
kernel_size = 5
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

# Try different operations
eroded_img = cv.erode(threshold_img, kernel, iterations=1)
dilated_img = cv.dilate(threshold_img, kernel, iterations=1)
opened_img = cv.morphologyEx(threshold_img, cv.MORPH_OPEN, kernel)
closed_img = cv.morphologyEx(threshold_img, cv.MORPH_CLOSE, kernel)

# Plot morphological operations
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(threshold_img, cmap='gray')
plt.title('Original Binary', fontsize=11, fontweight='bold')
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

plt.suptitle('Morphological Operations Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('task1_step1.4_morphological_ops.jpg', dpi=150, bbox_inches='tight')
plt.show()

# Select opening (best for noise removal)
cleaned_img = opened_img.copy()

print(f"✓ Selected operation: MORPH_OPEN (Opening)")
print(f"✓ Structuring element: ELLIPSE (disk shape)")
print(f"✓ Kernel size: {kernel_size}×{kernel_size}")
print(f"  White pixels before: {np.sum(threshold_img == 255):,}")
print(f"  White pixels after: {np.sum(cleaned_img == 255):,}")

# Side-by-side before/after
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(threshold_img, cmap='gray')
plt.title('Before Noise Removal', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cleaned_img, cmap='gray')
plt.title('After Opening (Noise Removed)', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

cv.imwrite('task1_step1.4_cleaned.jpg', cleaned_img)

### STEP 1.5: CONNECTED COMPONENTS ###
print("\n--- STEP 1.5: Connected Components Analysis ---")

# Find connected components
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
    cleaned_img,
    connectivity=8
)

num_regions = num_labels - 1  # Subtract background

print(f"✓ Connected components found: {num_regions}")
print(f"  (Including background label 0)")

# Create colored visualization
colored_components = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
np.random.seed(42)
for label in range(1, num_labels):
    color = np.random.randint(0, 255, size=3)
    colored_components[labels == label] = color

# Display
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(cleaned_img, cmap='gray')
plt.title('Cleaned Binary Image', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(colored_components)
plt.title(f'Connected Components\n({num_regions} regions found)', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('task1_step1.5_connected_components.jpg', dpi=150, bbox_inches='tight')
plt.show()

# Show top 10 largest regions by area
print("\nTop 10 largest regions:")
print(f"{'Label':<8} {'Area (pixels)':<15} {'Width':<8} {'Height':<8}")
print("-" * 50)

sorted_indices = np.argsort(stats[1:, cv.CC_STAT_AREA])[::-1] + 1
for i, label in enumerate(sorted_indices[:10]):
    area = stats[label, cv.CC_STAT_AREA]
    width = stats[label, cv.CC_STAT_WIDTH]
    height = stats[label, cv.CC_STAT_HEIGHT]
    print(f"{label:<8} {area:<15} {width:<8} {height:<8}")

### STEP 1.6: FILTERING FAT GLOBULES ###
print("\n--- STEP 1.6: Filtering Fat Globules ---")

# Calculate properties for each region
print("\nCalculating shape properties for each region...")

all_properties = []

for label in range(1, num_labels):
    # Get mask for this region
    region_mask = (labels == label).astype(np.uint8) * 255

    # Find contour
    contours, _ = cv.findContours(region_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        continue

    contour = contours[0]

    # Area (from stats)
    area = stats[label, cv.CC_STAT_AREA]

    # Perimeter
    perimeter = cv.arcLength(contour, True)

    # Compactness (circularity)
    if perimeter > 0:
        compactness = (4 * np.pi * area) / (perimeter ** 2)
    else:
        compactness = 0

    # Eccentricity
    if len(contour) >= 5:
        ellipse = cv.fitEllipse(contour)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        if major_axis > 0:
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        else:
            eccentricity = 0
    else:
        eccentricity = 0

    all_properties.append({
        'label': label,
        'area': area,
        'perimeter': perimeter,
        'compactness': compactness,
        'eccentricity': eccentricity
    })

print(f"✓ Properties calculated for {len(all_properties)} regions")

# Visualize property distributions
areas = [p['area'] for p in all_properties]
compactnesses = [p['compactness'] for p in all_properties]
eccentricities = [p['eccentricity'] for p in all_properties]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(areas, bins=30, color='skyblue', edgecolor='black')
axes[0].set_title('Area Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Area (pixels)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(x=100, color='red', linestyle='--', linewidth=2, label='Min=100')
axes[0].axvline(x=5000, color='red', linestyle='--', linewidth=2, label='Max=5000')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(compactnesses, bins=30, color='lightcoral', edgecolor='black')
axes[1].set_title('Compactness Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Compactness')
axes[1].set_ylabel('Frequency')
axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Min=0.5')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].hist(eccentricities, bins=30, color='lightgreen', edgecolor='black')
axes[2].set_title('Eccentricity Distribution', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Eccentricity')
axes[2].set_ylabel('Frequency')
axes[2].axvline(x=0.85, color='red', linestyle='--', linewidth=2, label='Max=0.85')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1_step1.6_property_distributions.jpg', dpi=150, bbox_inches='tight')
plt.show()

# Set filtering criteria
min_area = 100
max_area = 5000
min_compactness = 0.5
max_eccentricity = 0.85

print("\nFiltering Criteria:")
print(f"  Area: {min_area} to {max_area} pixels")
print(f"  Compactness: ≥ {min_compactness}")
print(f"  Eccentricity: ≤ {max_eccentricity}")

# Apply filters
fat_globule_labels = []

for prop in all_properties:
    if (min_area <= prop['area'] <= max_area and
            prop['compactness'] >= min_compactness and
            prop['eccentricity'] <= max_eccentricity):
        fat_globule_labels.append(prop['label'])

print(f"\nRegions before filtering: {len(all_properties)}")
print(f"Fat globules detected: {len(fat_globule_labels)}")
print(f"Regions filtered out: {len(all_properties) - len(fat_globule_labels)}")

