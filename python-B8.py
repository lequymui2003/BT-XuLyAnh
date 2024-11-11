import cv2
import numpy as np
from matplotlib import pyplot as plt

# Đọc ảnh vệ tinh
image = cv2.imread('Input//3182197_1.jpg', cv2.IMREAD_GRAYSCALE)

# Gaussian Blur (Làm mờ Gaussian)
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# 1. Sobel Edge Detection
sobel_x = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)

# 2. Prewitt Edge Detection (dùng bộ lọc thủ công vì OpenCV không có Prewitt)
prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

prewitt_x = cv2.filter2D(gaussian_blur, cv2.CV_32F, prewitt_kernel_x)
prewitt_y = cv2.filter2D(gaussian_blur, cv2.CV_32F, prewitt_kernel_y)
prewitt = cv2.magnitude(prewitt_x, prewitt_y)

# 3. Roberts Edge Detection
roberts_kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

roberts_x = cv2.filter2D(gaussian_blur, cv2.CV_32F, roberts_kernel_x)
roberts_y = cv2.filter2D(gaussian_blur, cv2.CV_32F, roberts_kernel_y)
roberts = cv2.magnitude(roberts_x, roberts_y)

# 4. Canny Edge Detection
canny = cv2.Canny(np.uint8(gaussian_blur), 100, 200)

# Hiển thị các kết quả
titles = ['Original Image', 'Gaussian Blur', 'Sobel Edge', 'Prewitt Edge', 'Roberts Edge', 'Canny Edge']
images = [image, gaussian_blur, sobel, prewitt, roberts, canny]

plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
