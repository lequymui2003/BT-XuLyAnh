import numpy as np
import cv2
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Đọc và xử lý ảnh
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):  # Kiểm tra nếu đây là file
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Chuyển ảnh sang ảnh xám
            img = cv2.resize(img, (64, 64))  # Thay đổi kích thước ảnh
            images.append(img.flatten() / 255.0)  # Chuyển ảnh thành vector và chuẩn hóa
            # Lấy nhãn từ phần tên file (trước dấu ".")
            label = filename.split('.')[0]  # Lấy nhãn là phần tên trước dấu "."
            labels.append(label)
    return np.array(images), np.array(labels)

# Đường dẫn đến thư mục chứa ảnh động vật trong cùng thư mục với file mã nguồn
folder_path = 'Image\\test1'  # Sử dụng đường dẫn tương đối
X, y = load_images_from_folder(folder_path)

# Kiểm tra dữ liệu sau khi tải
print("Các nhãn trong dữ liệu:", np.unique(y))
print("Số lượng mẫu dữ liệu:", len(X))
print("Số lượng nhãn:", len(y))

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hàm đánh giá mô hình
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, training_time, prediction_time

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn_accuracy, knn_training_time, knn_prediction_time = evaluate_model(knn, X_train, X_test, y_train, y_test)

# SVM
svm = SVC(kernel='linear')
svm_accuracy, svm_training_time, svm_prediction_time = evaluate_model(svm, X_train, X_test, y_train, y_test)

# ANN
ann = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
ann_accuracy, ann_training_time, ann_prediction_time = evaluate_model(ann, X_train, X_test, y_train, y_test)

# Kết quả
print("Kết quả phân lớp ảnh động vật:")
print(f"KNN: Độ chính xác = {knn_accuracy:.2f}, Thời gian huấn luyện = {knn_training_time:.2f}s, Thời gian dự đoán = {knn_prediction_time:.2f}s")
print(f"SVM: Độ chính xác = {svm_accuracy:.2f}, Thời gian huấn luyện = {svm_training_time:.2f}s, Thời gian dự đoán = {svm_prediction_time:.2f}s")
print(f"ANN: Độ chính xác = {ann_accuracy:.2f}, Thời gian huấn luyện = {ann_training_time:.2f}s, Thời gian dự đoán = {ann_prediction_time:.2f}s")
