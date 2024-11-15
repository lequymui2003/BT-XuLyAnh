import os
import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Đặt kích thước ảnh
img_size = 64

# Đường dẫn đến tập dữ liệu
train_dir = 'data/train'
val_dir = 'data/validation'

def load_data(directory):
    data = []
    labels = []
    for label, animal in enumerate(['cats', 'dogs']):
        folder_path = os.path.join(directory, animal)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            labels.append(label)
    data = np.array(data) / 255.0  # Chuẩn hóa ảnh
    labels = np.array(labels)
    return data, labels

# Tải dữ liệu huấn luyện và validation
X_train, y_train = load_data(train_dir)
X_val, y_val = load_data(val_dir)

# Sử dụng cho SVM (biến đổi ảnh thành vector)
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)

# Huấn luyện mô hình SVM
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train_flat, y_train)

# Đánh giá SVM
y_pred_svm = svm_model.predict(X_val_flat)
print("Accuracy SVM:", accuracy_score(y_val, y_pred_svm))

# Chuyển đổi nhãn sang dạng one-hot cho ANN và CNN
y_train_oh = to_categorical(y_train, 2)
y_val_oh = to_categorical(y_val, 2)

# Xây dựng mô hình ANN
ann_model = Sequential([
    Flatten(input_shape=(img_size, img_size, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

ann_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình ANN với nhiều epochs hơn
ann_model.fit(X_train, y_train_oh, epochs=20, validation_data=(X_val, y_val_oh))

# Đánh giá ANN
ann_loss, ann_acc = ann_model.evaluate(X_val, y_val_oh)
print("Accuracy ANN:", ann_acc)

# Xây dựng mô hình CNN với Dropout để giảm overfitting
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Thêm Dropout
    Dense(2, activation='softmax')
])

cnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình CNN với nhiều epochs hơn
cnn_model.fit(X_train, y_train_oh, epochs=20, validation_data=(X_val, y_val_oh))

# Đánh giá CNN
cnn_loss, cnn_acc = cnn_model.evaluate(X_val, y_val_oh)
print("Accuracy CNN:", cnn_acc)

def predict_images_in_folder(model, folder_path, model_type='cnn'):
    predictions = {}
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Error: Cannot load image at path: {img_path}")
            predictions[img_name] = "Error"
            continue
        
        # Tiền xử lý ảnh
        img = cv2.resize(img, (img_size, img_size)) / 255.0
        img = img.reshape(1, img_size, img_size, 3) if model_type in ['ann', 'cnn'] else img.reshape(1, -1)

        # Dự đoán nhãn
        if model_type == 'svm':
            prediction = model.predict(img.reshape(1, -1))
        else:
            prediction = model.predict(img)
            print(f"Raw prediction for {img_name}: {prediction}")  # In xác suất đầu ra của các lớp
            prediction = np.argmax(prediction, axis=1)
        
        # Gán nhãn cho ảnh
        label = 'Cat' if prediction == 0 else 'Dog'
        predictions[img_name] = label
    return predictions

# Đường dẫn đến thư mục chứa ảnh cần dự đoán
folder_path = 'data/input'

# Dự đoán cho tất cả ảnh trong thư mục
results = predict_images_in_folder(cnn_model, folder_path, model_type='cnn')
for img_name, label in results.items():
    print(f"Image: {img_name} - Predicted label: {label}")
