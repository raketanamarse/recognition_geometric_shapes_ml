import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import numpy as np
import os

# Параметры
image_size = (64, 64)  # Размер изображений
num_classes = 3  # Количество классов (треугольник, круг, квадрат)

# Преобразования для загрузки и предобработки изображений
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Нормализация
])

# Загрузка датасета
# Предполагаем, что у вас есть папка с изображениями, где каждая подкатегория соответствует классу
dataset = datasets.ImageFolder(root='data/train/', transform=transform)

# Разделение данных на обучающую и тестовую выборки
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Извлечение данных и меток
def extract_data_labels(dataset):
    data = []
    labels = []
    for img, label in dataset:
        data.append(img.numpy().flatten())  # Преобразуем изображение в вектор
        labels.append(label)
    return np.array(data), np.array(labels)

X_train, y_train = extract_data_labels(train_dataset)
X_test, y_test = extract_data_labels(test_dataset)

# Обучение модели kNN
k = 3  # Количество соседей
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Предсказания
y_pred = knn_model.predict(X_test)

# Оценка метрик
accuracy = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Вывод результатов
print(f'Accuracy: {accuracy:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
