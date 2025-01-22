import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')  # Или другой бэкенд, например 'Qt5Agg'
import matplotlib.pyplot as plt  # Импортируем библиотеку для построения графиков

# Параметры
image_size = (64, 64)  # Размер изображений
batch_size = 32
num_classes = 3  # Количество классов (например, круги, квадраты, треугольники)
num_epochs = 100
learning_rate = 0.001

# Путь к папке с изображениями
dataset_path = '../data/train/'  # Укажите путь к вашему набору данных

# Преобразования для данных с аугментацией
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),  # Случайное горизонтальное отражение
    transforms.RandomRotation(10),      # Случайное вращение на 10 градусов
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация
])

# Загрузка данных
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Разделение на обучающую и валидационную выборки
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Создание загрузчиков данных
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Определение модели с Dropout
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (image_size[0] // 4) * (image_size[1] // 4), 128)
        self.dropout = nn.Dropout(0.5)  # Dropout с вероятностью 50%
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * (image_size[0] // 4) * (image_size[1] // 4))  # Преобразование для полносвязного слоя
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Применение Dropout
        x = self.fc2(x)
        return x

# Инициализация модели, потерь и оптимизатора
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2-регуляризация

# Списки для хранения значений потерь и точности
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Переменные для ранней остановки
best_val_loss = float('inf')
patience = 5  # Количество эпох без улучшения, после которых остановим обучение
counter = 0

# Обучение модели
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()  # Обнуление градиентов
        outputs = model(images)  # Прямой проход
        loss = criterion(outputs, labels)  # Вычисление потерь
        loss.backward()  # Обратный проход
        optimizer.step()  # Обновление параметров
        running_loss += loss.item()

        # Подсчет точности
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Сохранение средней потери и точности для текущей эпохи
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_accuracy = 100 * correct / total
    train_accuracies.append(train_accuracy)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

    # Валидация
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():  # Отключаем градиенты для валидации
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            # Подсчет точности
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # Сохранение средней потери и точности для валидации
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = 100 * correct_val / total_val
    val_accuracies.append(val_accuracy)
    print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Ранняя остановка
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0  # Сброс счетчика
        # Сохранение лучшей модели
        torch.save(model.state_dict(), 'best_shape_recognition_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# Создание графика для Train и Validation Accuracy и Loss
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

# График точности
plt.subplot(1, 2, 1)  # 1 строка, 2 столбца, 1-й график
plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o')
plt.title('Train vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# График потерь
plt.subplot(1, 2, 2)  # 1 строка, 2 столбца, 2-й график
plt.plot(epochs, train_losses, label='Train Loss', marker='o', color='red')
plt.plot(epochs, val_losses, label='Validation Loss', marker='o', color='orange')
plt.title('Train vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Показать графики
plt.tight_layout()
plt.show()

# Сохранение модели
torch.save(model.state_dict(), 'shape_recognition_model.pth')
