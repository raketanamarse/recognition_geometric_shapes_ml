from tkinter import Image
from sympy import Range
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from PIL import ImageDraw, Image
from tqdm import tqdm

# Параметры
img_size = 64
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2} MB")

# Определение модели
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Вычисляем размер после свертки
        self.flatten_size = 64 * 14 * 14  # Может потребоваться корректировка
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Подготовка данных
transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

dataset = datasets.ImageFolder('genData/Class', transform=transform)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Используем tqdm для прогресс-бара
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(loader), 100.*correct/total

# Функция для валидации
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss/len(loader), 100.*correct/total



def predict_image(model, image_path):
    model.eval()
    # Загружаем и преобразуем изображение
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = output.max(1)
        
    return dataset.classes[predicted.item()]

def detect_multiple_shapes(model, image_path):
    # Загружаем изображение
    image = Image.open(image_path)
    original_image = image.copy()
    
    # Находим все фигуры
    results = []
    # Разбиваем изображение на части
    for y in range(0, image.height - img_size, img_size//2):
        for x in range(0, image.width - img_size, img_size//2):
            # Вырезаем участок
            window = image.crop((x, y, x + img_size, y + img_size))
            # Преобразуем как для обучения
            window_tensor = transform(window).unsqueeze(0).to(device)
            
            # Получаем предсказание
            with torch.no_grad():
                output = model(window_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = prob.max(1)
                
                # Если модель уверена, что это фигура
                if confidence > 0.7:  # Порог уверенности можно настроить
                    results.append({
                        'class': dataset.classes[predicted.item()],
                        'position': (x, y),
                        'confidence': confidence.item()
                    })
    
    return results, original_image

def show_detections(image, detections):
    draw = ImageDraw.Draw(image)
    for det in detections:
        x, y = det['position']
        text = f"{det['class']}: {det['confidence']:.2f}"
        draw.rectangle([x, y, x + img_size, y + img_size], outline='red', width=2)
        draw.text((x, y-15), text, fill='red')
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():

    # Разделяем датасет на train и validation
    train_size = int(0.8 * len(dataset))  # 80% для обучения
    val_size = len(dataset) - train_size   # 20% для валидации

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Создаем загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Инициализация модели, оптимизатора и функции потерь
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
        # Обучение модели
    epochs = 1000
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Валидация
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch: {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break
        
    # Визуализация результатов
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

if __name__ == "__main__":
    main()