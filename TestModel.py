import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Параметры
img_size = 64
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Определяем такую же архитектуру модели
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
        
        self.flatten_size = 64 * 14 * 14
        
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

# Трансформации для изображений
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_multiple_shapes(image_path, model):
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    
    results = []
    # Уменьшаем шаг ещё больше для лучшей точности
    step = img_size // 16
    min_area_threshold = (img_size * 0.5) ** 2
    
    for y in range(0, image.height - img_size, step):
        for x in range(0, image.width - img_size, step):
            window = image.crop((x, y, x + img_size, y + img_size))
            # Проверяем, не является ли окно просто белым фоном
            if np.mean(np.array(window)) > 250:  # для белого фона
                continue
                
            window_tensor = transform(window).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(window_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = prob.max(1)
                
                if confidence > 0.98:  # Повышаем порог уверенности
                    results.append({
                        'class': ['circle', 'square', 'triangle'][predicted.item()],
                        'position': (x + img_size//2, y + img_size//2),
                        'confidence': confidence.item()
                    })
    
    def merge_nearby_detections(detections, threshold=30):
        merged = []
        groups = {}  # Группируем по классам
        
        # Группируем детекции по классам
        for i, det in enumerate(detections):
            cls = det['class']
            if cls not in groups:
                groups[cls] = []
            groups[cls].append(det)
        
        # Обрабатываем каждый класс отдельно
        for cls, class_dets in groups.items():
            while class_dets:
                base = class_dets.pop(0)
                current_group = [base]
                
                i = 0
                while i < len(class_dets):
                    x1, y1 = base['position']
                    x2, y2 = class_dets[i]['position']
                    
                    if abs(x1 - x2) < threshold and abs(y1 - y2) < threshold:
                        current_group.append(class_dets.pop(i))
                    else:
                        i += 1
                
                # Усредняем координаты и берём максимальную уверенность
                avg_x = sum(d['position'][0] for d in current_group) / len(current_group)
                avg_y = sum(d['position'][1] for d in current_group) / len(current_group)
                max_conf = max(d['confidence'] for d in current_group)
                
                merged.append({
                    'class': cls,
                    'position': (int(avg_x), int(avg_y)),
                    'confidence': max_conf
                })
        
        return merged
    
    filtered_results = merge_nearby_detections(results)
    return filtered_results, original_image

def show_detections(image, detections):
    draw = ImageDraw.Draw(image)
    
    # Консольный вывод
    print(f"\nОбнаружено фигур: {len(detections)}")
    print("-" * 40)
    
    for i, det in enumerate(detections, 1):
        print(f"Фигура {i}:")
        print(f"  Тип: {det['class']}")
        print(f"  Уверенность: {det['confidence']:.2%}")
        print(f"  Координаты центра: {det['position']}")
    
    # Рисование на изображении
    box_size = int(img_size * 0.8)
    for det in detections:
        x, y = det['position']
        # Рисуем квадрат относительно центра
        draw.rectangle([
            x - box_size//2,  # Левый верхний
            y - box_size//2,  # угол
            x + box_size//2,  # Правый нижний
            y + box_size//2   # угол
        ], outline='red', width=2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    print("Загрузка модели...")
    model = ConvNet().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    print(f"Используется устройство: {device}")
    
    for i in range(100):
        try:
            image_path = f'genData/Group1/shapes_{i}.png'
            print(f"\nОбработка изображения {i}...")
            
            detections, image = detect_multiple_shapes(image_path, model)
            if detections:  # Показываем только если что-то нашли
                show_detections(image, detections)
                
                # Создаём директорию если её нет
                import os
                os.makedirs('results', exist_ok=True)
                image.save(f'results/result_{i}.png')
            
        except Exception as e:
            print(f"Ошибка при обработке изображения {i}: {e}")

def visualize_training_sample(image_path):
    image = Image.open(image_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    # Показываем как модель "видит" изображение после трансформации
    transformed = transform(image).unsqueeze(0)
    recovered = transforms.ToPILImage()(transformed.squeeze(0))
    plt.figure(figsize=(5, 5))
    plt.imshow(recovered)
    plt.axis('off')
    plt.show()

def show_confidence_map(image_path, model):
    image = Image.open(image_path)
    confidence_map = np.zeros((image.height, image.width))
    class_map = np.zeros((image.height, image.width))
    
    step = img_size // 16
    for y in range(0, image.height - img_size, step):
        for x in range(0, image.width - img_size, step):
            window = image.crop((x, y, x + img_size, y + img_size))
            window_tensor = transform(window).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(window_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = prob.max(1)
                
                confidence_map[y:y+step, x:x+step] = confidence.cpu().item()
                class_map[y:y+step, x:x+step] = predicted.cpu().item()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title('Оригинал')
    plt.imshow(image)
    
    plt.subplot(132)
    plt.title('Карта уверенности')
    plt.imshow(confidence_map, cmap='hot')
    
    plt.subplot(133)
    plt.title('Карта классов')
    plt.imshow(class_map, cmap='tab10')
    
    plt.show()

if __name__ == "__main__":
    main()