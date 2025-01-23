import random
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

img_size = 64
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from main_ai import ConvNet
from TestModel import detect_multiple_shapes

def test_simple_shapes(model, base_path, samples_per_class=3):
    """
    base_path - путь к папке с тремя подпапками (circle, square, triangle)
    samples_per_class - сколько случайных примеров каждого класса проверить
    """
    classes = ['circle', 'square', 'triangle']
    plt.figure(figsize=(15, 5 * samples_per_class))
    
    for class_idx, class_name in enumerate(classes):
        # Путь к папке с фигурами конкретного класса
        class_path = os.path.join(base_path, class_name)
        # Получаем список всех файлов
        files = os.listdir(class_path)
        # Выбираем случайные файлы
        selected_files = random.sample(files, samples_per_class)
        
        print(f"\nТестирование класса: {class_name}")
        print("-" * 40)
        
        for i, file in enumerate(selected_files):
            image_path = os.path.join(class_path, file)
            
            # Загружаем и обрабатываем изображение
            detections, image = detect_multiple_shapes(image_path, model)
            
            # Рисуем результат
            plt.subplot(len(classes), samples_per_class, class_idx * samples_per_class + i + 1)
            
            # Рисуем детекции
            draw = ImageDraw.Draw(image)
            box_size = int(img_size * 0.8)
            for det in detections:
                x, y = det['position']
                draw.rectangle([
                    x - box_size//2,
                    y - box_size//2,
                    x + box_size//2,
                    y + box_size//2
                ], outline='red', width=2)
            
            plt.imshow(image)
            plt.title(f"Реальный класс: {class_name}\nОбнаружено: {[d['class'] for d in detections]}")
            plt.axis('off')
            
            # Выводим информацию в консоль
            print(f"\nФайл: {file}")
            for det in detections:
                print(f"  Обнаружено: {det['class']} (уверенность: {det['confidence']:.2%})")
    
    plt.tight_layout()
    plt.show()

def main():
    print("Загрузка модели...")
    model = ConvNet().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Путь к папке с тестовыми данными
    base_path = "genData/Class"  # Укажи путь к своему датасету
    
    print("Начинаем тестирование...")
    test_simple_shapes(model, base_path, samples_per_class=5)

if __name__ == "__main__":
    main()