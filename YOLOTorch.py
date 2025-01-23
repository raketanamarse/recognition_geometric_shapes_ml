import json
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO



def process_dataset(dataset_path):
    """
    Обрабатывает весь датасет
    """
    dataset_path = Path(dataset_path)
    
    # Создаём директории для YOLO
    (dataset_path / 'labels').mkdir(exist_ok=True)
    (dataset_path / 'images').mkdir(exist_ok=True)
    
    # Создаём yaml файл для конфигурации
    yaml_content = """
path: {dataset_path}
train: images/train
val: images/val
test: images/test

nc: 3 
names: ['circle', 'square', 'triangle']  
    """.format(dataset_path=str(dataset_path.absolute()))
    
    with open(dataset_path / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    # Обрабатываем все JSON файлы
    for json_file in dataset_path.glob('*.json'):
        image_file = json_file.with_suffix('.png')  # или .jpg
        if not image_file.exists():
            print(f"Пропускаем {json_file}, нет соответствующего изображения")
            continue
        
        # Получаем размеры изображения
        from PIL import Image
        with Image.open(image_file) as img:
            width, height = img.size
        
        # Конвертируем разметку
        yolo_annotations = convert_json_to_yolo(json_file, width, height)
        
        # Сохраняем в формате YOLO
        label_file = dataset_path / 'labels' / json_file.with_suffix('.txt').name
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        # Копируем изображение в папку images
        import shutil
        shutil.copy(image_file, dataset_path / 'images' / image_file.name)


def convert_json_to_yolo(json_path, image_width, image_height):
    """
    Конвертирует JSON разметку в формат YOLO
    """
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    yolo_annotations = []
    
    # Обработка квадрата (4 точки)
    if "square" in annotations:
        points = np.array(annotations["square"])
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Вычисляем центр и размеры
        x_center = (x_min + x_max) / (2 * image_width)
        y_center = (y_min + y_max) / (2 * image_height)
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height
        
        yolo_annotations.append(f"1 {x_center} {y_center} {width} {height}")  # 1 - класс square
    
    # Обработка круга (x1, y1, x2, y2)
    if "circle" in annotations:
        x1, y1, x2, y2 = annotations["circle"]
        
        x_center = (x1 + x2) / (2 * image_width)
        y_center = (y1 + y2) / (2 * image_height)
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        yolo_annotations.append(f"0 {x_center} {y_center} {width} {height}")  # 0 - класс circle
    
    # Обработка треугольника (3 точки)
    if "triangle" in annotations:
        points = np.array(annotations["triangle"])
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        x_center = (x_min + x_max) / (2 * image_width)
        y_center = (y_min + y_max) / (2 * image_height)
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height
        
        yolo_annotations.append(f"2 {x_center} {y_center} {width} {height}")  # 2 - класс triangle
    
    return yolo_annotations


def main():
    dataset_path = "dataset" 
    process_dataset(dataset_path)

if (__name__ == "__main__"):
    main()