import json
import os
from pathlib import Path
import shutil
import random
import numpy as np

def convert_coordinates_to_yolo(box_coords, img_width, img_height):
    x_min = min(coord[0] for coord in box_coords)
    x_max = max(coord[0] for coord in box_coords)
    y_min = min(coord[1] for coord in box_coords)
    y_max = max(coord[1] for coord in box_coords)
    
    # Вычисляем центр и размеры в формате YOLO (относительные координаты)
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return [x_center, y_center, width, height]

def create_yolo_dataset(source_path, output_path, split_ratio=(0.7, 0.2, 0.1)):
    # Создаем структуру папок
    output_path = Path(output_path)
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Словарь для маппинга классов
    class_map = {'circle': 0, 'square': 1, 'triangle': 2}
    
    # Получаем список всех файлов
    source_path = Path(source_path)
    image_files = list(source_path.rglob('*.png'))  # или *.jpg
    random.shuffle(image_files)
    
    # Распределяем файлы
    n = len(image_files)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    
    splits = {
        'train': image_files[:n_train],
        'val': image_files[n_train:n_train + n_val],
        'test': image_files[n_train + n_val:]
    }
    
    for split_name, files in splits.items():
        for img_path in files:
            # Путь к JSON файлу
            json_path = img_path.with_suffix('.json')
            if not json_path.exists():
                continue
                
            # Читаем JSON
            with open(json_path) as f:
                annotation = json.load(f)
            
            # Создаем YOLO аннотацию
            yolo_annotations = []
            for class_name, coords in annotation.items():
                class_id = class_map[class_name]
                box_coords = [
                    coords['top_left'],
                    coords['top_right'],
                    coords['bottom_right'],
                    coords['bottom_left']
                ]
                
                # Предполагаем размер изображения 200x200 (или получите реальный размер)
                img_width = 200  # Замените на реальный размер
                img_height = 200 # Замените на реальный размер
                
                yolo_box = convert_coordinates_to_yolo(box_coords, img_width, img_height)
                yolo_annotations.append([class_id] + yolo_box)
            
            # Копируем изображение
            dst_img_path = output_path / split_name / 'images' / img_path.name
            shutil.copy(img_path, dst_img_path)
            
            # Сохраняем YOLO аннотации
            label_path = output_path / split_name / 'labels' / img_path.with_suffix('.txt').name
            with open(label_path, 'w') as f:
                for ann in yolo_annotations:
                    f.write(f"{' '.join(map(str, ann))}\n")
    
    # Создаем yaml файл
    yaml_content = f"""
path: {str(output_path.absolute())}
train: train/images
val: val/images
test: test/images

nc: 3
names: ['circle', 'square', 'triangle']
"""
    
    with open(output_path / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"Датасет создан:")
    print(f"Train: {len(splits['train'])} files")
    print(f"Val: {len(splits['val'])} files")
    print(f"Test: {len(splits['test'])} files")

# Использование
path = "dataset"  # папка, куда будет сохранен датасет в формате YOLO
create_yolo_dataset(path, path)