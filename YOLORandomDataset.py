
import os
import shutil
from pathlib import Path
import random

def distribute_files(base_path, split_ratio=(0.7, 0.2, 0.1)):
    base_path = Path(base_path)
    images_dir = base_path / 'images'
    labels_dir = base_path / 'labels'
    
    # Создаём нужные папки
    for split in ['train', 'val', 'test']:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Получаем список всех изображений
    all_images = list(images_dir.glob('*.png'))  # или *.jpg
    random.shuffle(all_images)
    
    # Вычисляем размеры сплитов
    n = len(all_images)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    
    # Распределяем файлы
    splits = {
        'train': all_images[:n_train],
        'val': all_images[n_train:n_train + n_val],
        'test': all_images[n_train + n_val:]
    }
    
    # Перемещаем файлы
    for split_name, split_images in splits.items():
        for img_path in split_images:
            # Перемещаем изображение
            shutil.move(img_path, images_dir / split_name / img_path.name)
            
            # Перемещаем соответствующий label
            label_path = labels_dir / img_path.with_suffix('.txt').name
            if label_path.exists():
                shutil.move(label_path, labels_dir / split_name / label_path.name)
    
    print(f"Распределение завершено:")
    print(f"Train: {n_train} files")
    print(f"Val: {len(splits['val'])} files")
    print(f"Test: {len(splits['test'])} files")

# Использование
dataset_path = "dataset"  # укажи путь к папке, где лежат папки images и labels
distribute_files(dataset_path)