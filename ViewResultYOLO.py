
import os
from pathlib import Path
import json
import random
import cv2
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO

def create_and_train_model():
    """
    Создание и обучение новой модели
    """
    print("*Фыр* Создаём новую модель...")
    
    # Создаём модель из yaml
    model = YOLO('yolov8n.yaml')
    
    # Загружаем конфигурацию датасета
    with open('dataset/dataset.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)
    num_classes = len(dataset_config['names'])
    print(f"Классы в датасете: {dataset_config['names']}")
    
    # Настраиваем количество классов
    model.model.model[-1].nc = num_classes
    
    # Обучаем модель
    results = model.train(
        data='dataset/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        save=True,
        pretrained=False,
        name='geometric_shapes',
        device='cuda',
        workers=8,
        optimizer='Adam',
        lr0=0.01,
        warmup_epochs=3,
        save_period=10
    )
    
    return model

def evaluate_model(model):
    """
    Оценка метрик модели
    """
    print("\nПроверяем, как модель справляется...")
    results = model.val(data='dataset/dataset.yaml')
    
    
    metrics = {
        'mAP50': float(results.box.map50),
        'mAP50-95': float(results.box.map),
        'precision': float(results.box.p),
        'recall': float(results.box.r),
        'speed': {
            'preprocess': results.speed['preprocess'],
            'inference': results.speed['inference'],
            'postprocess': results.speed['postprocess']
        }
    }
    
    print("\nИтоговые метрики:")
    print(f"mAP50: {metrics['mAP50']:.3f}")
    print(f"mAP50-95: {metrics['mAP50-95']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    
    # Сохраняем метрики
    with open('runs/detect/geometric_shapes/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def test_on_random_images(model, test_dir='dataset/test', num_images=5):
    """
    Тестирование на случайных изображениях
    """
    print("\nТестируем на случайных изображениях...")
    
    image_files = list(Path(test_dir).glob('*.png'))
    if not image_files:
        print("Не нашли изображений для теста :(")
        return
        
    test_images = random.sample(image_files, min(num_images, len(image_files)))
    
    plt.figure(figsize=(15, 3*num_images))
    
    for idx, img_path in enumerate(test_images):
        # Предсказание с повышенным порогом уверенности
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            verbose=False
        )
        
        # Информация о предсказаниях
        boxes = results[0].boxes
        print(f"\nРезультаты для {img_path.name}:")
        for box in boxes:
            conf = float(box.conf)
            cls = int(box.cls)
            print(f"Класс: {cls}, Уверенность: {conf:.2f}")
        
        # Визуализация
        res_plotted = results[0].plot()
        res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        plt.subplot(num_images, 1, idx+1)
        plt.imshow(res_plotted)
        plt.title(f'Результат для {img_path.name}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('runs/detect/geometric_shapes/test_results.png')
    plt.show()

def plot_metrics(metrics):
    """
    Визуализация метрик
    """
    labels = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
    values = [metrics['mAP50'], metrics['mAP50-95'], 
              metrics['precision'], metrics['recall']]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title('Метрики модели')
    plt.ylabel('Значение')
    plt.ylim(0, 1)
    plt.savefig('runs/detect/geometric_shapes/metrics_plot.png')
    plt.close()

def main():
    # Создаём и обучаем модель
    model = create_and_train_model()
    
    # Оцениваем результаты
    metrics = evaluate_model(model)
    
    # Визуализируем метрики
    plot_metrics(metrics)
    
    # Тестируем на случайных изображениях
    test_on_random_images(model)
    
    print("\n*Радостно виляет хвостом* Обучение и тестирование завершено!")

if __name__ == "__main__":
    main()