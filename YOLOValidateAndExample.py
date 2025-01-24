from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import random
from pathlib import Path

def train_and_validate():
    # Создаем и обучаем модель
    model = YOLO('yolov8n.yaml')  # n - nano версия, можно использовать s/m/l/x для больших моделей
    
    # Путь к твоему yaml файлу
    yaml_path = 'dataset/dataset.yaml'
    
    # Обучаем модель
    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,  # размер изображения
        batch=16,    # размер батча
        patience=20,  # early stopping
        save=True,    # сохранять лучшие веса
        pretrained=False,
        name='geometric_shapes',  # имя для логов
        device='cuda'  # явно указываем использование GPU
    )
    
    # Валидация на тестовом наборе
    model.val(data=yaml_path)
    
    return model

def test_on_random_images(model, test_dir, num_images=5):
    """
    Визуализация работы модели на случайных изображениях
    """
    # Получаем список всех изображений
    image_files = list(Path(test_dir).glob('*.png'))  # или *.jpg
    
    # Выбираем случайные изображения
    test_images = random.sample(image_files, min(num_images, len(image_files)))
    
    plt.figure(figsize=(15, 3*num_images))
    
    for idx, img_path in enumerate(test_images):
        # Предсказание
        results = model.predict(str(img_path))
        
        # Получаем изображение с отрисованными предсказаниями
        res_plotted = results[0].plot()
        
        # Конвертируем из BGR в RGB
        res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Отображаем
        plt.subplot(num_images, 1, idx+1)
        plt.imshow(res_plotted)
        plt.title(f'Тест на {img_path.name}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    print("Начинаем обучение...")
    model = train_and_validate()
    
    print("\nТестируем на случайных изображениях...")
    test_dir = 'dataset/images'
    test_on_random_images(model, test_dir)
    
    print("\nСохраняем графики метрик...")
    metrics = model.metrics
    
    print("\nИтоговые метрики:")
    # Исправленный вывод метрик
    try:
        print(f"mAP50: {float(metrics.box.map50):.3f}")
        print(f"mAP50-95: {float(metrics.box.map):.3f}")
        print(f"Precision: {float(metrics.box.p):.3f}")
        print(f"Recall: {float(metrics.box.r):.3f}")
    except:
        print("Не удалось получить метрики. Возможно, требуется валидация модели")

if __name__ == "__main__":
    main()