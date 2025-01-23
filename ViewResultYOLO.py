import os
from pathlib import Path
import random
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO

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


def load_or_train_model(weights_path='runs/detect/train/weights/best.pt', force_train=False):
    """
    Загружает существующую модель или обучает новую
    """
    if os.path.exists(weights_path) and not force_train:
        print(f"Загружаем существующую модель из {weights_path}")
        return YOLO(weights_path)
    else:
        print("Начинаем новое обучение...")
        model = YOLO('yolov8n.pt')
        model.train(data='dataset/dataset.yaml', epochs=100)
        return model

def evaluate_model(model):
    """
    Оценка результатов модели
    """
    print("\nТестируем на случайных изображениях...")
    test_dir = 'dataset/images/test'
    test_on_random_images(model, test_dir)
    
    print("\nОценка метрик модели...")
    results = model.val()  # запускаем валидацию
    
    print("\nИтоговые метрики:")
    try:
        metrics = results.box
        print(f"mAP50: {float(metrics.map50):.3f}")
        print(f"mAP50-95: {float(metrics.map):.3f}")
        print(f"Precision: {float(metrics.p):.3f}")
        print(f"Recall: {float(metrics.r):.3f}")
    except:
        print("Не удалось получить метрики")

def main():
    # Загружаем существующую модель или обучаем новую
    model = load_or_train_model()
    
    # Оцениваем результаты
    evaluate_model(model)

if __name__ == "__main__":
    main()