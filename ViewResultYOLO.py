
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
    print("Создаём новую модель...")
    
    # Создаём модель из yaml
    model = YOLO('yolov8n.pt')
    
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
    epochs=100,                    # Оставляем как есть
    imgsz=640,                    # Стандартный размер для YOLO
    batch=32,                     # Увеличил batch для лучшей сходимости
    patience=25,                  # Увеличил терпение для early stopping
    save=True,
    pretrained=True,             # Включил предобученные веса - так обычно лучше
    name='geometric_shapes',
    device='cuda',
    workers=4,                    # Уменьшил число workers для стабильности
    optimizer='AdamW',           # AdamW обычно работает лучше чем Adam
    lr0=0.001,                   # Уменьшил начальный learning rate для стабильности
    warmup_epochs=5,             # Увеличил warmup для лучшей инициализации
    save_period=5,               # Сохраняем чаще
    cos_lr=True,                 # Добавил косинусное затухание learning rate
    weight_decay=0.0005,         # Добавил L2 регуляризацию
    momentum=0.937,              # Оптимальное значение момента
    close_mosaic=10,             # Отключаем мозаику в конце обучения
    augment=True                 # Включаем аугментацию
    )
    
    return model

def evaluate_model(model):
    """
    Оценка метрик модели с добавлением accuracy и F1
    """
    print("\nПроверяем, как модель справляется...")
    try:
        # Проводим валидацию
        results = model.val(data='dataset/dataset.yaml')
        
        # Получаем предсказания и истинные метки
        predictions = []
        true_labels = []
        
        # Собираем все предсказания из результатов валидации
        for batch in results.pred:
            pred_classes = batch.cls.cpu().numpy()
            true_classes = batch.target.cls.cpu().numpy()
            predictions.extend(pred_classes)
            true_labels.extend(true_classes)
            
        # Рассчитываем метрики используя sklearn
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        metrics = {
            # Стандартные метрики YOLO
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            
            # Дополнительные метрики
            'accuracy': float(accuracy_score(true_labels, predictions)),
            'precision': float(precision_score(true_labels, predictions, average='weighted')),
            'recall': float(recall_score(true_labels, predictions, average='weighted')),
            'f1': float(f1_score(true_labels, predictions, average='weighted')),
            
            # Скоростные метрики
            'speed': {
                'preprocess': results.speed['preprocess'],
                'inference': results.speed['inference'],
                'postprocess': results.speed['postprocess']
            }
        }
        
        print("\nИтоговые метрики:")
        print(f"mAP50: {metrics['mAP50']:.3f}")
        print(f"mAP50-95: {metrics['mAP50-95']:.3f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-score: {metrics['f1']:.3f}")
        
        # Сохраняем метрики
        save_path = Path('runs/detect/geometric_shapes')
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Строим матрицу ошибок
        plot_confusion_matrix(true_labels, predictions, save_path)
        
        return metrics
    
    except Exception as e:
        print(f"\n*Грустно опускает ушки* Произошла ошибка при оценке метрик: {str(e)}")
        return None

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Построение матрицы ошибок
    """
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Матрица ошибок')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.savefig(save_path / 'confusion_matrix.png')
        plt.close()
    except Exception as e:
        print(f"Не удалось построить матрицу ошибок: {str(e)}")

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