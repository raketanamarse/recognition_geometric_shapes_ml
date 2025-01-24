from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

def draw_fancy_box(img, box, label, color):
    # Координаты бокса
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Толщина линий и шрифта
    t = max(1, int(min(img.shape[:2])/200))
    
    # Рисуем основной бокс
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t+1)
    
    # Добавляем текст с фоном
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, t)[0]
    cv2.rectangle(img, (x1, y1-20), (x1+label_size[0], y1), color, -1)
    cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), t)
    
    return img

def create_test_graphs(detection_stats, total_time, n_images, save_folder):
    graph_folder = Path(save_folder)
    graph_folder.mkdir(exist_ok=True)
    
    # 1. График количества детекций по классам
    plt.figure(figsize=(12, 6))
    classes = list(detection_stats.keys())
    counts = [stats['count'] for stats in detection_stats.values()]
    
    bars = plt.bar(classes, counts)
    plt.title('Количество детекций по классам')
    plt.ylabel('Количество детекций')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.savefig(graph_folder / 'detections_per_class.png')
    plt.close()
    
    # 2. График confidence модели
    plt.figure(figsize=(12, 6))
    data = []
    labels = []
    for cls, stats in detection_stats.items():
        if stats['confidence']:
            data.extend(stats['confidence'])
            labels.extend([cls] * len(stats['confidence']))
    
    if data:  # Проверяем, есть ли данные для построения графика
        sns.violinplot(x=labels, y=data)
        plt.title('Распределение confidence по классам')
        plt.ylabel('confidence')
        plt.savefig(graph_folder / 'confidence_distribution.png')
    plt.close()
    
    # 3. Тепловая карта метрик
    plt.figure(figsize=(10, 8))
    metrics_data = []
    for cls, stats in detection_stats.items():
        if stats['confidence']:
            metrics_data.append([
                stats['count'],
                np.mean(stats['confidence']),
                np.min(stats['confidence']),
                np.max(stats['confidence'])
            ])
        else:
            metrics_data.append([0, 0, 0, 0])
    
    sns.heatmap(metrics_data,
                annot=True,
                fmt='.3f',
                xticklabels=['Количество', 'Ср. confidence', 'Мин. confidence', 'Макс. confidence'],
                yticklabels=classes,
                cmap='YlOrRd')
    plt.title('Метрики тестирования')
    plt.tight_layout()
    plt.savefig(graph_folder / 'test_metrics_heatmap.png')
    plt.close()

def run_test_evaluation(model_path, test_folder):
    print("🦊 Загружаю модель...")
    model = YOLO(model_path)
    
    # Проверяем пути
    test_path = Path(test_folder)
    if not test_path.exists():
        print(f"❌ Папка {test_folder} не найдена!")
        return
    
    # Получаем список изображений
    image_files = [f for f in test_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    if not image_files:
        print("❌ В тестовой папке нет изображений!")
        return
    
    print(f"\n🦊 Найдено {len(image_files)} тестовых изображений")
    
    # Создаем папки для результатов
    results_folder = Path('test_results')
    results_folder.mkdir(exist_ok=True)
    
    # Определяем цвета для классов (BGR формат)
    colors = {
        'circle': (0, 255, 0),    # Зеленый
        'square': (0, 0, 255),    # Красный
        'triangle': (255, 0, 0)   # Синий
    }
    
    class_names = ['circle', 'square', 'triangle']
    detection_stats = {cls: {'count': 0, 'confidence': []} for cls in class_names}
    total_time = 0
    
    print("\n🦊 Начинаю тестирование...")
    
    for img_path in image_files:
        print(f"\n📸 Обработка: {img_path.name}")
        
        # Читаем и обрабатываем изображение
        img = cv2.imread(str(img_path))
        original_img = img.copy()
        
        # Получаем предсказания
        start_time = time.time()
        results = model.predict(img, conf=0.25)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Обрабатываем результаты
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                class_name = class_names[cls_id]
                label = f"{class_name} {conf:.2f}"
                
                # Рисуем бокс
                img = draw_fancy_box(img, xyxy, label, colors[class_name])
                
                # Обновляем статистику
                detection_stats[class_name]['count'] += 1
                detection_stats[class_name]['confidence'].append(conf)
        
        # Добавляем время обработки
        cv2.putText(img, f"Time: {inference_time:.3f}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Сохраняем результат
        output_path = results_folder / f"detected_{img_path.name}"
        cv2.imwrite(str(output_path), img)
        
        # Показываем результат
        scale = min(1.0, 1024/max(img.shape[:2]))
        if scale < 1.0:
            display_img = cv2.resize(img, None, fx=scale, fy=scale)
        else:
            display_img = img.copy()
            
        cv2.imshow('Detection Results', display_img)
        cv2.waitKey(2000)  # Показываем на 2 секунды
        
        print(f"⏱️ Время обработки: {inference_time:.3f} сек")
        print(f"💾 Сохранено в: {output_path}")
    
    cv2.destroyAllWindows()
    
    # Создаем графики
    create_test_graphs(detection_stats, total_time, len(image_files), 'test_result_graphs')
    
    # Выводим статистику
    print("\n📊 Общая статистика:")
    print("-" * 50)
    print(f"Всего обработано изображений: {len(image_files)}")
    print(f"Среднее время обработки: {total_time/len(image_files):.3f} сек")
    
    print("\nСтатистика по классам:")
    for cls_name, stats in detection_stats.items():
        detections = stats['count']
        confidences = stats['confidence']
        
        if detections > 0:
            avg_conf = np.mean(confidences)
            print(f"\n{cls_name}:")
            print(f"  - Всего обнаружено: {detections}")
            print(f"  - Средняя confidence: {avg_conf:.3f}")
            print(f"  - Мин. confidence: {min(confidences):.3f}")
            print(f"  - Макс. confidence: {max(confidences):.3f}")

if __name__ == "__main__":
    model_path = "runs/detect/geometric_shapes/weights/best.pt"
    test_folder = "dataset/test"
    
    run_test_evaluation(model_path, test_folder)