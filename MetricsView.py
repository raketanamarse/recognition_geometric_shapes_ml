from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def analyze_metrics(model_path):
    # Загружаем модель
    model = YOLO(model_path)
    
    # Запускаем валидацию и получаем результаты
    results = model.val()
    
    # Получаем метрики из результатов
    metrics = {
        'Общие метрики': {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'Precision': float(results.box.mp),
            'Recall': float(results.box.mr),
            'Accuracy': float(results.box.accuracy) if hasattr(results.box, 'accuracy') else float(results.box.map50),  # Используем mAP50 как accuracy если нет прямого атрибута
            'F1': 2 * (float(results.box.mp) * float(results.box.mr)) / (float(results.box.mp) + float(results.box.mr))  # Вычисляем F1 из precision и recall
        }
    }
    classes = ['circle', 'square', 'triangle']
    for i, cls in enumerate(classes):
        prec = float(results.box.mp_per_class[i] if hasattr(results.box, 'mp_per_class') else results.box.p[i])
        rec = float(results.box.mr_per_class[i] if hasattr(results.box, 'mr_per_class') else results.box.r[i])
    # Получаем метрики по классам
    classes = ['circle', 'square', 'triangle']
    for i, cls in enumerate(classes):
        metrics[cls] = {
            'Precision': float(results.box.mp_per_class[i] if hasattr(results.box, 'mp_per_class') else results.box.p[i]),
            'Recall': float(results.box.mr_per_class[i] if hasattr(results.box, 'mr_per_class') else results.box.r[i]),
            'mAP50': float(results.box.map50_per_class[i] if hasattr(results.box, 'map50_per_class') else results.box.ap50[i]),
            'mAP50-95': float(results.box.map_per_class[i] if hasattr(results.box, 'map_per_class') else results.box.ap[i]),
            'Accuracy': float(results.box.accuracy_per_class[i] if hasattr(results.box, 'accuracy_per_class') else results.box.ap50[i]),  # Аналогично для каждого класса
            'F1': 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0  # Вычисляем F1 для каждого класса
        }
    
    # Визуализация
    plt.figure(figsize=(12, 8))
    
    # Создаем тепловую карту
    data = []
    for cls in metrics:
        data.append([metrics[cls][m] for m in ['Precision', 'Recall', 'mAP50', 'mAP50-95']])
    
    sns.heatmap(data, 
                annot=True, 
                fmt='.3f',
                xticklabels=['Precision', 'Recall', 'mAP50', 'mAP50-95'],
                yticklabels=list(metrics.keys()),
                cmap='YlOrRd')
    
    plt.title('Метрики модели')
    plt.tight_layout()
    plt.savefig('metrics_heatmap.png')
    plt.close()
    
    # Вывод в консоль
    print("\nРезультаты оценки модели:")
    print("-" * 50)
    for cls, cls_metrics in metrics.items():
        print(f"\n{cls}:")
        for metric_name, value in cls_metrics.items():
            print(f"{metric_name}: {value:.3f}")

if __name__ == "__main__":
    model_path = 'runs/detect/geometric_shapes/weights/best.pt'
    analyze_metrics(model_path)