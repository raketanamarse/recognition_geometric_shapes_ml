import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Загрузка модели
model = load_model('shape_classifier.h5')

# Директория с тестовыми изображениями
test_dir = 'data/check'

# Размер изображений, который использовался для обучения модели
img_size = (64, 64)

# Названия классов (в том же порядке, что и при обучении)
class_names = ['circle', 'square', 'triangle']

# Функция для предсказания класса изображения
def predict_image(model, img_path):
    try:
        # Загрузка изображения
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0  # Нормализация
        img_array = np.expand_dims(img_array, axis=0)  # Добавление измерения батча

        # Предсказание
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)  # Уверенность модели в предсказании
        return predicted_class, confidence
    except Exception as e:
        return f"Error processing {img_path}: {e}", 0

# Обработка изображений из директории
if not os.path.exists(test_dir):
    print(f"Directory '{test_dir}' not found!")
else:
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.jpg', '.png')):  # Проверка формата файла
            img_path = os.path.join(test_dir, filename)
            predicted_class, confidence = predict_image(model, img_path)
            print(f"Файл: {filename} -> Класс: {predicted_class}, Уверенность: {confidence:.2f}")
        else:
            print(f"Файл {filename} пропущен (не поддерживаемый формат).")
