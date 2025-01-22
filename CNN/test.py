import torch
from PIL import Image
import torchvision.transforms as transforms

from CNN.q import SimpleCNN

# Параметры
image_size = (64, 64)  # Размер изображений
class_names = ['Circle', 'Square', 'Triangle']  # Названия классов

# Преобразования для тестовых данных
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация
])

# Загрузка модели
model = SimpleCNN()
model.load_state_dict(torch.load('shape_recognition_model.pth'))
model.eval()  # Переводим модель в режим оценки

# Функция для предсказания класса изображения
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Открываем изображение и конвертируем в RGB
    image = transform(image)  # Применяем преобразования
    image = image.unsqueeze(0)  # Добавляем размер батча

    with torch.no_grad():  # Отключаем градиенты
        outputs = model(image)  # Получаем выходные данные модели
        _, predicted = torch.max(outputs.data, 1)  # Получаем индекс класса с максимальным значением
    return predicted.item()  # Возвращаем предсказанный класс

# Путь к изображению, которое вы хотите протестировать
image_path = '../data/check/circle/98.jpg'  # Укажите путь к вашему изображению

# Получение предсказания
predicted_class_index = predict_image(image_path)
predicted_class_name = class_names[predicted_class_index]  # Получаем название класса

print(f'Predicted figure for the image: {predicted_class_name}')
