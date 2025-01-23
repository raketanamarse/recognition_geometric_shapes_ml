from PIL import Image, ImageDraw
import random
import os
import math

def draw_circle_with_thick_spots(image_size, max_thickness):
    """Создает изображение с одним ровным кругом и толстыми участками на контуре."""
    # Создаем новое изображение с белым фоном
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    # Генерируем случайный радиус круга
    circle_radius = random.randint(10, 40)  # Радиус круга от 20 до 40

    # Генерируем случайную позицию для круга
    center_x = random.randint(circle_radius, image_size[0] - circle_radius)
    center_y = random.randint(circle_radius, image_size[1] - circle_radius)

    # Рисуем ровный круг
    draw.ellipse(
        (center_x - circle_radius, center_y - circle_radius,
         center_x + circle_radius, center_y + circle_radius),
        outline="black", width=1
    )

    # Добавляем толстые участки на контуре
    num_thick_spots = random.randint(20, 900)  # Случайное количество толстых участков
    for _ in range(num_thick_spots):
        # Выбираем случайный угол для толстого участка
        angle = random.uniform(0, 2 * math.pi)
        thick_radius = random.randint(1, max_thickness)  # Случайная толщина от 1 до max_thickness

        # Вычисляем координаты
        x1 = center_x + (circle_radius * math.cos(angle))
        y1 = center_y + (circle_radius * math.sin(angle))
        x2 = center_x + ((circle_radius + thick_radius) * math.cos(angle))
        y2 = center_y + ((circle_radius + thick_radius) * math.sin(angle))

        # Рисуем толстый участок
        draw.line([x1, y1, x2, y2], fill="black", width=thick_radius)

    return image

# Параметры генерации изображений
num_images = 1000
image_size = (100, 100)
output_dir = "circle"  # Папка для сохранения
max_thickness = 1  # Максимальная толщина участков

# Создаем папку, если она не существует
os.makedirs(output_dir, exist_ok=True)

for i in range(num_images):
    # Рисуем круг с толстыми участками и получаем изображение
    img = draw_circle_with_thick_spots(image_size, max_thickness)

    # Сохраняем изображение в папку circle
    img.save(os.path.join(output_dir, f"circle_{i}.png"))

    # Показываем изображение
    #img.show()
