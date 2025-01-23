from PIL import Image, ImageDraw
import random
import math
import os

def rotate_point(point, angle, center):
    """Поворачивает точку вокруг центра на заданный угол."""
    angle_rad = math.radians(angle)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    # Сместим точку к центру
    x, y = point
    x -= center[0]
    y -= center[1]

    # Применяем вращение
    new_x = x * cos_angle - y * sin_angle + center[0]
    new_y = x * sin_angle + y * cos_angle + center[1]

    return new_x, new_y

def draw_shapes(image_size):
    # Создаем новое изображение с белым фоном
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    # Генерируем случайные размеры для фигур
    square_size = random.randint(10, 30)  # Размер квадрата от 10 до 30
    circle_radius = random.randint(5, 15)  # Радиус круга от 5 до 15
    triangle_size = random.randint(20, 40)  # Размер треугольника от 20 до 40

    # Генерируем случайные позиции для фигур
    square_pos = (random.randint(0, image_size[0] - square_size), random.randint(0, image_size[1] - square_size))
    circle_pos = (random.randint(circle_radius, image_size[0] - circle_radius), random.randint(circle_radius, image_size[1] - circle_radius))
    triangle_pos = (random.randint(0, image_size[0] - triangle_size), random.randint(0, image_size[1] - triangle_size))

    # Центры фигур
    square_center = (square_pos[0] + square_size / 2, square_pos[1] + square_size / 2)
    circle_center = circle_pos
    triangle_center = (triangle_pos[0] + triangle_size / 2, triangle_pos[1] + triangle_size)

    # Генерируем случайный угол для вращения
    angle = random.randint(0, 360)

    # Рисуем квадрат
    square_coords = [
        (square_pos[0], square_pos[1]),
        (square_pos[0] + square_size, square_pos[1]),
        (square_pos[0] + square_size, square_pos[1] + square_size),
        (square_pos[0], square_pos[1] + square_size)
    ]
    rotated_square = [rotate_point(p, angle, square_center) for p in square_coords]
    draw.polygon(rotated_square, outline="black", fill=None)

    # Рисуем круг
    circle_bbox = (
        circle_center[0] - circle_radius, circle_center[1] - circle_radius,
        circle_center[0] + circle_radius, circle_center[1] + circle_radius
    )
    draw.ellipse(circle_bbox, outline="black", fill=None)

    # Рисуем произвольный треугольник
    triangle_coords = [
        (triangle_pos[0], triangle_pos[1] + triangle_size),  # Нижняя левая точка
        (triangle_pos[0] + triangle_size, triangle_pos[1] + triangle_size),  # Нижняя правая точка
        (triangle_pos[0] + random.randint(-triangle_size // 2, triangle_size // 2), triangle_pos[1])  # Верхняя точка (случайная)
    ]
    rotated_triangle = [rotate_point(p, angle, triangle_center) for p in triangle_coords]
    draw.polygon(rotated_triangle, outline="black", fill=None)

    return image

# Параметры генерации изображений
num_images = 100
image_size = (100, 100)
output_dir = "Group_Fig"

# Создаем папку, если она не существует
os.makedirs(output_dir, exist_ok=True)

for i in range(num_images):
    # Рисуем фигуры и получаем изображение
    img = draw_shapes(image_size)

    # Сохраняем изображение в папку Group_Fig
    img.save(os.path.join(output_dir, f"shapes_{i}.png"))

    # Показываем изображение
    #img.show()
