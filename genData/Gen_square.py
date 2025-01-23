from PIL import Image, ImageDraw
import random
import os

def draw_square_with_thick_spots(image_size, max_thickness):
    """Создает изображение с одним квадратом и толстыми участками на контуре."""
    # Генерируем случайный размер квадрата
    square_size = random.randint(10, 40)  # Размер квадрата от 10 до 40

    # Генерируем случайную позицию для квадрата
    # Убедимся, что квадрат не выходит за границы
    max_x = image_size[0] - square_size
    max_y = image_size[1] - square_size

    top_left_x = random.randint(0, max_x)
    top_left_y = random.randint(0, max_y)

    # Создаем изображение для квадрата
    square_image = Image.new("RGBA", image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(square_image)

    # Рисуем квадрат
    draw.rectangle(
        [top_left_x, top_left_y, top_left_x + square_size, top_left_y + square_size],
        outline="black", fill=None
    )

    # Добавляем толстые участки на контуре
    num_thick_spots = random.randint(5, 20)  # Случайное количество толстых участков
    for _ in range(num_thick_spots):
        # Выбираем случайное место на контуре квадрата
        side = random.randint(0, 3)  # 0: верх, 1: право, 2: низ, 3: лево
        thick_radius = random.randint(1, max_thickness)  # Случайная толщина от 1 до max_thickness

        # Определяем координаты для толстого участка в зависимости от стороны
        if side == 0:  # Верх
            x1 = random.uniform(top_left_x, top_left_x + square_size)
            y1 = top_left_y
            y2 = y1 - thick_radius
            draw.line([x1, y1, x1, y2], fill="black", width=thick_radius)

        elif side == 1:  # Право
            x1 = top_left_x + square_size
            y1 = random.uniform(top_left_y, top_left_y + square_size)
            x2 = x1 + thick_radius
            draw.line([x1, y1, x2, y1], fill="black", width=thick_radius)

        elif side == 2:  # Низ
            x1 = random.uniform(top_left_x, top_left_x + square_size)
            y1 = top_left_y + square_size
            y2 = y1 + thick_radius
            draw.line([x1, y1, x1, y2], fill="black", width=thick_radius)

        elif side == 3:  # Лево
            x1 = top_left_x
            y1 = random.uniform(top_left_y, top_left_y + square_size)
            x2 = x1 - thick_radius
            draw.line([x1, y1, x2, y1], fill="black", width=thick_radius)

    return square_image

def rotate_square(square_image, angle):
    """Поворачивает квадрат на заданный угол."""
    return square_image.rotate(angle, expand=True)

# Параметры генерации изображений
num_images = 1000
image_size = (100, 100)
output_dir = "square"  # Папка для сохранения
max_thickness = 1  # Максимальная толщина участков

# Создаем папку, если она не существует
os.makedirs(output_dir, exist_ok=True)

for i in range(num_images):
    # Рисуем квадрат с толстыми участками
    square_img = draw_square_with_thick_spots(image_size, max_thickness)

    # Поворачиваем квадрат на случайный угол
    angle = random.randint(0, 360)  # Случайный угол от 0 до 360 градусов
    rotated_square = rotate_square(square_img, angle)

    # Создаем новое изображение с белым фоном
    final_image = Image.new("RGBA", image_size, (255, 255, 255, 255))

    # Находим координаты для вставки повернутого квадрата
    offset_x = (final_image.width - rotated_square.width) // 2
    offset_y = (final_image.height - rotated_square.height) // 2

    # Вставляем повернутый квадрат на белый фон
    final_image.paste(rotated_square, (offset_x, offset_y), rotated_square)

    # Сохраняем изображение в папку square
    final_image.save(os.path.join(output_dir, f"square_{i}.png"))

