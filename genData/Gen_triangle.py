from PIL import Image, ImageDraw
import random
import os


def generate_random_triangle(image_size):
    """Генерирует координаты для треугольника, чтобы избежать коллинеарности и обеспечить большее расстояние между вершинами."""
    while True:
        # Генерируем первую вершину
        x1, y1 = random.randint(10, image_size[0] - 10), random.randint(10, image_size[1] - 10)

        # Генерируем вторую вершину, удаленную от первой
        x2, y2 = random.randint(10, image_size[0] - 10), random.randint(10, image_size[1] - 10)

        # Убедимся, что вторая точка не слишком близка к первой
        if abs(x2 - x1) < 20 and abs(y2 - y1) < 20:
            continue

        # Генерируем третью вершину, удаленную от первых двух
        x3, y3 = random.randint(10, image_size[0] - 10), random.randint(10, image_size[1] - 10)

        # Убедимся, что третья точка не слишком близка к первой и второй
        if (abs(x3 - x1) < 20 and abs(y3 - y1) < 20) or (abs(x3 - x2) < 20 and abs(y3 - y2) < 20):
            continue

        # Проверяем, чтобы не было коллинеарных точек
        area = (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if area != 0:  # Если площадь не равна нулю, точки не коллинеарны
            return (x1, y1), (x2, y2), (x3, y3)


def draw_rotated_triangle(image_size):
    """Создает изображение с одним повёрнутым треугольником."""
    # Создаем новое изображение с белым фоном
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    # Генерируем случайные координаты для треугольника
    p1, p2, p3 = generate_random_triangle(image_size)

    # Создаем изображение для треугольника
    triangle_image = Image.new("RGBA", image_size, (255, 255, 255, 0))
    triangle_draw = ImageDraw.Draw(triangle_image)

    # Рисуем треугольник
    triangle_draw.polygon([p1, p2, p3], outline="black", fill=None)

    # Генерируем случайный угол поворота
    angle = random.randint(0, 360)

    # Поворачиваем треугольник
    triangle_image = triangle_image.rotate(angle, expand=True)

    # Вычисляем позицию для размещения повёрнутого треугольника
    rotated_width, rotated_height = triangle_image.size
    position = ((image_size[0] - rotated_width) // 2, (image_size[1] - rotated_height) // 2)

    # Накладываем повёрнутый треугольник на основное изображение
    image.paste(triangle_image, position, triangle_image)

    return image


# Параметры генерации изображений
num_images = 1000
image_size = (100, 100)
output_dir = "triangle"  # Папка для сохранения

# Создаем папку, если она не существует
os.makedirs(output_dir, exist_ok=True)

for i in range(num_images):
    # Рисуем повёрнутый треугольник
    img = draw_rotated_triangle(image_size)

    # Сохраняем изображение в папку triangle
    img.save(os.path.join(output_dir, f"triangle_{i}.png"))

    # Показываем изображение (раскомментируйте, если хотите видеть каждое изображение)
    # img.show()
