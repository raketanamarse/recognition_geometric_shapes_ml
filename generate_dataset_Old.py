from PIL import Image, ImageDraw, ImageFilter
import random
import math
import os
import json

num_images = 1000

def rotate_point(point, angle, center):
    """Поворачивает точку вокруг центра на заданный угол."""
    angle_rad = math.radians(angle)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    x, y = point
    x -= center[0]
    y -= center[1]

    new_x = x * cos_angle - y * sin_angle + center[0]
    new_y = x * sin_angle + y * cos_angle + center[1]

    return new_x, new_y

def shapes_overlap(shape1, shape2, threshold=0.5):
    """Проверяет, перекрываются ли две фигуры более чем на заданный процент площади."""
    x1_min, y1_min, x1_max, y1_max = shape1
    x2_min, y2_min, x2_max, y2_max = shape2

    overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    overlap_area = overlap_x * overlap_y
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    return overlap_area > 0 and overlap_area / min(area1, area2) > threshold

def apply_noise_and_blur(image, effect_type):
    """Добавляет шум или размытие к изображению."""
    if effect_type == "noise":
        noise = Image.effect_noise(image.size, random.randint(10, 50))
        image = Image.blend(image, noise.convert("RGB"), alpha=random.uniform(0.1, 0.3))
    elif effect_type == "blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
    elif effect_type == "both":
        noise = Image.effect_noise(image.size, random.randint(10, 50))
        image = Image.blend(image, noise.convert("RGB"), alpha=random.uniform(0.1, 0.3))
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
    return image

def draw_shapes(image_size, file_index):
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    shapes_data = {}
    bounding_boxes = []

    # Генерация квадратов
    while True:
        square_size = random.randint(10, 30)
        square_pos = (random.randint(0, image_size[0] - square_size), random.randint(0, image_size[1] - square_size))
        square_coords = [
            (square_pos[0], square_pos[1]),
            (square_pos[0] + square_size, square_pos[1]),
            (square_pos[0] + square_size, square_pos[1] + square_size),
            (square_pos[0], square_pos[1] + square_size)
        ]
        square_bbox = (
            square_pos[0], square_pos[1],
            square_pos[0] + square_size, square_pos[1] + square_size
        )

        if all(not shapes_overlap(square_bbox, bbox) for bbox in bounding_boxes):
            shapes_data['square'] = square_coords
            bounding_boxes.append(square_bbox)
            draw.polygon(square_coords, outline="black")
            break

    # Генерация кругов
    while True:
        circle_radius = random.randint(10, 30)
        circle_center = (random.randint(circle_radius, image_size[0] - circle_radius), random.randint(circle_radius, image_size[1] - circle_radius))
        circle_bbox = (
            circle_center[0] - circle_radius, circle_center[1] - circle_radius,
            circle_center[0] + circle_radius, circle_center[1] + circle_radius
        )

        if all(not shapes_overlap(circle_bbox, bbox) for bbox in bounding_boxes):
            shapes_data['circle'] = circle_bbox
            bounding_boxes.append(circle_bbox)
            draw.ellipse(circle_bbox, outline="black")
            break

    # Генерация треугольников
    while True:
        triangle_size = random.randint(10, 30)
        triangle_pos = (random.randint(0, image_size[0] - triangle_size), random.randint(0, image_size[1] - triangle_size))
        triangle_coords = [
            (triangle_pos[0], triangle_pos[1] + triangle_size),
            (triangle_pos[0] + triangle_size, triangle_pos[1] + triangle_size),
            (triangle_pos[0] + triangle_size // 2, triangle_pos[1])
        ]
        x_coords = [c[0] for c in triangle_coords]
        y_coords = [c[1] for c in triangle_coords]
        triangle_bbox = (
            min(x_coords), min(y_coords),
            max(x_coords), max(y_coords)
        )

        if all(not shapes_overlap(triangle_bbox, bbox) for bbox in bounding_boxes):
            shapes_data['triangle'] = triangle_coords
            bounding_boxes.append(triangle_bbox)
            draw.polygon(triangle_coords, outline="black")
            break

    # Применение различных эффектов к изображению
    effect_type = random.choice(["noise", "blur", "both"])
    image = apply_noise_and_blur(image, effect_type)

    # Сохранение изображения
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, f"{file_index}.png")
    image.save(img_path)

    # Генерация изображения с обведенными фигурами
    annotated_image = image.copy()
    annotated_draw = ImageDraw.Draw(annotated_image)
    for shape, coords in shapes_data.items():
        if shape == 'circle':
            annotated_draw.rectangle(coords, outline="red", width=2)
        else:
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            bounding_box = [
                (min(x_coords), min(y_coords)),
                (max(x_coords), max(y_coords))
            ]
            annotated_draw.rectangle(bounding_box, outline="red", width=2)

    # annotated_img_path = os.path.join(output_dir, f"{file_index}_map.png")
    # annotated_image.save(annotated_img_path)

    # Сохранение JSON файла
    json_path = os.path.join(output_dir, f"{file_index}.json")
    with open(json_path, 'w') as json_file:
        json.dump(shapes_data, json_file, indent=4)

def main():
    image_size = (200, 200)
    for i in range(1, num_images + 1):
        print(f"Генерация изображения {i} из {num_images}")
        draw_shapes(image_size, i)

if __name__ == "__main__":
    main()
