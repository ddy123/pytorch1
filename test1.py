from PIL import Image
import numpy as np
def create_image_from_array():
    """从三维数组创建图片"""
    # 创建一个 100x100 的渐变图片
    height, width = 100, 100
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 创建红色渐变 (从左到右)
    for x in range(width):
        red_value = int(255 * x / width)
        image_array[:, x, 0] = red_value  # 红色通道
    
    # 创建绿色渐变 (从上到下)  
    for y in range(height):
        green_value = int(255 * y / height)
        image_array[y, :, 1] = green_value  # 绿色通道
    
    # 蓝色通道保持128
    image_array[:, :, 2] = 128
    
    # 转换为图片
    img = Image.fromarray(image_array)
    img.save("gradient_from_array.jpg")
    return img

create_image_from_array()