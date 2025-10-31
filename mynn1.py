import cv2
import numpy as np
import random

def opencv_random_effects(input_path, output_path):
    """使用OpenCV应用随机特效"""
    
    img = cv2.imread(input_path)
    
    # 随机选择特效
    effect_type = random.randint(0, 5)
    
    if effect_type == 0:
        # 油画效果
        result = cv2.xphoto.oilPainting(img, 7, 1)
    elif effect_type == 1:
        # 边缘保留滤波
        result = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
    elif effect_type == 2:
        # 风格化滤波
        result = cv2.stylization(img, sigma_s=60, sigma_r=0.07)
    elif effect_type == 3:
        # 细节增强
        result = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    elif effect_type == 4:
        # 铅笔素描效果
        _, result = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    else:
        # 添加随机噪声
        noise = np.random.randint(-50, 50, img.shape, dtype='int16')
        result = np.clip(img.astype('int16') + noise, 0, 255).astype('uint8')
    
    cv2.imwrite(output_path, result)
    return result

# 使用示例
opencv_random_effects("./cat.jpeg", "opencv_random_output.jpg")