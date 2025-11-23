from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO('yolov8n.pt')  # 自动下载（如果不存在）并加载

# 单张图片检测
#results = model('./rose.png')
#results[0].show()  # 显示结果
#results[0].save('output.jpg')  # 保存结果

# 使用摄像头实时检测
results = model(source=0, show=True, conf=0.5, save=False)

# 视频文件检测
#results = model('./input.mp4', save=True)