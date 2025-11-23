import cv2

def display_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    
    # 减少缓冲区大小以降低延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧，尝试重新连接...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue
        
        # 显示帧
        cv2.imshow('RTSP Stream', frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    rtsp_url = "rtsp://222.20.126.228:8554/mystream"
    display_rtsp_stream(rtsp_url)