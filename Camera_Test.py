import cv2

for i in range(5):  # 嘗試 0-4 的索引
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"攝影機索引 {i} 可用")
        cap.release()
    else:
        print(f"攝影機索引 {i} 無法使用")