import cv2
from ultralytics import YOLO
import numpy as np
# 加載預訓練的 YOLO 模型
model = YOLO("./yolov11_cube_640/cube_640_Training2/weights/best.pt")  # 換成你使用的模型

# 開啟攝影機或影片文件
cap = cv2.VideoCapture(1)  # 0 為開啟攝影機，或替換為影片檔案路徑
# cap = cv2.VideoCapture(0)  # 0 為開啟攝影機，或替換為影片檔案路徑
img_mask = cv2.imread('mask2.png')

color_map = {
    'red': (0, 0, 255),      # 紅色
    'blue': (255, 0, 0),     # 藍色
    'green': (0, 255, 0),    # 綠色
    'yellow': (0, 255, 255), # 黃色
}
kernel = np.ones((5, 5), np.uint8)
while True:
    ret, cap_input = cap.read()
    if not ret:
        break
    
    # 應用預先載入的遮罩
    cap_mask = cv2.bitwise_and(cap_input, img_mask)
    
    # 轉換為HSV色彩空間以便更好地檢測黑色
    hsv = cv2.cvtColor(cap_mask, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 99])
    upper_black = np.array([255, 255, 255])  # 黑色亮度低
    # 創建黑色區域的遮罩
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # 反轉遮罩以獲取非黑色區域
    # mask_non_black = cv2.bitwise_not(mask_black)
    mask_non_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
    
    
    
    contours, _ = cv2.findContours(mask_non_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # cv2.drawContours(cap_input,cnt,-1,(128, 128, 128),3)
        edge = cv2.arcLength(cnt,True)
        vertices = cv2.approxPolyDP(cnt,edge*0.04,True)
        cornors = len(vertices)
        x,y,w,h = cv2.boundingRect(vertices)
        cv2.rectangle(cap_input,(x,y),(x+w,y+h),(128, 128, 128),3)
    
    
    cv2.imshow('cap_input',cap_input)
    cv2.imshow('mask',mask_non_black)
                        
    
    

    # 按 'q' 鍵退出循環
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
