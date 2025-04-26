import cv2
from ultralytics import YOLO

# 加載預訓練的 YOLO 模型
model = YOLO("./yolov11_cube_640/cube_640_Training2/weights/best.pt")  # 換成你使用的模型

# 開啟攝影機或影片文件
cap = cv2.VideoCapture(1)  # 0 為開啟攝影機，或替換為影片檔案路徑
# cap = cv2.VideoCapture(0)  # 0 為開啟攝影機，或替換為影片檔案路徑
mask = cv2.imread('mask.png')

color_map = {
    'red': (0, 0, 255),      # 紅色
    'blue': (255, 0, 0),     # 藍色
    'green': (0, 255, 0),    # 綠色
    'yellow': (0, 255, 255), # 黃色
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    imgRegion = cv2.bitwise_and(frame, mask)  # 透過這方式將影像結合 mask
    
    results = model.track(
        imgRegion,
        persist=True,
        stream=True,
        conf=0.5
    )
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            
            class_name = r.names[cls].lower()
            
            box_color = color_map.get(class_name,(255,255,255))
            
            label = f'{r.names[cls]} {conf:.2f} ID:{track_id}'
            
            cv2.rectangle(frame, (x1,y1),(x2,y2),box_color,2)
            cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    
    cv2.imshow('Cube Tracking',frame)
    cv2.imshow('mask',imgRegion)
                        
    
    

    # 按 'q' 鍵退出循環
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
