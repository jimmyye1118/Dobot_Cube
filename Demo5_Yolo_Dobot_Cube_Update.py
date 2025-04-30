import cv2
import numpy as np
import base64
from flask import Flask
from flask_socketio import SocketIO
import time
from ultralytics import YOLO
import DobotDllType as dType
from pygame import mixer
import threading
import signal

app = Flask(__name__)
socketio = SocketIO(app)

# Vision init
mask = None
capture = None
lastIndex = 5

# 吸盤中心點調整
X_Center = 310
Y_Center = 285
model = YOLO("./Cube_Color_4_and_Defect_Model/V11_4_Color_Training2_Continue/weights/best.pt")

# 影像編號
Video_num = 1  # 修改為 0，測試是否正確
# 亮度調整參數 0.1(暗)---0.9(亮)
Gamma_Value = 0.6

# 下面為不動參數
n1 = 0
color_th = 1500
color_state = "None"
state = "None"
kernel = np.ones((5, 5), np.uint8)
capture = cv2.VideoCapture(Video_num)

color_map = {
    'red': (0, 0, 255),      # 紅色
    'blue': (255, 0, 0),     # 藍色
    'green': (0, 255, 0),    # 綠色
    'yellow': (0, 255, 255), # 黃色
    'broken': (141, 23, 232)     # 損毀
}

# 物件計數
object_counts = {
    'red': 0,
    'blue': 0,
    'yellow': 0,
    'green': 0,
    'broken': 0,
    'unknown': 0
}
total_objects = 0
good_rate = 0.0

# Dobot init
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError: "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}

# Load Dll
api = dType.load()

# Connect Dobot
state = dType.ConnectDobot(api, "COM4", 115200)[0]
print("Connect status:", CON_STR[state])

# 控制主迴圈運行
running = True
flag_start_work = False

# mp3 播放函數
def speak(file_name):
    mixer.init()
    mixer.music.load(str(file_name) + '.mp3')
    mixer.music.play()

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# 佇列釋放, 工作執行函數
def work(lastIndex):
    dType.SetQueuedCmdStartExec(api)
    while lastIndex[0] > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(100)
    dType.SetQueuedCmdClear(api)

# Dobot 工作函數
def Dobot_work(cX, cY, tag_id, hei_z):
    if (cY - Y_Center) >= 0:
        offy = (cY - Y_Center) * 0.5001383
    else:
        offy = (cY - Y_Center) * 0.5043755

    if (cX - X_Center) >= 0:
        offx = (X_Center - cX) * 0.4921233
    else:
        offx = (X_Center - cX) * 0.5138767
    obj_x = 268.3032 + offx
    obj_y = offy

    dType.SetEMotor(api, 0, 1, 12500, 1)
    dType.SetWAITCmd(api, 4850, isQueued=1)
    dType.SetEMotor(api, 0, 1, 0, 1)
    dType.SetWAITCmd(api, 100, isQueued=1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, obj_x, obj_y, 50, 0, 1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, obj_x, obj_y, hei_z, 0, 1)
    dType.SetEndEffectorSuctionCup(api, 1, 1, isQueued=1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, obj_x, obj_y, 70, 0, 1)

    print("color_state = " + str(tag_id))
    if tag_id == "yellow":
        goal_x = 10
        goal_y = 213
    elif tag_id == "blue":
        goal_x = 150
        goal_y = 213
    elif tag_id == "red":
        goal_x = 80
        goal_y = 213
    elif tag_id == "green":
        goal_x = 220
        goal_y = 213

    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, goal_x, -goal_y, 70, 0, 1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, goal_x, -goal_y, 40, 0, 1)
    dType.SetEndEffectorSuctionCup(api, 1, 0, isQueued=1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, goal_x, -goal_y, 70, 0, 1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, 270, 0, 50, 0, 1)
    lastIndex = dType.SetWAITCmd(api, 100, isQueued=1)
    work(lastIndex)
    print("End")

# 輸送帶運行函數
def run_conveyor():
    dType.SetEMotor(api, 0, 1, 12500, 1)
    dType.SetWAITCmd(api, 4850, isQueued=1)
    dType.SetEMotor(api, 0, 1, 0, 1)
    lastIndex = dType.SetWAITCmd(api, 100, isQueued=1)
    work(lastIndex)

# 更新物件計數並傳送到前端
def update_counts(class_name):
    global total_objects, good_rate
    object_counts[class_name] = object_counts.get(class_name, 0) + 1
    total_objects += 1
    good_objects = total_objects - object_counts.get('unknown', 0) - object_counts.get('broken', 0)
    good_rate = (good_objects / total_objects * 100) if total_objects > 0 else 0.0
    socketio.emit('object_counts', {
        'counts': object_counts,
        'total': total_objects,
        'good_rate': round(good_rate, 2)
    })

# 接收前端控制指令
@socketio.on('control')
def handle_control(data):
    global flag_start_work
    command = data.get('command')
    print(f"收到控制指令: {command}")
    if command == 'start':
        flag_start_work = True
        print("GO Work")
    elif command == 'stop':
        flag_start_work = False
        print("Finish")

# 主迴圈（非阻塞）
def main_loop():
    global running, flag_start_work
    print("主迴圈啟動")
    img_mask = cv2.imread("mask2.png")
    if img_mask is None:
        print("無法載入 mask2.png，檢查文件是否存在")
        return

    if state == dType.DobotConnect.DobotConnect_NoError:
        print("初始化 Dobot 參數")
        dType.SetQueuedCmdClear(api)
        dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=1)
        dType.SetPTPCoordinateParams(api, 200, 200, 200, 200, isQueued=1)
        dType.SetPTPCommonParams(api, 100, 100, isQueued=1)
        dType.SetHOMECmd(api, temp=0, isQueued=1)
        lastIndex = dType.SetWAITCmd(api, 2000, isQueued=1)
        work(lastIndex)

    while running:
        ret, cap_input = capture.read()
        if not ret:
            print("攝影機讀取失敗，退出主迴圈")
            break
        print("原始影像尺寸:", cap_input.shape)

        cap_mask = cv2.bitwise_and(cap_input, img_mask)
        print("遮罩後影像尺寸:", cap_mask.shape)
        hsv = cv2.cvtColor(cap_mask, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 99])
        upper_black = np.array([255, 255, 255])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_non_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask_non_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = model.track(cap_mask, persist=True, stream=True, conf=0.7)
        Model_detected_objects = []
        Unknown_detected_objects = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_name = r.names[int(box.cls[0])].lower()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                Model_detected_objects.append({
                    'class': class_name,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                })

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            edge = cv2.arcLength(contour, True)
            vertices = cv2.approxPolyDP(contour, edge * 0.04, True)
            x, y, w, h = cv2.boundingRect(vertices)
            x1, y1, x2, y2 = x, y, x + w, y + h

            is_known = False
            for obj in Model_detected_objects:
                if (x1 >= obj['bbox'][0] - 20 and x2 <= obj['bbox'][2] + 20 and
                        y1 >= obj['bbox'][1] - 20 and y2 <= obj['bbox'][3] + 20):
                    is_known = True
                    break

            if not is_known:
                Unknown_detected_objects.append({
                    'class': 'unknown',
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                })

        for obj in Model_detected_objects:
            x1, y1, x2, y2 = obj['bbox']
            box_color = color_map.get(obj['class'], (255, 255, 255))
            label = f"{obj['class']} {obj['confidence']:.2f}"
            cv2.rectangle(cap_input, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(cap_input, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for obj in Unknown_detected_objects:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(cap_input, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(cap_input, obj['class'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 傳送影像到前端
        _, buffer = cv2.imencode('.jpg', cap_input)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        print(f"傳送影像大小: {len(jpg_as_text)} 字元")
        socketio.emit('frame', {'frame': jpg_as_text})

        if flag_start_work:
            Model_detected_objects.sort(key=lambda x: x['center'][0])
            Unknown_detected_objects.sort(key=lambda x: x['center'][0])

            for obj in Model_detected_objects:
                cX, cY = obj['center']
                class_name = obj['class']
                update_counts(class_name)

                if class_name == 'blue':
                    color_state = "blue"
                    speak(11)
                    time.sleep(1)
                    Dobot_work(cX, cY, class_name, 8)
                elif class_name == 'yellow':
                    color_state = "yellow"
                    speak(12)
                    time.sleep(1)
                    Dobot_work(cX, cY, class_name, 8)
                elif class_name == 'green':
                    color_state = "green"
                    speak(13)
                    time.sleep(1)
                    Dobot_work(cX, cY, class_name, 8)
                elif class_name == 'red':
                    color_state = "red"
                    speak(14)
                    time.sleep(1)
                    Dobot_work(cX, cY, class_name, 8)
                elif class_name == 'broken':
                    speak(16)
                    time.sleep(1)
                    run_conveyor()
                    time.sleep(4)
                time.sleep(1)

            for obj in Unknown_detected_objects:
                update_counts('unknown')
                print("檢測到異物，運行輸送帶")
                speak(15)
                time.sleep(1)
                run_conveyor()
                time.sleep(5)

        cv2.imshow("camera_input", cap_input)

        socketio.sleep(0.1)  # 控制 WebSocket 傳輸頻率

    # 清理
    cleanup()

# 清理函數
def cleanup():
    global running
    running = False
    dType.SetQueuedCmdStopExec(api)
    cv2.destroyAllWindows()
    if capture:
        capture.release()
    dType.DisconnectDobot(api)
    print("程式已清理並結束")

# 處理程式終止信號
def signal_handler(sig, frame):
    cleanup()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

# 啟動主迴圈
@socketio.on('connect')
def on_connect():
    print("WebSocket 客戶端已連線")
    global running
    running = True
    threading.Thread(target=main_loop, daemon=True).start()

@socketio.on('disconnect')
def on_disconnect():
    print("WebSocket 客戶端已斷線")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)