import cv2 
import numpy as np
import time
from ultralytics import YOLO
import DobotDllType as dType
from pygame import mixer

#Vision init
mask=None
capture=None
lastIndex = 5

#吸盤中心點調整
X_Center = 310   
Y_Center = 285   
model = YOLO("./yolov11_cube_640/cube_640_Training2/weights/best.pt")  # 換成你使用的模型  

"""
<--負--Y--正-->

^
| 正
X
| 負
V

"""
#影像編號
Video_num = 1
#亮度調整參數0.1(暗)---0.9(亮)
Gamma_Value = 0.6

#下面為不動參數
n1 = 0
color_th = 1500
color_state = "None"
state = "None"
kernel = np.ones((15, 15), np.uint8) 
capture = cv2.VideoCapture(Video_num)

color_map = {
    'red': (0, 0, 255),      # 紅色
    'blue': (255, 0, 0),     # 藍色
    'green': (0, 255, 0),    # 綠色
    'yellow': (0, 255, 255), # 黃色
}

#Dobot init
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

#Load Dll
api = dType.load()

#Connect Dobot
state = dType.ConnectDobot(api, "COM4", 115200)[0]
print("Connect status:",CON_STR[state])

#mp3播放函數

def speak(file_name):
    mixer.init()
    mixer.music.load(str(file_name) + '.mp3')
    mixer.music.play()

def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

#佇列釋放,工作執行函數
def work(lastIndex):
    #Start to Execute Command Queued
    dType.SetQueuedCmdStartExec(api)    
    #Wait for Executing Last Command 
    while lastIndex[0] > dType.GetQueuedCmdCurrentIndex(api)[0]:
        
        dType.dSleep(100)
    dType.SetQueuedCmdClear(api)  
    
# 將相機讀取到的X,Y值,及要判斷的tag_id和吸盤高度hei_z  輸入此函數
# 輸送帶便會將物件移至手臂下,並分類.    
def Dobot_work(cX, cY, tag_id, hei_z):
    #以X_center,Y_center為中心,計算相機座標系統及手臂座標系統轉換.
    if(cY-Y_Center) >= 0 :
        offy = (cY-Y_Center)*0.5001383    
    else:
        offy = (cY-Y_Center)*0.5043755    

    if(cX-X_Center) >= 0:
        offx = (X_Center-cX)*0.4921233      
    else:
        offx = (X_Center-cX)*0.5138767    
    obj_x = 268.3032+offx
    obj_y = offy
    #輸送帶移動至手臂下
    dType.SetEMotor(api, 0, 1, 12500,1)    
    dType.SetWAITCmd(api, 4850, isQueued=1)    
    dType.SetEMotor(api, 0, 1, 0,1)
    dType.SetWAITCmd(api, 100, isQueued=1)
    #手臂至影像計算後及座標轉換後obj_x,obj_y位置,吸取物件
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, obj_x, obj_y , 50, 0, 1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, obj_x, obj_y , hei_z, 0, 1)
    dType.SetEndEffectorSuctionCup(api, 1,  1, isQueued=1)

    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, obj_x, obj_y , 70, 0, 1)

    #判斷是什麼物件並給予"各個類別"放置X,Y位置.  以本案為例則是黃色藍色及紅色
    print("color_state = " + str(tag_id))
    if(tag_id == "yellow"):
       goal_x=10
       goal_y=213
    elif(tag_id == "blue"):
       goal_x=150
       goal_y=213
    elif(tag_id == "red"):
       goal_x=80
       goal_y=213
       
    elif(tag_id == "green"):
       goal_x=220
       goal_y=213
    #依類別不同,將物件放置在各個位置.
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, goal_x, -goal_y , 70, 0, 1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, goal_x, -goal_y , 40, 0, 1)
    dType.SetEndEffectorSuctionCup(api, 1,  0, isQueued=1)
    #手爪控制函數說明
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, goal_x, -goal_y , 70, 0, 1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, 270, 0 , 50, 0, 1)
    lastIndex = dType.SetWAITCmd(api, 100, isQueued=1)
    work(lastIndex)
    print("End")
    
#main start
if (state == dType.DobotConnect.DobotConnect_NoError):

    #Clean Command Queued
    dType.SetQueuedCmdClear(api)
    dType.SetPTPJointParams(api,200,200,200,200,200,200,200,200, isQueued = 1)
    dType.SetPTPCoordinateParams(api,200,200,200,200, isQueued = 1)
    dType.SetPTPCommonParams(api, 100, 100, isQueued = 1)
    
    dType.SetHOMECmd(api, temp = 0, isQueued = 1)
    lastIndex = dType.SetWAITCmd(api, 2000, isQueued=1)
    work(lastIndex)
    
flag_start_work = False
flag_debug = False
img_mask = cv2.imread("mask1.png")
# mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
while True:
    ret, cap_input = capture.read()
    if not ret:
        break
    cap_mask = cv2.bitwise_and(cap_input, img_mask)
    
    # YOLO 模型辨識
    results = model.track(cap_mask, persist=True, stream=True, conf=0.5)
    detected = False
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            
            class_name = r.names[cls].lower()
            box_color = color_map.get(class_name, (255, 255, 255))
            label = f'{r.names[cls]} {conf:.2f} ID:{track_id}'
            
            # 繪製辨識框與標籤
            cv2.rectangle(cap_input, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(cap_input, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if flag_start_work and not detected:
                
                if(class_name=='blue'):
                        color_state = "blue"
                        speak(11)
                        time.sleep(1)                                             
                elif(class_name=='yellow'):
                    color_state = "yellow"
                    speak(12)
                    time.sleep(1)
                elif(class_name=='green'):
                    color_state = "green"
                    speak(13)
                    time.sleep(1)
                elif(class_name=='red'):
                    color_state = "red"
                    speak(14)
                    time.sleep(1)
                # 計算中心點
                cX = int((x1 + x2) / 2)
                cY = int((y1 + y2) / 2)
                Dobot_work(cX, cY, class_name, 8)
                flag_start_work = False
                detected = True
    
    cv2.imshow("camera_input", cap_input)
    if flag_debug:
        cv2.imshow("camera_mask", cap_mask)
    
    keypress = cv2.waitKey(1)
    if keypress == ord('q'):
        print("Escape hit, closing...")
        cv2.destroyAllWindows()
        break
    elif keypress == ord('g'):
        flag_start_work = True
        print("GO Work")
    elif keypress == ord('L'):
        flag_debug = True
        print("GO Debug")
   

#Stop to Execute Command Queued
dType.SetQueuedCmdStopExec(api)
#ser.close()
cv2.destroyAllWindows()
#Disconnect Dobot
dType.DisconnectDobot(api)
