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
model = YOLO("./Cube_Color_4_and_Defect_Model/V11_4_Color_Training2_Continue/weights/best.pt")  # 換成你使用的模型  

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
kernel = np.ones((5, 5), np.uint8) 
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


# 輸送帶運行函數（僅運行，不夾取）
def run_conveyor():
    dType.SetEMotor(api, 0, 1, 12500, 1)
    dType.SetWAITCmd(api, 4850, isQueued=1)
    dType.SetEMotor(api, 0, 1, 0, 1)
    lastIndex = dType.SetWAITCmd(api, 100, isQueued=1)
    work(lastIndex)

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
img_mask = cv2.imread("mask2.png")
# mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
while True:
    ret, cap_input = capture.read()
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
    
    # 先進行YOLO檢測，獲取所有檢測到的物體
    results = model.track(cap_mask,persist=True, stream=True, conf=0.5)
    
    # 創建一個列表來存儲檢測到的Model物體信息
    Model_detected_objects = []
    
    # 創建一個列表來存儲檢測到的未知物體信息
    Unknown_detected_objects = []
    
    # 處理Yolo辨識的結果
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_name = r.names[int(box.cls[0])].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            # 儲存每個符合Model內的物件
            Model_detected_objects.append({
                'class': class_name,
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'center': ((x1+x2)//2, (y1+y2)//2)
            })  
    
    # 計算所有檢測到的物體
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # 過濾小噪點
            continue
            
        # 計算輪廓的邊界框
        edge = cv2.arcLength(contour,True)
        vertices = cv2.approxPolyDP(contour,edge*0.04,True)
        cornors = len(vertices)
        x,y,w,h = cv2.boundingRect(vertices)
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        # 檢查這個輪廓是否已經被YOLO檢測到
        is_known = False
        for obj in Model_detected_objects:
            if (x1 >= obj['bbox'][0]-20 and x2 <= obj['bbox'][2]+20 and
                y1 >= obj['bbox'][1] -20 and y2 <= obj['bbox'][3] + 20):
                is_known = True
                break
            
        # 儲存每個未知的物件    
        if not is_known:
            Unknown_detected_objects.append({
                'class': 'Unknown',
                'bbox': (x1, y1, x2, y2),
                'center': ((x1+x2)//2, (y1+y2)//2)
            })
    
    # 繪製所有Model內的物體
    for obj in Model_detected_objects:
        x1, y1, x2, y2 = obj['bbox']
        box_color = color_map.get(obj['class'], (255, 255, 255))
        label = f"{obj['class']} {obj['confidence']:.2f}"
        cv2.rectangle(cap_input, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(cap_input, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)        

    # 繪製所有未知的物體
    for obj in Unknown_detected_objects:
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(cap_input, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(cap_input, obj['class'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 
            
        
        
    # 執行夾取操作
    if flag_start_work:
        
        # 按照某種順序處理檢測到的物體（例如從左到右）
        Model_detected_objects.sort(key=lambda x: x['center'][0])  # 按x坐標排序
        Unknown_detected_objects.sort(key=lambda x: x['center'][0])  # 按x坐標排序
        
        # 夾取所有Model內的物件
        for obj in Model_detected_objects:
            cX, cY = obj['center']
            class_name = obj['class']
            
            if class_name == 'blue':
                color_state = "blue"
                speak(11)
            elif class_name == 'yellow':
                color_state = "yellow"
                speak(12)
            elif class_name == 'green':
                color_state = "green"
                speak(13)
            elif class_name == 'red':
                color_state = "red"
                speak(14)
                
            time.sleep(1)
            Dobot_work(cX, cY, class_name, 8)
            time.sleep(1)  # 夾取間隔
            # flag_start_work = False
        
        # 處理未知物件，不做夾取
        for obj in Unknown_detected_objects:
            cX, cY = obj['center']    
            print("檢測到異物，運行輸送帶")
            speak(15)
            time.sleep(1)
            run_conveyor()
            time.sleep(5)
            # flag_start_work = False 一步一步夾取
    
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
    elif keypress == ord('c'):
        flag_start_work = False
        print("Finish")
   

#Stop to Execute Command Queued
dType.SetQueuedCmdStopExec(api)

#ser.close()
cv2.destroyAllWindows()
#Disconnect Dobot
dType.DisconnectDobot(api)
