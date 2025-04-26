import cv2 
import numpy as np
import time

import DobotDllType as dType
from pygame import mixer

#Vision init
mask=None
capture=None
lastIndex = 5

#吸盤中心點調整
X_Center = 310   
Y_Center = 285     

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

#Color_Ranger

Blue_lower = np.array([102, 201, 73])
Blue_upper = np.array([220, 255, 255])
Green_lower = np.array([66, 54, 57])
Green_upper = np.array([97, 255, 137])
Yellow_lower = np.array([9, 66, 29])
Yellow_upper = np.array([38, 255, 255])
Red_lower1 = np.array([0, 102, 21])
Red_upper1 = np.array([7, 255, 255])
Red_lower2 = np.array([156,43,46])
Red_upper2 = np.array([180,255,255])

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
    if(tag_id == "Yellow"):
       goal_x=10
       goal_y=213
    elif(tag_id == "Blue"):
       goal_x=150
       goal_y=213
    elif(tag_id == "Red"):
       goal_x=80
       goal_y=213
       
    elif(tag_id == "Green"):
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
mask_img = cv2.imread("mask.png")
# mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
while(True):
    #Vision Start
    ret, cap_input = capture.read()
    cap_copy = cap_input.copy()
    cap_mask = cv2.bitwise_and(cap_input,mask_img)
    
    cap_blur = cv2.GaussianBlur(cap_mask , (5,5), 0)
   
    # gamma = Gamma_Value                                   # change the value here to get different result
    # cap_gamma = adjust_gamma(cap_blur, gamma=gamma)

    # hsv = cv2.cvtColor(cap_gamma,cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(cap_blur,cv2.COLOR_BGR2HSV) 
    Blue_mask = cv2.inRange(hsv, Blue_lower, Blue_upper)
    Green_mask = cv2.inRange(hsv, Green_lower, Green_upper)
    Yellow_mask = cv2.inRange(hsv, Yellow_lower, Yellow_upper)
    Red_mask1 = cv2.inRange(hsv, Red_lower1, Red_upper1)
    Red_mask2 = cv2.inRange(hsv, Red_lower2, Red_upper2)
    Red_mask = Red_mask1 + Red_mask2

    mask = Yellow_mask + Blue_mask + Red_mask + Green_mask
    
    mask = cv2.erode(mask,kernel,iterations=1)
    mask = cv2.dilate(mask,kernel,iterations=1)
    
    c1 = cv2.countNonZero(Blue_mask)
    c2 = cv2.countNonZero(Yellow_mask)
    c3 = cv2.countNonZero(Green_mask)
    c4 = cv2.countNonZero(Red_mask)

    if (flag_start_work == True):
        n1=n1+1      
        if(True):
            n1=0
            if((c1 > c2) and (c1 > c4) and (c1>c3)):
                color_state = "Blue"
                speak(11)
                time.sleep(1)                                             
            elif((c2 > c1) and (c2 > c4) and (c2 > c3)):
                color_state = "Yellow"
                speak(12)
                time.sleep(1)
            elif((c3 > c1) and (c3 > c4) and (c3 > c2)):
                color_state = "Green"
                speak(13)
                time.sleep(1)
            elif((c4 > c1) and (c4 > c2) and (c4>c3)):
                color_state = "Red"
                speak(14)
                time.sleep(1)
            
            contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            if(len(contours) > 0):
                contour = max(contours, key = cv2.contourArea)
                M = cv2.moments(contour)
                (x, y, w, h) = cv2.boundingRect(contour)
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                #(X,Y,類別,Z軸高度)
                Dobot_work(cX, cY, color_state, 8)
                flag_start_work = False
                n1=0
                color_state = "None"           
    else:
        n1=0
        color_state = "None"        
    

    cv2.imshow("camera_input",cap_input)
    keypress = cv2.waitKey(1)
    if (flag_debug == True):
        cv2.imshow("camera_cut",cap_copy)
        # cv2.imshow("camera" ,cap_gamma)    
        cv2.imshow("camera_mask" ,mask) 
    
    
    if keypress == ord('q'):
        # ESC pressed
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
