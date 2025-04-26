import cv2 
import numpy as np
import time

from time import sleep
import DobotDllType as dType

#Dobot init
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

#Load Dll
api = dType.load()

#Connect Dobot
state = dType.ConnectDobot(api, "", 115200)[0]
print("Connect status:",CON_STR[state])
#佇列
def work(lastIndex):
    #Start to Execute Command Queued
    dType.SetQueuedCmdStartExec(api)    
    #Wait for Executing Last Command 
    while lastIndex[0] > dType.GetQueuedCmdCurrentIndex(api)[0]:
        
        dType.dSleep(100)
    dType.SetQueuedCmdClear(api)
    
    

    
#main start
if (state == dType.DobotConnect.DobotConnect_NoError):

    #Clean Command Queued
    dType.SetQueuedCmdClear(api)
    dType.SetPTPCommonParams(api, 50, 50, isQueued = 1)
    dType.SetHOMECmd(api, temp = 0, isQueued = 1)
    #print("start")
    #輸送帶速度控制
    dType.SetEMotor(api, 0, 1, 5000, 1)
    dType.SetWAITCmd(api, 3000, isQueued=1)
    dType.SetEMotor(api, 0, 1, 0, 1)
    #手臂PTP動作
    #Pose1
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, 230, 0 , 40, 0, 1)
    dType.SetEndEffectorSuctionCup(api, 1,  1, isQueued=1)
    dType.SetWAITCmd(api, 3000, isQueued=1)
    #Pose2
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, 226, -11 , 104, 0, 1)
    dType.SetWAITCmd(api, 1000, isQueued=1)
    #Pose3
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, 179, -109 , 56, 0, 1)
    dType.SetEndEffectorSuctionCup(api, 1,  0, isQueued=1)
    dType.SetWAITCmd(api, 3000, isQueued=1)
    lastIndex = dType.SetWAITCmd(api, 1, isQueued=1)
    work(lastIndex)

#Stop to Execute Command Queued
dType.SetQueuedCmdStopExec(api)
#Disconnect Dobot
dType.DisconnectDobot(api)


#dType.SetEMotor(api, 0, 1, 5000,1)  輸送帶控制
#dType.SetEndEffectorSuctionCup(api, 1,  1, isQueued=1)   吸盤控制
