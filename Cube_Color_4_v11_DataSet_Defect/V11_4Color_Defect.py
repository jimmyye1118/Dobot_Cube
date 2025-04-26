# 安裝 YOLOv8 相關套件
# pip uninstall pip setuptools
# pip3 install --upgrade pip
# pip3 install --upgrade setuptools
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# git clone https://github.com/ultralytics/ultralytics
# cd ultralytics
# pip install ultralytics

import ultralytics
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Step 1 載入預訓練模型 yolov8s.pt
    model = YOLO('../Cube_Color_4_and_Defect_Model/V11_4_Color_Training2/weights/best.pt')

    # Step 2 訓練 YOLOv8
    results = model.train(
        data="C:\\Users\\jimmy\\Desktop\\Dobot_Cube_Tracking_Project\\Cube_Color_4_v11_DataSet_Defect\\data.yaml",  # 指定訓練任務檔 *.yaml
        imgsz=640,  # 輸入影像大小
        epochs=50,  # 訓練世代數
        patience=10,  # 等待世代數，無改善就提前停止訓練
        batch=4,  # 批次大小
        project='Cube_Color_4_and_Defect_Model',  # 專案名稱
        name='V11_4_Color_Training3_Continue'  # 訓練實驗名稱
    )
