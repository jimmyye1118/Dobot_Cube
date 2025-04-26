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
    model = YOLO("../yolov11_cube_640/cube_640_Training2/weights/best.pt")

    # Step 2 訓練 YOLOv8
    results = model.train(
        data="C:\\Users\\jimmy\\Desktop\\project\\Cube Color.v5i.yolov11\\data.yaml",  # 指定訓練任務檔 *.yaml
        imgsz=640,  # 輸入影像大小
        epochs=50,  # 訓練世代數
        patience=10,  # 等待世代數，無改善就提前停止訓練
        batch=2,  # 批次大小
        project='yolov11_cube_640',  # 專案名稱
        name='cube_640_v3_Training2'  # 訓練實驗名稱
    )
