import ultralytics
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 載入預訓練模型
    model = YOLO('../Cube_Color_4_and_Defect_Model/V11_4_Color_Training2/weights/best.pt')

    # 繼續訓練 YOLOv11
    results = model.train(
        data="C:\\Users\\jimmy\\Desktop\\Dobot_Cube\\Cube_Color_4_v12_DataSet_Defect\\data.yaml",
        imgsz=640,
        epochs=100,  # 增加訓練世代數
        patience=20,  # 增加等待世代數
        batch=8,  # 增加批次大小
        project='Cube_Color_4_and_Defect_Model',
        name='V11_4_Color_Training3',
        # 數據增強
        degrees=90,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        crop_fraction=0.1,
        # 學習率調整
        lr0=0.01,
        lrf=0.01,
        # 凍結骨幹層
        freeze=10,
        # 優化器
        optimizer='AdamW'
    )