import roboflow

rf=roboflow.Roboflow(api_key="48TrqHbazjT2kcISiJpD")

project = rf.workspace().project("cube-color-gzmh4")
version = project.version(11)
model_dir = "C:/Users/jimmy/Desktop/Dobot_Cube/Cube_Color_4_and_Defect_Model/V11_4_Color_Training2_Continue/weights"
version.deploy("yolov11", model_dir, "best.pt")
