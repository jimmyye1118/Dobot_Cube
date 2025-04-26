import roboflow

rf=roboflow.Roboflow(api_key="48TrqHbazjT2kcISiJpD")

project = rf.workspace().project("cube-color-gzmh4")
version = project.version(5)
model_dir = "c:/Users/jimmy/Desktop/project/yolov11_cube_640/cube_640_Training2/weights"
version.deploy("yolov11", model_dir, "best.pt")
