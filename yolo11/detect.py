import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('E:/yolo11/runs/train/exp/weights/best.pt') # select your model.pt path
    model.predict(source='E:/data/slab/real_data-1/images/val',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # classes=0, 是否指定检测某个类别.
                )