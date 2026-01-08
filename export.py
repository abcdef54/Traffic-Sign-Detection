from ultralytics import YOLO
import os


def export(model_path: str, format: str = 'engine', batch: int = 1,
            half: bool = True, dynamic: bool = False, simplify: bool = True,
            imgsz: int = 1280, workspace: int = 5):
    assert os.path.exists(model_path)

    model = YOLO(model_path, 'detect')

    print("Starting Export...")

    model.export(
        format=format,
        batch=batch,
        half=half,
        dynamic=dynamic,
        device=0,
        simplify=simplify,
        workspace=workspace,
        imgsz=imgsz
    )

if __name__ == '__main__':
    export(
        model_path="runs/detect/yolo11s_1280_tuned/weights/best.pt",
        format='engine',    
        dynamic=False,      
        batch=1,
        imgsz=1280,
        half=True,
        simplify=True,
        workspace=4
    )