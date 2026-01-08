import albumentations as A
import ultralytics.data.augment
from ultralytics.data.augment import Albumentations as UAlbumentations
from ultralytics import YOLO

def get_custom_transform(p=1.0):
    return A.Compose(
        [
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.95, 1.05), p=0.15),


            A.RandomGamma(gamma_limit=(85, 115), p=0.25),


            A.RandomBrightnessContrast(
                brightness_limit=0.08,
                contrast_limit=0.12,   
                p=0.25
            ),


            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),


            A.MotionBlur(blur_limit=(3, 7), p=0.15),


            A.ImageCompression(quality_lower=65, quality_upper=95, p=0.25),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.35 
        ),
        p=p
    )


class CustomAlbumentations(UAlbumentations):
    def __init__(self, p=1.0, **kwargs):
        super().__init__(p)
        self.transform = get_custom_transform(p=1.0)
        print("[INFO] ✅ Production-Grade Traffic Augmentations Injected")


ultralytics.data.augment.Albumentations = CustomAlbumentations


if __name__ == '__main__':
    model = YOLO("models/signs/yolo11s.pt", task="detect") 

    model.train(
        data="data.yaml",
        epochs=80,
        patience=20,


        imgsz=1280,      
        batch=48,        
        device=[0, 1],      
        workers=32,
        

        scale=0.8,       
        
        copy_paste=0.3,  
        

        translate=0.15,   


        fliplr=0.0,     
        flipud=0.0,
        mosaic=1.0,
        close_mosaic=20,
        mixup=0.1,       
        

        hsv_h=0.015,
        hsv_s=0.5, 
        hsv_v=0.3, 

        name="yolo11s_1280_production_tuned",
        cache="ram",
        amp=True,
    )