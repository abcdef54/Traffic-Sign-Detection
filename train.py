import albumentations as A
import ultralytics.data.augment
from ultralytics.data.augment import Albumentations as UAlbumentations
from ultralytics import YOLO

# --- 1. Physically Correct Augmentation Pipeline ---
def get_custom_transform(p=1.0):
    return A.Compose(
        [
            # --- SENSOR & OPTICAL REALISM ---
            # Simulates lens sharpness variation in dashcams
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.95, 1.05), p=0.15),

            # Simulates slight exposure curve shifts (not extreme)
            A.RandomGamma(gamma_limit=(85, 115), p=0.25),

            # --- NIGHT/HEADLIGHT SIMULATION ---
            # Replaces SunFlare. Simulates "sensor bloom" and "washout" from headlights.
            # We keep contrast slightly higher to simulate night lights vs dark background.
            A.RandomBrightnessContrast(
                brightness_limit=0.08, # Conservative to avoid "fake" washouts
                contrast_limit=0.12,   # Night video usually has higher contrast
                p=0.25
            ),

            # CRITICAL: Simulates High ISO Noise common in night dashcam footage
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),

            # --- MOTION & QUALITY ---
            # Simulates moving car blur (vital for side-view signs)
            A.MotionBlur(blur_limit=(3, 7), p=0.15),

            # Simulates video compression artifacts (MP4 blocking)
            A.ImageCompression(quality_lower=65, quality_upper=95, p=0.25),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            # STRICTER: If >65% of the sign is hidden/cut, don't train on it.
            # This reduces false positives on "ghost" objects.
            min_visibility=0.35 
        ),
        p=p
    )

# --- 2. Monkey Patch Class ---
class CustomAlbumentations(UAlbumentations):
    def __init__(self, p=1.0, **kwargs):
        super().__init__(p)
        self.transform = get_custom_transform(p=1.0)
        print("[INFO] ✅ Production-Grade Traffic Augmentations Injected")

# --- 3. Apply Patch ---
ultralytics.data.augment.Albumentations = CustomAlbumentations

# --- 4. Training ---
if __name__ == '__main__':
    model = YOLO("models/signs/yolo11s.pt", task="detect") 

    model.train(
        data="data.yaml",
        epochs=80,
        patience=20,

        # --- Hardware & Res ---
        imgsz=1280,      
        batch=48,        
        device=[0, 1],      
        workers=32,
        
        # 1. DISTANCE: scale=0.8 allows zooming out to make signs TINY (simulating 40m away).
        scale=0.8,       
        
        # 2. DENSITY: copy_paste creates more instances of small signs.
        copy_paste=0.3,  
        
        # 3. EDGE CASES: translate=0.2 forces signs to be learned even if 20% cut off.
        translate=0.15,   

        # --- Standard Params ---
        fliplr=0.0,      # Traffic signs are directional (Turn Left != Turn Right)
        flipud=0.0,
        mosaic=1.0,
        close_mosaic=20, # Disable heavy augmentations for the last 20 epochs
        mixup=0.1,       
        
        # Color Jitter: Kept mild to respect retro-reflective properties
        hsv_h=0.015,
        hsv_s=0.5, 
        hsv_v=0.3, 

        name="yolo11s_1280_production_tuned",
        cache="ram",
        amp=True,
    )