import cv2
import numpy as np
from ultralytics import YOLO
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.predict import get_sliced_prediction
import os

class TrtYoloDetectionModel(DetectionModel):
    def load_model(self):
        try:
            self.model = YOLO(self.model_path, task='detect')
            print(f"SUCCESS: Loaded TensorRT Engine: {self.model_path}")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load engine. {e}")

    def perform_inference(self, image: np.ndarray):
        # 1. Convert RGB (SAHI) -> BGR (YOLO)
        image = np.array(image)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 2. Run Inference with low confidence to capture all candidates
        self._original_predictions = self.model(
            image_bgr, 
            verbose=False, 
            conf=0.1  # Low confidence ensures we don't miss anything before SAHI NMS
        )

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list=None,
        full_shape_list=None,
    ):
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        object_prediction_list = []

        for image_ind, result in enumerate(self._original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            boxes = result.boxes

            if boxes is None: continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                if confidence < self.confidence_threshold:
                    continue
                
                class_name = self.model.names[class_id] if self.model.names else str(class_id)

                prediction = ObjectPrediction(
                    bbox=[x1, y1, x2, y2],
                    category_id=class_id,
                    score=confidence,
                    category_name=class_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(prediction)

        return object_prediction_list

# ==========================================
# FINAL RUN CONFIGURATION
# ==========================================
if __name__ == "__main__":
    MODEL_PATH = "models/signs/untouch/yolo11s.engine" 
    IMAGE_PATH = "cars.jpg" 

    detection_model = TrtYoloDetectionModel(
        model_path=MODEL_PATH,
        confidence_threshold=0.25, 
        device="cuda:0"
    )

    print(f"Processing {IMAGE_PATH}...")
    
    # CRITICAL FIX: Using 'postprocess_type' instead of the old argument
    result = get_sliced_prediction(
        IMAGE_PATH,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_type="NMS",          # <--- The fix for your version
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.5
    )
    
    print(f"SUCCESS: Detected {len(result.object_prediction_list)} objects.")
    result.export_visuals(export_dir=".", file_name="sahi_trt_fixed")
    print("Saved result to sahi_trt_fixed.png")