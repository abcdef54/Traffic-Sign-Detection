import tensorrt as trt
import numpy as np
import cv2
from ultralytics import YOLO

# SAHI imports
from sahi.prediction import ObjectPrediction
from sahi.postprocess.combine import NMSPostprocess

# 1. DEFINE THE HELPER FUNCTION
def convert_to_sahi_format(results):
    object_prediction_list = []
    for result in results:
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id]

            prediction = ObjectPrediction(
                bbox=[x1, y1, x2, y2],
                category_id=cls_id,
                category_name=cls_name,
                score=conf
            )
            object_prediction_list.append(prediction)
    return object_prediction_list

def load_model(model_path) -> YOLO | None:
    try:
        model = YOLO(model_path, task='detect')
        print(f"SUCCESS: Loaded TensorRT Engine: {model_path}")
        return model
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load engine. {e}")
        return None

def perform_inference(model: YOLO, image: np.ndarray):
    # Ultralytics handles RGB/BGR conversion internally usually, 
    # but if passing numpy, BGR is standard for OpenCV.
    
    # Run Inference with low confidence to capture all candidates
    preds = model(
        image, 
        verbose=False, 
        conf=0.1  # Low confidence ensures we don't miss anything before SAHI NMS
    )
    return preds

# --- MAIN EXECUTION ---

# 1. Load Model
model = load_model("yolo11s.engine")

# 2. Load Image
image = cv2.imread("cars.jpg")

if model and image is not None:
    # 3. Perform Inference
    # ERROR FIX: You originally called perform_inference(model) without the image
    preds = perform_inference(model, image)

    # 4. Convert Results
    object_prediction_list = convert_to_sahi_format(preds)

    print(f"Raw detections: {len(object_prediction_list)}")

    # 5. Apply SAHI NMS
    # Note: match_threshold is the IoU threshold for NMS
    nms_postprocess = NMSPostprocess(
        match_threshold=0.5, 
        # export_visual=False,
        match_metric="IOS" # Intersection over Smaller (or 'IOU')
    )
    
    # NMSPostprocess is technically a class to hold configuration. 
    # To actually RUN it on a list, usually 'batched_greedy_nms' is easier, 
    # but using the class you pass the list directly to the call:
    combined_results = nms_postprocess(object_prediction_list)

    print(f"Detections after NMS: {len(combined_results)}")
    
    # 6. (Optional) Visualize
    from sahi.utils.cv import visualize_object_predictions
    visualize_object_predictions(
        image,
        object_prediction_list=combined_results,
        rect_th=2,
        text_th=1,
        output_dir=".",
        file_name="sahi_result"
    )