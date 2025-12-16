import tensorrt as trt
import numpy as np
import cv2
from ultralytics import YOLO

# SAHI imports
from sahi.prediction import ObjectPrediction
from sahi.postprocess.combine import NMSPostprocess
from sahi.slicing import slice_image

# 1. HELPER: Convert YOLO Results -> SAHI Objects (With Offset Support)
def convert_to_sahi_format(results, shift_amount=[0, 0]):
    """
    results: List of Ultralytics Results
    shift_amount: [x_shift, y_shift] to map slice coordinates back to full image
    """
    object_prediction_list = []
    
    shift_x, shift_y = shift_amount

    for result in results:
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id] if names else str(cls_id)

            # --- CRITICAL: SHIFT COORDINATES ---
            # Map the box from the "slice" world to the "original image" world
            x1 += shift_x
            x2 += shift_x
            y1 += shift_y
            y2 += shift_y

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
    # Run Inference with low confidence to capture all candidates
    preds = model(
        image, 
        verbose=False, 
        conf=0.15 # Slightly higher conf for slices to reduce noise
    )
    return preds

# --- MAIN EXECUTION ---

# 1. Load Model
model = load_model("yolo11s.engine")

# 2. Load Image (OpenCV loads as BGR)
image_path = "cars.jpg"
image = cv2.imread(image_path)

if model and image is not None:
    all_predictions = []

    # --- STEP 3A: STANDARD INFERENCE (Optional but recommended) ---
    # Detects large objects that might be larger than a slice
    print("Running standard inference...")
    preds_full = perform_inference(model, image)
    full_preds_sahi = convert_to_sahi_format(preds_full, shift_amount=[0, 0])
    all_predictions.extend(full_preds_sahi)

    # --- STEP 3B: SLICED INFERENCE ---
    print("Running sliced inference...")
    
    # SAHI slice_image expects RGB usually, but if we just slice numpy arrays 
    # and pass to YOLO (which handles BGR/RGB), it's fine. 
    # However, SAHI slice_image returns slices in the same format as input.
    
    slice_result = slice_image(
        image=image,
        slice_height=512, # Adjust based on your object size
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # Iterate over every slice
    for i, slice_img in enumerate(slice_result.images):
        # slice_result.starting_pixels contains [x_min, y_min] for this slice
        start_pixel = slice_result.starting_pixels[i] # e.g. [0, 512]
        
        # Perform inference on the small slice
        preds_slice = perform_inference(model, slice_img)
        
        # Convert and Offset
        # We pass start_pixel as the shift amount
        slice_preds_sahi = convert_to_sahi_format(preds_slice, shift_amount=start_pixel)
        
        all_predictions.extend(slice_preds_sahi)
        print(f" - Slice {i+1}/{len(slice_result.images)}: Found {len(slice_preds_sahi)} objects")

    print(f"Total raw detections (Standard + Sliced): {len(all_predictions)}")

    # --- STEP 4: APPLY NMS ---
    # Merge overlapping boxes from the standard pass and the slice passes
    nms_postprocess = NMSPostprocess(
        match_threshold=0.5, 
        match_metric="IOS" # Intersection Over Smaller is often better for mixed-scale
    )
    
    combined_results = nms_postprocess(all_predictions)

    print(f"Detections after NMS: {len(combined_results)}")
    
    # --- STEP 5: VISUALIZE ---
    from sahi.utils.cv import visualize_object_predictions
    
    # Convert BGR to RGB for SAHI Visualization if using OpenCV image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    visualize_object_predictions(
        image_rgb,
        object_prediction_list=combined_results,
        rect_th=2,
        text_th=1,
        output_dir=".",
        file_name="sahi_sliced_result"
    )
    print("Result saved to sahi_sliced_result.png")