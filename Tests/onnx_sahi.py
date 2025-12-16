import cv2
import numpy as np
import onnxruntime as ort
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.predict import get_sliced_prediction

class OnnxYoloDetectionModel(DetectionModel):
    def load_model(self):
        # Load the ONNX model onto the GPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Hardcoded classes for GTSDB (Update this list matches your data.yaml!)
        self.class_names = [
            "speed_limit_20", "speed_limit_30", "stop", "yield", 
            "go", "turn_left", "turn_right", "etc..." 
        ]

    def perform_inference(self, image):
        # 1. Preprocess: Resize to 640x640 and Normalize
        image = cv2.resize(image, (640, 640))
        input_tensor = image.transpose(2, 0, 1) # HWC -> CHW
        input_tensor = input_tensor[np.newaxis, :, :, :].astype(np.float32)
        input_tensor /= 255.0  # 0-255 -> 0.0-1.0

        # 2. Run ONNX Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 3. Parse Output (YOLOv8 Output is [1, 84, 8400])
        # [Batch, 4 box coords + 80 classes, 8400 anchors]
        predictions = np.squeeze(outputs[0]).T # Transpose to [8400, 84]

        object_prediction_list = []
        
        # Filter logic (Simplified for ONNX raw output)
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.confidence_threshold, :]
        scores = scores[scores > self.confidence_threshold]
        
        if len(scores) == 0:
            return []

        # Convert outputs to SAHI format
        # Note: You might need Non-Maximum Suppression (NMS) here if raw ONNX output isn't pre-filtered
        # But for a quick test, we map raw boxes:
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4] 
        
        # YOLO format is usually [cx, cy, w, h], SAHI needs [x1, y1, x2, y2]
        # We need a quick converter here
        for box, score, cls_id in zip(boxes, scores, class_ids):
            cx, cy, w, h = box
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            
            # Scale back to original image size (since we resized to 640)
            # (Note: SAHI handles the slicing scaling, we just return relative to the 640 crop)
            
            prediction = ObjectPrediction(
                bbox=[x1, y1, x2, y2],
                category_id=int(cls_id),
                score=float(score),
                category_name=self.class_names[int(cls_id)] if int(cls_id) < len(self.class_names) else str(cls_id)
            )
            object_prediction_list.append(prediction)

        return object_prediction_list

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Path to your ONNX file
    MODEL_PATH = "models/signs/yolo11s.onnx"
    
    # Initialize the ONNX Model Wrapper
    detection_model = OnnxYoloDetectionModel(
        model_path=MODEL_PATH,
        confidence_threshold=0.5,
        device="cuda:0"
    )

    # Use SAHI to slice and predict
    result = get_sliced_prediction(
        "light.jpg", # Replace with your test image
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    result.export_visuals(export_dir=".", file_name="onnx_sahi_result")
    print("Inference complete. Check onnx_sahi_result.png")