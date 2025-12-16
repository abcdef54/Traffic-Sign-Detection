import torch
import onnxruntime as ort
import numpy as np
import cv2

# PATH to your newly exported model
MODEL_PATH = "models/signs/yolo11s.onnx"

def test_onnx_model():
    print(f"Testing model: {MODEL_PATH}")

    # 1. Check for GPU Availability
    # This list tells us what hardware ONNX Runtime can see
    available_providers = ort.get_available_providers()
    print(f"Available Providers: {available_providers}")

    if 'CUDAExecutionProvider' not in available_providers:
        print("WARNING: CUDA (GPU) is NOT available. Using CPU instead.")
    else:
        print("SUCCESS: CUDA (GPU) is available! 'Speed Demon' mode ready.")

    # 2. Load the Model
    try:
        # We explicitly ask for GPU first, then CPU as backup
        session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model. {e}")
        return

    # 3. Create a Dummy Input (Fake Image)
    # YOLO expects: [Batch_Size, Channels, Height, Width]
    # We use a random noise image just to check if the math runs
    dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

    # 4. Run Inference
    try:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print("Running inference on dummy input...")
        outputs = session.run([output_name], {input_name: dummy_input})
        
        # Output shape should be [1, 84, 8400] for YOLOv8/v11
        print(f"Inference Successful!")
        print(f"Output Shape: {outputs[0].shape}")
        
    except Exception as e:
        print(f"Inference Failed: {e}")

if __name__ == "__main__":
    test_onnx_model()