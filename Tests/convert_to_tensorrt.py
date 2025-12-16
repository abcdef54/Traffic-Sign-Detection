from ultralytics import YOLO

# 1. Load the PyTorch model (it will download yolo11s.pt automatically if not found)
model = YOLO("yolo11s.pt") 

# 2. Export to TensorRT format
# This creates 'yolo11s.engine' in your current directory
# Note: This process might take a few minutes as it optimizes for your specific GPU.
model.export(format="engine", half=True)  # half=True uses FP16 for faster inference on Tensor Cores