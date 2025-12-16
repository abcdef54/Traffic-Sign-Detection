import tensorrt as trt
import os

def build_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # 1. Create Network
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # 2. Config: Set Memory & FP16 (Optional but recommended)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2GB Workspace
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Enabled FP16 precision.")

    # 3. Parse ONNX
    print(f"Parsing {onnx_path}...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # 4. Build Engine
    print("Building engine... (This takes time)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine:
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        print(f"SUCCESS: Saved to {engine_path}")
    else:
        print("Build failed.")

if __name__ == "__main__":
    build_engine("yolo11s.onnx", "yolo11s.engine")