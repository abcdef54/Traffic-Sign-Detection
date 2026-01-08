import numpy as np
import supervision as sv
from ultralytics import YOLO
from typing import Dict, Tuple, Optional
import os

class TensorRTSliceModel:
    def __init__(self, 
                 sign_model_path: str,
                 ped_model_path: Optional[str] = None,
                 class_names: Dict[int, str] = None,
                 conf: float = 0.1,
                 slice_inference: bool = True,
                 dual_core: bool = True,
                 slice_interval: int = 5,
                 overlap_ratio: Tuple[float, float] = (0.0, 0.0)
                 ) -> None:
        
        print("-------------Initializing TensorRTSliceModel-------------")
        self.sign_model_path = sign_model_path
        self.ped_model_path = ped_model_path
        self.class_names = class_names or {}
        self.conf = conf
        self.frame_count = 0
        
        self.imgsz = 1280 
        
        self.slice_inference = slice_inference
        self.slice_interval = slice_interval
        
        if self.slice_inference:
            print("[INFO] Slice Inference: ON")
            print(f"[INFO] Slice interval: {self.slice_interval}")
        else:
            print("[INFO] Slice Inference: OFF")
            
        self.dual_core = dual_core
        
        self.sign_model = self._load_model(self.sign_model_path)
        print("[INFO] Sign Detection Model Loaded")
        
        self.ped_model = None
        if self.dual_core and not self.ped_model_path:
            print("[WARNING] Dual-Core enabled but no model path provided. Disabling Dual-Core.")
            self.dual_core = False
        
        if self.ped_model_path:
            self.ped_model = self._load_model(self.ped_model_path)
            print(f"[INFO] Dual-Core Loaded: {self.ped_model_path}")

        slice_wh = (self.imgsz, self.imgsz)
        overlap_wh = (
            int(slice_wh[0] * overlap_ratio[0]), 
            int(slice_wh[1] * overlap_ratio[1])
        )
        
        stride_w = slice_wh[0] - overlap_wh[0]
        stride_h = slice_wh[1] - overlap_wh[1]
        
        if stride_w <= 0 or stride_h <= 0:
            raise ValueError("Overlap is too large! Stride must be positive.")

        print(f"[INFO] Model Resolution: {self.imgsz}x{self.imgsz}")
        print(f"[INFO] Slicing Config: Fixed Slice: {slice_wh} | Overlap: {overlap_wh}")
        
        workers = os.cpu_count() or 1
        workers = min(workers, 8) 
        print(f"[INFO] Thread Workers: {workers}")

        self.slicer = sv.InferenceSlicer(
            callback=self._slice_callback,
            slice_wh=slice_wh,
            overlap_wh=overlap_wh,
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            thread_workers=workers
        )

        print("-------------Finished-------------")


    def _load_model(self, path: str) -> YOLO:
        try:
            return YOLO(path, task='detect', verbose=False)
        except Exception as e:
            raise Exception(f"CRITICAL: Could not load model at {path}. Error: {e}")
        
    def toggle_dual_core(self):
        if not self.ped_model:
            print("[ERROR] Cannot enable Dual-Core: No model loaded.")
            return
        self.dual_core = not self.dual_core
        status = "ON" if self.dual_core else "OFF"
        print(f"[CONTROL] Dual-Core Mode: {status}")
    
    def toggle_slice_inference(self):
        self.slice_inference = not self.slice_inference
        status = "ON" if self.slice_inference else "OFF"
        print(f"[CONTROL] Slice Inference: {status}")

    def _slice_callback(self, image_slice: np.ndarray) -> sv.Detections:
        result = self.sign_model(image_slice, verbose=False, conf=self.conf, imgsz=self.imgsz)[0]
        return sv.Detections.from_ultralytics(result)

    def __call__(self, frame: np.ndarray | list[np.ndarray]) -> sv.Detections:
        self.frame_count = (self.frame_count + 1) % self.slice_interval
        should_slice = self.slice_inference and (self.frame_count % self.slice_interval == 0)

        if should_slice:
            sign_detections = self.slicer(frame)
        else:
            result = self.sign_model(frame, verbose=False, conf=self.conf, imgsz=self.imgsz)[0]
            sign_detections = sv.Detections.from_ultralytics(result)
        
        ped_detections = None
        if self.dual_core and self.ped_model:
            ped_result = self.ped_model(frame, verbose=False, conf=self.conf, imgsz=self.imgsz // 2)[0]
            ped_det = sv.Detections.from_ultralytics(ped_result)
            
            target_ids = [0, 1, 2, 5, 7]
            ped_det = ped_det[np.isin(ped_det.class_id, target_ids)]
            
            ped_det.class_id += 100
            ped_detections = ped_det

        if ped_detections:
            return sv.Detections.merge([sign_detections, ped_detections])
        
        return sign_detections