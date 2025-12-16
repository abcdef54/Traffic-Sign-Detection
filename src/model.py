import numpy as np
import supervision as sv
from ultralytics import YOLO
from typing import Dict, Tuple, Optional

class TensorRTSliceModel:
    def __init__(self, 
                 sign_model_path: str,
                 ped_model_path: Optional[str] = None, # Path for the 2nd core
                 class_names: Dict[int, str] = None,
                 conf: float = 0.25,
                 slice_inference: bool = True,
                 dual_core: bool = True,
                 imgsz: int = 960,
                 # Slicing Config
                 slice_interval: int = 5,
                 tiles: Tuple[int, int] = (2, 2),      
                 overlap_ratio: Tuple[float, float] = (0.2, 0.2)
                 ) -> None:
        
        self.sign_model_path = sign_model_path
        self.ped_model_path = ped_model_path
        self.class_names = class_names or {}
        self.conf = conf
        self.frame_count = 0
        self.slice_interval = slice_interval
        self.imgsz = imgsz
        
        self.slice_inference = slice_inference
        self.dual_core = dual_core
        
        self.sign_model = self._load_model(self.sign_model_path)
        self.ped_model = None

        if self.dual_core and not self.ped_model_path:
            print("[WARNING] Dual-Core enabled but no model path provided. Disabling Dual-Core.")
            self.dual_core = False

        if self.ped_model_path:
            self.ped_model = self._load_model(self.ped_model_path)
            print(f"[INFO] Dual-Core Loaded: {self.ped_model_path}")

        slice_wh, overlap_wh = self._calculate_slice_params(imgsz, tiles, overlap_ratio)
        print(f"[INFO] Slicing Config: {tiles} grid | Slice: {slice_wh} | Overlap: {overlap_wh}")
        
        self.slicer = sv.InferenceSlicer(
            callback=self._slice_callback,
            slice_wh=slice_wh,
            overlap_wh=overlap_wh,
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            thread_workers=4
        )

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
        """Callback for InferenceSlicer. Runs on small image chunks."""
        result = self.sign_model(image_slice, verbose=False, conf=self.conf)[0]
        return sv.Detections.from_ultralytics(result)

    def _calculate_slice_params(self, imgsz: int, tiles: Tuple[int, int], overlap_ratio: Tuple[float, float]):
        rows, cols = tiles
        overlap_h, overlap_w = overlap_ratio
        
        slice_w = int(imgsz / cols)
        slice_h = int(imgsz / rows)
        
        slice_w = int(slice_w * (1 + overlap_w))
        slice_h = int(slice_h * (1 + overlap_h))
        
        overlap_px_w = int(slice_w * overlap_w)
        overlap_px_h = int(slice_h * overlap_h)
        
        return (slice_w, slice_h), (overlap_px_w, overlap_px_h)

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        """
        Main Inference Entry Point. Returns Merged sv.Detections
        """
        self.frame_count = (self.frame_count + 1) % self.slice_interval
        
        # 1. Sign Detection Logic
        # Condition: Slicing is ON AND it is the correct "Nth" frame
        should_slice = self.slice_inference and (self.frame_count % self.slice_interval == 0)

        if should_slice:
            # Slow path (High Accuracy)
            sign_detections = self.slicer(frame)
        else:
            # Fast path (Standard YOLO)
            # We still run detection to update the tracker, but on the full frame only
            result = self.sign_model(frame, verbose=False, conf=self.conf, imgsz=self.imgsz)[0]
            sign_detections = sv.Detections.from_ultralytics(result)
        ped_detections = sv.Detections.empty()
        
        if self.dual_core and self.ped_model:
            ped_result = self.ped_model(frame, verbose=False, conf=self.conf, imgsz=self.imgsz)[0]
            ped_detections = sv.Detections.from_ultralytics(ped_result)
            
            # Filter for Humans/Cars (COCO IDs: 0=Person, 1=Bike, 2=Car, 5=Bus, 7=Truck)
            target_ids = [0, 1, 2, 5, 7]
            ped_detections = ped_detections[np.isin(ped_detections.class_id, target_ids)]
            
            # SHIFT IDs so they don't clash with signs
            ped_detections.class_id += 100

        return sv.Detections.merge([sign_detections, ped_detections])