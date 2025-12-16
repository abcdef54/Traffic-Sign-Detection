import cv2
import time
import os
import sys
from collections import deque, defaultdict
from src import TensorRTSliceModel
import supervision as sv

# --- CONFIGURATION ---
# UPDATE THIS PATH to test different engines:
SIGN_MODEL = "models/signs/untouch/best_f16_static_960_batch1.engine"

PED_MODEL  = "models/peds/yolov8n.engine"
INPUT_VIDEO = "D:/Work/Code/GithubProjects/Traffic-Sign-Detection/SanityCheck/videos/Test_8mins_Part3.mp4"


# --- DETERMINE FAST PATH SIZE ---
# (Prevent static engine crashes by matching input size to engine name)
if "960" in SIGN_MODEL:
    FAST_IMGSZ = 960
elif "1280" in SIGN_MODEL:
    FAST_IMGSZ = 1280
else:
    FAST_IMGSZ = 1920 # Fallback for dynamic models

OUTPUT_VIDEO = f"Test_8mins_Part3_processed_{FAST_IMGSZ}_old.mp4"

print(f"[CONFIG] Engine: {SIGN_MODEL}")
print(f"[CONFIG] Fast Path Resolution: {FAST_IMGSZ}x{FAST_IMGSZ}")

# --- IMPROVED STABILIZER ---
class PredictionStabilizer:
    def __init__(self, history_length: int = 15, stability_threshold: int = 5) -> None:
        self.history = defaultdict(lambda: deque(maxlen=history_length))
        self.locked_label = {} 
        self.streak_counter = defaultdict(int) 
        self.stability_threshold = stability_threshold

    def vote(self, object_id, new_class_name, confident: float) -> str:
        self.history[object_id].append((new_class_name, confident))
        scores = defaultdict(float)
        for cls_name, conf in self.history[object_id]:
            scores[cls_name] += conf
        
        if not scores: return new_class_name
        challenger = max(scores, key=scores.get)

        if object_id not in self.locked_label:
            self.locked_label[object_id] = challenger
            self.streak_counter[object_id] = 0
            return challenger

        current_winner = self.locked_label[object_id]
        if challenger == current_winner:
            self.streak_counter[object_id] = 0
            return current_winner
        else:
            self.streak_counter[object_id] += 1
            if self.streak_counter[object_id] >= self.stability_threshold:
                self.locked_label[object_id] = challenger
                self.streak_counter[object_id] = 0
                return challenger
            else:
                return current_winner

# --- CLASS DEFINITIONS ---
class_names = { 
    0: "Pedestrian Crossing", 1: "Equal-level Intersection", 2: "No Entry", 
    3: "Right Turn Only", 4: "Intersection", 5: "Intersection with a non-priority road",
    6: "Danger zone on the left", 7: "No Left Turn", 8: "Bus Stop",
    9: "Roundabout", 10: "No Stopping and No Parking", 11: "U-Turn Allowed",
    12: "Lane Allocation", 13: "Slow Down", 14: "No Trucks Allowed",
    15: "Narrow Road on the Right", 16: "Height Limit", 17: "No U-Turn",
    18: "No Passenger Cars and Trucks", 19: "No U-Turn and No Right Turn",
    20: "No Cars Allowed", 21: "Narrow Road on the Left", 22: "Uneven Road",
    23: "No Two or Three-wheeled Vehicles", 24: "Customs Checkpoint",
    25: "Motorcycles Only", 26: "Obstacle on the Road", 27: "Children Present",
    28: "Trucks and Containers", 29: "No Motorcycles Allowed", 30: "Trucks Only",
    31: "Road with Surveillance Camera", 32: "No Right Turn", 33: "Double curve first to right",
    34: "No Containers Allowed", 35: "No Left or Right Turn", 36: "No Straight and Right Turn",
    37: "Intersection with T-Junction", 38: "Speed limit (50km/h)", 39: "Speed limit (60km/h)",
    40: "Speed limit (80km/h)", 41: "Speed limit (40km/h)", 42: "Left Turn",
    43: "Low Clearance", 44: "Other Danger", 45: "One-way street",
    46: "No Parking", 47: "No U-Turn for Cars", 48: "Crossing with Barriers",
    49: "No U-Turn and No Left Turn", 50: "Danger zone on the right",
    51: "Warning - Obstacle ahead - pass on the right", 52: "Stop"
}

coco_names = {
    100: "Person", 101: "Bicycle", 102: "Car", 103: "Motorcycle", 
    105: "Bus", 107: "Truck"
}

def get_class_name(class_id):
    if class_id >= 100:
        return coco_names.get(class_id, f"Object-{class_id}")
    return class_names.get(class_id, f"Sign-{class_id}")

def process():
    input_path = INPUT_VIDEO 
    if not os.path.exists(input_path):
        print(f"ERROR: Input video not found at: {input_path}")
        sys.exit(1)

    print(f"Initializing Engines... [RTX 4050]")
    
    # We initialize with imgsz=1920 for the SLICER configuration (to get 960 crops).
    # But for the fast path, we will manually override in the loop below.
    engine = TensorRTSliceModel(
        SIGN_MODEL, PED_MODEL,
        class_names=None, conf=0.1,
        slice_inference=True, dual_core=False,
        imgsz=1960, tiles=(2,2),
        slice_interval=2, 
        overlap_ratio=(0.1,0.1)
    )

    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Loaded: {width}x{height} @ {fps} FPS | {total_frames} Frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    tracker = sv.ByteTrack(frame_rate=fps, lost_track_buffer=60, track_activation_threshold=0.5)
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    label_stabilizer = PredictionStabilizer(history_length=15, stability_threshold=5)

    print("Starting Inference... (Press Ctrl+C to stop early)")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            t0 = time.time()
            
            # --- MODIFIED INFERENCE LOGIC ---
            # We override the internal engine logic slightly here to ensure safety
            engine.frame_count = frame_count
            is_sliced = (frame_count % engine.slice_interval == 0)

            if is_sliced:
                # Slicer uses the full frame and cuts it.
                # Slicer handles sizes internally, so 1920 input is fine.
                detections = engine.slicer(frame)
                mode_str = "SLICE"
            else:
                # Fast Path: MUST resize to avoid static engine crash
                # Ultralytics accepts 'imgsz' argument to handle the resize/padding
                result = engine.sign_model(frame, verbose=False, conf=engine.conf, imgsz=FAST_IMGSZ)[0]
                detections = sv.Detections.from_ultralytics(result)
                mode_str = "FAST "
            
            # (If you enable Peds later, handle them here similarly)
            
            t1 = time.time()
            inference_ms = (t1 - t0) * 1000

            # --- TRACKING ---
            tracked_detections = tracker.update_with_detections(detections)

            # --- LABELING & LOGGING ---
            labels = []
            log_entries = []
            
            if tracked_detections.tracker_id is not None:
                for class_id, tracker_id, conf in zip(tracked_detections.class_id, tracked_detections.tracker_id, tracked_detections.confidence):
                    name = get_class_name(class_id)
                    winner = label_stabilizer.vote(tracker_id, name, conf)
                    labels.append(f"#{tracker_id} {name} {conf:.2f}")

                    if conf <= 0.5:
                        status = f"[RESCUED] {conf:.2f}"
                    else:
                        status = f"(Strong)  {conf:.2f}"
                    
                    log_entries.append(f"#{tracker_id} {winner}: {status}")

            track_log = ", ".join(log_entries) if log_entries else "None"
            print(f"[Frame {frame_count}] {mode_str} ({inference_ms:.0f}ms) | Active Tracks: {track_log}")

            # --- ANNOTATION ---
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

            out.write(annotated_frame)

    except KeyboardInterrupt:
        print("\nStopping early...")
    finally:
        cap.release()
        out.release()
        print(f"\nDone! Output saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    process()