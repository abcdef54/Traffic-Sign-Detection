import cv2
import time
from src import MultithreadVideoCapture, PredictionStabilizer, TensorRTSliceModel
import supervision as sv

sign_model_path = "models/signs/besthalf.engine"
ped_model_path = "models/peds/yolov8n.engine"
output_path = "output_recording.mp4"
class_names = {
    0: "Pedestrian Crossing",
    1: "Equal-level Intersection",
    2: "No Entry",
    3: "Right Turn Only",
    4: "Intersection",
    5: "Intersection with a non-priority road",
    6: "Danger zone on the left",
    7: "No Left Turn",
    8: "Bus Stop",
    9: "Roundabout",
    10: "No Stopping and No Parking",
    11: "U-Turn Allowed",
    12: "Lane Allocation",
    13: "Slow Down",
    14: "No Trucks Allowed",
    15: "Narrow Road on the Right",
    16: "Height Limit",
    17: "No U-Turn",
    18: "No Passenger Cars and Trucks",
    19: "No U-Turn and No Right Turn",
    20: "No Cars Allowed",
    21: "Narrow Road on the Left",
    22: "Uneven Road",
    23: "No Two or Three-wheeled Vehicles",
    24: "Customs Checkpoint",
    25: "Motorcycles Only",
    26: "Obstacle on the Road",
    27: "Children Present",
    28: "Trucks and Containers",
    29: "No Motorcycles Allowed",
    30: "Trucks Only",
    31: "Road with Surveillance Camera",
    32: "No Right Turn",
    33: "Double curve first to right",
    34: "No Containers Allowed",
    35: "No Left or Right Turn",
    36: "No Straight and Right Turn",
    37: "Intersection with T-Junction",
    38: "Speed limit (50km/h)",
    39: "Speed limit (60km/h)",
    40: "Speed limit (80km/h)",
    41: "Speed limit (40km/h)",
    42: "Left Turn",
    43: "Low Clearance",
    44: "Other Danger",
    45: "One-way street",
    46: "No Parking",
    47: "No U-Turn for Cars",
    48: "Crossing with Barriers",
    49: "No U-Turn and No Left Turn",
    50: "Danger zone on the right",
    51: "Warning - Obstacle ahead - pass on the right",
    52: "Stop"
}
coco_names = {
    100: "Person", 101: "Bicycle", 102: "Car", 103: "Motorcycle", 
    105: "Bus", 107: "Truck"
}
video_path = 0

def get_class_name(class_id):
    """Helper to safely get names for both Signs and COCO objects"""
    if class_id >= 100:
        return coco_names.get(class_id, f"Object-{class_id}")
    return class_names.get(class_id, f"Sign-{class_id}")

def run():
    engine   = TensorRTSliceModel(sign_model_path,
                                ped_model_path,
                                class_names=None,
                                conf=0.2,
                                slice_inference=True,
                                dual_core=False,
                                imgsz=1920,
                                tiles=(2,2),
                                slice_interval=2,
                                overlap_ratio=(0.2,0.2))

    try:
        cap = MultithreadVideoCapture(0, 1)
    except Exception as e:
        print(f"Error accessing the camera: {e}")
        exit()

    width  = int(cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.stream.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = sv.ByteTrack(frame_rate=30)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    label_stablizer = PredictionStabilizer()

    print("\n=== CONTROLS ===")
    print(" [S] Toggle Slicing")
    print(" [D] Toggle Dual-Core")
    print(" [Q] Quit")
    print("================\n")

    fps = 30
    prev_time = 0
    while True:
        frame = cap.read()
        if frame is None:
            break
        
        detections = engine(frame)
        detections = tracker.update_with_detections(detections)

        labels = []
        if detections.tracker_id is not None:
            for class_id, tracker_id, confident in zip(detections.class_id, detections.tracker_id, detections.confidence):
                name = get_class_name(class_id)
                winner = label_stablizer.vote(tracker_id, name, confident)
                labels.append(f"{tracker_id}-{winner}-{confident:.2f}")

        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

        current_time = time.time()
        time_diff = current_time - prev_time
        prev_time = current_time

        if time_diff > 0:
            curr_fps = 1.0 / time_diff
            
            # FILTER: Ignore "instant" buffer reads that cause fake 100+ FPS spikes
            if time_diff < 0.005:
                pass 
            else:
                smoothing_factor = 0.07 
                # Correct Math:
                fps = (fps * (1 - smoothing_factor)) + (curr_fps * smoothing_factor)

        cv2.rectangle(annotated_frame, (0, 0), (960, 40), (0, 0, 0), -1)
        status_text = f"FPS: {fps:.1f} | Slice: {'ON' if engine.slice_inference else 'OFF'} | DualCore: {'ON' if engine.dual_core else 'OFF'}"
        cv2.putText(annotated_frame, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(annotated_frame)
        cv2.imshow("Night Eagle View", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            engine.toggle_slice_inference()
        elif key == ord('d'):
            engine.toggle_dual_core()

    cap.stop()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()   