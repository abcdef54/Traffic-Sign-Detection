import cv2
import time
import sys
import argparse
import supervision as sv

from src import TensorRTSliceModel, PredictionStabilizer, MultithreadVideoCapture

# These are the defaults if you run without arguments
DEFAULT_SIGN_MODEL = "models/signs/best.engine"
DEFAULT_PED_MODEL  = None # "models/peds/yolov8n.engine"
DEFAULT_INPUT      = "videos/test_clip.mp4" # Or set to 0 for webcam
TRAIN_IMGSZ        = 960  # The resolution you trained on

# Class Names Mapping
CLASS_NAMES = { 
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

COCO_NAMES = {
    100: "Person", 101: "Bicycle", 102: "Car", 103: "Motorcycle", 
    105: "Bus", 107: "Truck"
}
















def get_class_name(class_id):
    if class_id >= 100:
        return COCO_NAMES.get(class_id, f"Object-{class_id}")
    return CLASS_NAMES.get(class_id, f"Sign-{class_id}")



def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Sign Detection System.")
    parser.add_argument("--input", type=str, default="0", help="Path to video or '0' for webcam")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save processed video")
    parser.add_argument("--model", type=str, help="Path to Sign Detection Engine")
    parser.add_argument("--ped-model", type=str, default="", help="Path to Pedestrian Model (Optional)")
    parser.add_argument("--base-imgsz", type=int, default=960, help="The training resolution (e.g. 960)")
    parser.add_argument("--no-slice", action="store_false", dest="slice", help="Disable image slicing")
    parser.set_defaults(slice=True) 
    parser.add_argument("--slice-interval", type=int, default=5, help="Slice every N frames")
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap ratio (e.g. 0.2 for 20%)")
    parser.add_argument("--conf-detect", type=float, default=0.2, help="Detection Confidence Threshold")
    parser.add_argument("--conf-track", type=float, default=0.55, help="Tracking Confidence Threshold")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")
    parser.add_argument("--show", action="store_true", help="Show live window")
    parser.add_argument("--save", action="store_true", help="Save output video")

    return parser.parse_args()










def main():
    args = parse_args()

    if args.input.isdigit():
        input_source = int(args.input)
        print(f"[INFO] Opening Webcam: {input_source}")
        cap = MultithreadVideoCapture(input_source, queue_size=2)
        width, height, fps = cap.width, cap.height, cap.fps
        is_webcam = True
    
    else:
        print(f"[INFO] Opening Video: {args.input}")
        cap = cv2.VideoCapture(args.input)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        is_webcam = False
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {args.input}")
            sys.exit(1)

    out = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"[INFO] Saving output to: {args.output}")
    

    overlap_ratios = (args.overlap, args.overlap)
    print(f"[INFO] Initializing Model: {args.model}")
    engine = TensorRTSliceModel(
        sign_model_path=args.model,
        dual_core=args.ped_model,
        ped_model_path=args.ped_model,
        train_imgsz=args.base_imgsz,
        input_imgsz=(height, width),
        conf=args.conf_detect,
        slice_inference=args.slice,
        slice_interval=args.slice_interval,
        overlap_ratio=overlap_ratios
    )

    tracker = sv.ByteTrack(track_activation_threshold=args.conf_track,
                           lost_track_buffer=60,
                           frame_rate=fps)
    
    stabilizer = PredictionStabilizer()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    print("[INFO] Starting Loop. Press 'Q' or 'Ctrl+C' to stop.")
    frame_count = 0
    is_slice = True
    infernce_time = 0.0
    slice_time = 0.0
    try:
        while True:
            if is_webcam:
                frame = cap.read()
                ret = True
            
            else:
                ret, frame = cap.read()

            if not ret: break
            frame_count += 1
            is_slice = frame_count % args.slice_interval == 0
            start_time = time.time()

            detections = engine(frame)
            tracked_detections = tracker.update_with_detections(detections)

            labels = []
            if tracked_detections.tracker_id is not None:
                for (class_id, tracker_id, conf) in zip(tracked_detections.class_id,
                                                        tracked_detections.tracker_id,
                                                        tracked_detections.confidence):
                    raw_name =  get_class_name(class_id)
                    final_name = stabilizer.vote(tracker_id, raw_name, conf)
                    labels.append(f"#{tracker_id} {final_name} {conf:.2f}")
            
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(annotated_frame, tracked_detections)
            annotated_frame = label_annotator.annotate(annotated_frame, tracked_detections, labels)

            end_time = time.time()
            dt = (end_time - start_time) * 1000
            if is_slice: slice_time = dt
            else: infernce_time = dt

            cv2.putText(annotated_frame, f"Inference: {infernce_time:.1f}ms Inference (SLICING): {slice_time:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw Config Status
            status_text = f"Slice: {'ON' if args.slice else 'OFF'} | Base: {args.base_imgsz} Ped-Tracking: {'ON' if args.ped_model else 'OFF'}"
            cv2.putText(annotated_frame, status_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if args.save and out:
                out.write(annotated_frame)

            if args.show:
                cv2.imshow("Traffic Sign Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if args.verbose:
                print(f"[Frame {frame_count}] {dt:.1f}ms - Objects: {len(labels)}")
            
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        if is_webcam: cap.stop()
        else: cap.release()
        
        if out: out.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup Complete.")


if __name__ == "__main__":
    main()