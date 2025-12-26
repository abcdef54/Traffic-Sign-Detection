import cv2
import time
import sys
import argparse
import supervision as sv

from src import PredictionStabilizer, MultithreadVideoCapture, TensorRTSliceModel

# These are the defaults if you run without arguments
DEFAULT_SIGN_MODEL = "models/signs/best.engine"
DEFAULT_INPUT      = "outputs_vids/fix_2mins.mp4" # Or set to 0 for webcam

# Class Names Mapping
# Updated Class Names Mapping (57 Classes)
CLASS_NAMES = {
    0: "DP.135 - End all restrictions",
    1: "I.408 - Parking allowed",
    2: "I.423b - Pedestrian crossing",
    3: "P.102 - No entry",
    4: "P.103a - No cars",
    5: "P.103b - No left turn for cars",
    6: "P.103c - No right turn for cars",
    7: "P.104 - No motorcycles",
    8: "P.106a - No trucks",
    9: "P.106b - Weight limit for trucks",
    10: "P.107a - No buses",
    11: "P.112 - No pedestrians",
    12: "P.115 - Weight limit",
    13: "P.117 - Height limit",
    14: "P.123a - No left turn",
    15: "P.123b - No right turn",
    16: "P.124a - No U-turn",
    17: "P.124b - No U-turn for cars",
    18: "P.124c - No left turn or U-turn",
    19: "P.125 - No overtaking",
    20: "P.127 - Speed limit",
    21: "P.128 - No honking",
    22: "P.130 - No stopping or parking",
    23: "P.131a - No parking",
    24: "P.137 - No left or right turn",
    25: "UNUSED_P.245a",  # Removed (Noise)
    26: "R.301c - Obligatory left turn",
    27: "R.301d - Obligatory right turn",
    28: "MERGED_INTO_26", # R.301e merged into R.301c
    29: "R.302a - Right turn only",
    30: "R.302b - Left turn only",
    31: "R.303 - Roundabout",
    32: "R.407a - One way",
    33: "R.409 - U-turn allowed",
    34: "UNUSED_R.425",   # Removed (Hospital)
    35: "R.434 - Bus stop",
    36: "S.509a - Safe height info",
    37: "W.201 - Dangerous curve",         # Merged W.201a/b
    38: "MERGED_INTO_37",
    39: "W.202 - Zigzag road",             # Merged W.202a/b
    40: "MERGED_INTO_39",
    41: "W.203 - Narrow road",             # Merged W.203b/c
    42: "MERGED_INTO_41",
    43: "W.205a - 4-way intersection",
    44: "W.205b - T-intersection",         # Merged W.205b/d
    45: "MERGED_INTO_44",
    46: "W.207 - Non-priority intersection", # Merged W.207a/b/c
    47: "MERGED_INTO_46",
    48: "MERGED_INTO_46",
    49: "W.208 - Yield",
    50: "W.209 - Traffic lights ahead",
    51: "W.210 - Railway crossing",
    52: "W.224 - Pedestrian crossing ahead",
    53: "W.225 - Children crossing",
    54: "W.227 - Construction",
    55: "W.245a - Go slow",
    56: "P.124d - No right turn or U-turn" # New Synthetic Class
}

COCO_NAMES = {
    100: "Person", 101: "Bicycle", 102: "Car", 103: "Motorcycle", 
    105: "Bus", 107: "Truck"
}


def get_class_name(class_id):
    # Handle COCO Objects
    if class_id >= 100:
        return COCO_NAMES.get(class_id, f"Object-{class_id}")
    
    # Handle Merged Traffic Sign Classes (Redirect to parent)
    # This prevents "MERGED_INTO_X" from showing up on screen
    merge_map = {
        28: 26, # R.301e -> R.301c
        38: 37, # W.201b -> W.201
        40: 39, # W.202b -> W.202
        42: 41, # W.203c -> W.203
        45: 44, # W.205d -> W.205b
        47: 46, # W.207b -> W.207
        48: 46  # W.207c -> W.207
    }
    
    final_id = merge_map.get(class_id, class_id)
    return CLASS_NAMES.get(final_id, f"Sign-{final_id}")














def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Sign Detection System.")
    
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to video or '0' for webcam")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save processed video")
    parser.add_argument("--model", type=str, default=DEFAULT_SIGN_MODEL, help="Path to Sign Detection Engine")
    parser.add_argument("--ped-model", type=str, help="Path to Pedestrian Model")

    # Slicing Logic
    parser.add_argument("--no-slice", action="store_false", dest="slice", help="Disable image slicing")
    parser.set_defaults(slice=True) 
    parser.add_argument("--slice-interval", type=int, default=5, help="Slice every N frames")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio")
    
    # Thresholds
    parser.add_argument("--conf-detect", type=float, default=0.1, help="Detection Confidence")
    parser.add_argument("--conf-track", type=float, default=0.55, help="Tracking Confidence")
    
    # Display
    parser.add_argument("--verbose", default=False, action="store_true", help="Print detailed logs")
    parser.add_argument("--show", action="store_true", help="Show live window")
    parser.add_argument("--save", action="store_true", help="Save output video")

    return parser.parse_args()













def run_inference_loop(args, cap, engine, tracker, stabilizer, out):
    """
    Unified Inference Loop.
    Handles both Real-time (Webcam) and Offline (Video File) processing one frame at a time.
    """
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    print("[INFO] Starting Inference Loop...")
    
    frame_count = 0
    inference_time = 0.0
    slice_time = 0.0

    try:
        while True:
            # 1. Read Frame
            if isinstance(cap, MultithreadVideoCapture):
                frame = cap.read()
                ret = True
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                print("[INFO] End of stream.")
                break
            
            frame_count += 1
            is_slice_frame = (frame_count % args.slice_interval == 0) and args.slice
            
            start_time = time.time()
            
            detections = engine(frame)
            
            # 3. Tracking
            tracked_detections = tracker.update_with_detections(detections)

            # 4. Labeling & Stabilization
            labels = []
            if tracked_detections.tracker_id is not None:
                for (class_id, tracker_id, conf) in zip(tracked_detections.class_id,
                                                        tracked_detections.tracker_id,
                                                        tracked_detections.confidence):
                    raw_name = get_class_name(class_id)
                    final_name = stabilizer.vote(tracker_id, raw_name, conf)
                    labels.append(f"{final_name} {conf:.2f}")
            
            # 5. Annotation
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(annotated_frame, tracked_detections)
            annotated_frame = label_annotator.annotate(annotated_frame, tracked_detections, labels)

            end_time = time.time()
            dt = (end_time - start_time) * 1000
            
            if is_slice_frame: slice_time = dt
            else: inference_time = dt

            # 6. Draw HUD
            cv2.putText(annotated_frame, f"Inf: {inference_time:.1f}ms | Slice: {slice_time:.1f}ms", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            status_text = f"Slice: {'ON' if args.slice else 'OFF'} | Ped: {'ON' if args.ped_model else 'OFF'}"
            cv2.putText(annotated_frame, status_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 7. Output / Display
            if args.save and out: out.write(annotated_frame)
            if args.show:
                cv2.imshow("Traffic Sign Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            if args.verbose:
                print(f"[Frame {frame_count}] {dt:.1f}ms - Objects: {len(labels)}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        pass









def main():
    args = parse_args()

    # --- Video Source ---
    is_webcam = False
    if args.input.isdigit():
        input_source = int(args.input)
        print(f"[INFO] Opening Webcam: {input_source}")
        # Use threaded capture for webcam to reduce latency
        cap = MultithreadVideoCapture(input_source, queue_size=2)
        width, height, fps = cap.width, cap.height, cap.fps
        is_webcam = True
    else:
        if args.input == 'test':
            args.input = DEFAULT_INPUT

        print(f"[INFO] Opening Video: {args.input}")
        # Use standard capture for files
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {args.input}")
            sys.exit(1)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

    # --- Output ---
    out = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"[INFO] Saving output to: {args.output}")

    # --- Model Init ---
    print(f"[INFO] Initializing Model: {args.model}")
    overlap_ratios = (args.overlap, args.overlap)

    engine = TensorRTSliceModel(
        sign_model_path=args.model,
        ped_model_path=args.ped_model,
        class_names=CLASS_NAMES,
        conf=args.conf_detect,
        slice_inference=args.slice,
        slice_interval=args.slice_interval,
        overlap_ratio=overlap_ratios,
        dual_core=bool(args.ped_model) ,
    )

    tracker = sv.ByteTrack(track_activation_threshold=args.conf_track, lost_track_buffer=60, frame_rate=fps)
    stabilizer = PredictionStabilizer()

    # --- Single Unified Execution ---
    run_inference_loop(args, cap, engine, tracker, stabilizer, out)

    # Cleanup
    if is_webcam: cap.stop()
    else: cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    print("[INFO] Cleanup Complete.")

if __name__ == "__main__":
    # EXAMPLE RUN COMMAND
    # python main.py --input "0 for video or video path for inference" --slice-interval 5 --save --show --ped-model "path_to_pedmodel or None"
    main()