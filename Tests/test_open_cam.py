import cv2
from ultralytics import solutions, YOLO

if __name__ == '__main__':
    model_path = "models/peds/yolov8n.pt"
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Counldb't access the camera.")
        exit()
    
    print("Press q to quit")

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True, classes=[0])

        for result in results:
            annotated_frame = result.plot()

            cv2.imshow("YOLOv8n Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()