import cv2
from ultralytics import YOLO

def main():
    print("=== Virtual Technologies Internship: AI Task 2 - Real-Time Object Detection ===")
    
    # Load YOLOv8 pre-trained model
    # It will automatically download yolov8n.pt the first time
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    
    print("Initializing webcam (Press 'q' to quit)...")
    
    # Open the default webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Ensure your webcam is connected or provide a video file path instead.")
        return
        
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Perform object detection
        results = model(frame, stream=True)
        
        # Draw bounding boxes and labels on the frame
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Confidence and class
                conf = round(box.conf[0].item(), 2)
                cls = int(box.cls[0].item())
                class_name = model.names[cls]
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name} {conf}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        # Display the resulting frame
        cv2.imshow('YOLOv8 Real-Time Object Detection', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("=== AI Task 2 Exited Successfully ===")

if __name__ == "__main__":
    main()
