from ultralytics import YOLO
import cv2

# Initialize the model
model = YOLO('yolov8m-pose.pt')

# Open the video source (webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the current frame
    results = model(frame)[0]

    # Extract keypoints
    keypoints = results.keypoints.data.numpy()  # Convert tensor to numpy array
    confidences = results.keypoints.conf.numpy()  # Convert tensor to numpy array

    # Process each keypoint
    for keypoint_idx, keypoint in enumerate(keypoints[0]):
        x, y, conf = keypoint
        if conf > 0.5:  # Use a threshold to filter out low-confidence keypoints
            cv2.putText(frame, str(keypoint_idx), (int(x), int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('YOLO Pose Estimation', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
