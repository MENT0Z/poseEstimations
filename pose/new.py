from ultralytics import YOLO
import cv2
import numpy as np

# Initialize the model
model = YOLO('yolov8m-pose.pt')

# Define a function to calculate the angle between two vectors
def calculate_angle(pointA, pointB, pointC):
    ba = pointA - pointB
    bc = pointC - pointB
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Function to process results and draw angles on the frame
def process_and_draw_results(frame, results):
    for result in results:
        keypoints = result.keypoints
        if keypoints is not None:
            # Extract keypoints for left shoulder, left elbow, and left wrist
            left_shoulder = np.array([keypoints[5]['x'], keypoints[5]['y']])
            left_elbow = np.array([keypoints[7]['x'], keypoints[7]['y']])
            left_wrist = np.array([keypoints[9]['x'], keypoints[9]['y']])

            # Calculate the angle at the left elbow (biceps angle)
            left_biceps_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            cv2.putText(frame, f"Left biceps angle: {left_biceps_angle:.2f} degrees", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Extract keypoints for right shoulder, right elbow, and right wrist
            right_shoulder = np.array([keypoints[6]['x'], keypoints[6]['y']])
            right_elbow = np.array([keypoints[8]['x'], keypoints[8]['y']])
            right_wrist = np.array([keypoints[10]['x'], keypoints[10]['y']])

            # Calculate the angle at the right elbow (biceps angle)
            right_biceps_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            cv2.putText(frame, f"Right biceps angle: {right_biceps_angle:.2f} degrees", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

# Open the video source (webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the current frame
    results = model(frame)

    # Process and draw results on the frame
    process_and_draw_results(frame, results)

    # Display the frame
    cv2.imshow('YOLO Pose Estimation', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
