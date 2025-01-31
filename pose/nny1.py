from ultralytics import YOLO
import cv2
import numpy as np

def calculate_angle(pointA, pointB, pointC):
    pointA = np.array(pointA)
    pointB = np.array(pointB)
    pointC = np.array(pointC)
    
    AB = pointB - pointA
    BC = pointC - pointB
    
    cosine_angle = np.dot(AB, BC) / (np.linalg.norm(AB) * np.linalg.norm(BC))
    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

# Function to determine if the bicep angle is correct and provide feedback
def check_bicep_angle(angle, side):
    if 0 <= angle <= 180:
        return f'{side} Bicep Angle: {angle:.2f} (Good)', (0, 255, 0)
    else:
        return f'{side} Bicep Angle: {angle:.2f} (Adjust)', (0, 0, 255)

# Initialize the model
model = YOLO('yolov8n-pose.pt')

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

    if len(keypoints) > 0:
        # Right hand keypoints
        shoulder_r = keypoints[0][5][:2]
        elbow_r = keypoints[0][7][:2]
        wrist_r = keypoints[0][9][:2]

        # Left hand keypoints
        shoulder_l = keypoints[0][6][:2]
        elbow_l = keypoints[0][8][:2]
        wrist_l = keypoints[0][10][:2]

        # Calculate angles
        angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
        angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)

        # Get feedback for angles
        feedback_r, color_r = check_bicep_angle(angle_r, 'R')
        feedback_l, color_l = check_bicep_angle(angle_l, 'L')

        # Display angles and feedback on the frame
        cv2.putText(frame, feedback_r, (int(elbow_r[0]), int(elbow_r[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color_r, 2)
        cv2.putText(frame, feedback_l, (int(elbow_l[0]), int(elbow_l[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color_l, 2)

    # Display the frame
    cv2.imshow('YOLO Pose Estimation', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
