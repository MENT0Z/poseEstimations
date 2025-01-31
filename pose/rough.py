from ultralytics import YOLO
import cv2
import numpy as np

def calculate_angle(pointA, pointB, pointC):
    # Calculate the slopes m1 and m2
    m1 = (pointB[1] - pointA[1]) / (pointB[0] - pointA[0] + 1e-7)  # Adding small value to avoid division by zero
    m2 = (pointC[1] - pointB[1]) / (pointC[0] - pointB[0] + 1e-7)

    # Calculate the angle in radians using the arctan of the absolute value of (m1 - m2) / (1 + m1 * m2)
    angle_radians = np.arctan(abs((m1 - m2) / (1 + m1 * m2 + 1e-7)))  # Adding small value to avoid division by zero

    # Convert the angle from radians to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

# Function to determine if the bicep angle is correct and provide feedback
def check_bicep_angle(angle, side):
    if 80 <= angle <= 100:
        return f'{side} Bicep Angle: {angle:.2f} (Good)', (0, 255, 0)
    else:
        return f'{side} Bicep Angle: {angle:.2f} (Adjust)', (0, 0, 255)

# Initialize the model
model = YOLO('mruModel.pt')

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

        # Check if elbow is below shoulder before calculating angle
        if elbow_r[1] > shoulder_r[1]:
            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            # feedback_r, color_r = check_bicep_angle(angle_r, 'R')
            # cv2.putText(frame, feedback_r, (int(elbow_r[0]), int(elbow_r[1])), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, color_r, 2)
        
        if elbow_l[1] > shoulder_l[1]:
            angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
            feedback_l, color_l = check_bicep_angle(angle_l, 'L')
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
