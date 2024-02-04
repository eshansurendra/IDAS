import cv2
import requests
import time
import numpy as np

# Flask app URL
flask_app_url = 'http://20.2.128.189:5000/upload'

# OpenCV video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Placeholder for frame count and total response time
frame_count = 0
total_response_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame")
        break

    # Encode frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    frame_data = img_encoded.tobytes()

    # Send frame to Flask app
    start_time = time.time()
    response = requests.post(flask_app_url, data=frame_data)
    end_time = time.time()

    # Calculate response time
    response_time = end_time - start_time
    total_response_time += response_time
    frame_count += 1

    # Print frame information
    print(f"Frame {frame_count} - Response Time: {response_time:.4f} seconds")

    # Print response content
    print("Response Content:", response.text)

    # Display the frame (optional, for visualization)
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate average response time
average_response_time = total_response_time / frame_count
print(f"Average Response Time: {average_response_time:.4f} seconds")

# Release resources
cap.release()
cv2.destroyAllWindows()
