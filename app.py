from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import dlib
from math import hypot

app = Flask(__name__)

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize global variables to store the latest frame and ratios
global_frame = None
eye_open_ratio = 0.0
mouth_open_ratio = 0.0


def detect_objects(frame):
    # Resize the image
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Perform any necessary image processing or analysis here

    return resized_frame

def eye_aspect_ratio(eye_landmark, face_roi_landmark):
    left_point = (face_roi_landmark.part(eye_landmark[0]).x, face_roi_landmark.part(eye_landmark[0]).y)
    right_point = (face_roi_landmark.part(eye_landmark[3]).x, face_roi_landmark.part(eye_landmark[3]).y)
    center_top = mid(face_roi_landmark.part(eye_landmark[1]), face_roi_landmark.part(eye_landmark[2]))
    center_bottom = mid(face_roi_landmark.part(eye_landmark[5]), face_roi_landmark.part(eye_landmark[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio

def mouth_aspect_ratio(lips_landmark, face_roi_landmark):
    left_point = (face_roi_landmark.part(lips_landmark[0]).x, face_roi_landmark.part(lips_landmark[0]).y)
    right_point = (face_roi_landmark.part(lips_landmark[2]).x, face_roi_landmark.part(lips_landmark[2]).y)
    center_top = (face_roi_landmark.part(lips_landmark[1]).x, face_roi_landmark.part(lips_landmark[1]).y)
    center_bottom = (face_roi_landmark.part(lips_landmark[3]).x, face_roi_landmark.part(lips_landmark[3]).y)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    if hor_line_length == 0:
        return ver_line_length
    ratio = ver_line_length / hor_line_length
    return ratio

def calculate_ratios(face_roi_landmark):
    left_eye_ratio = eye_aspect_ratio([36, 37, 38, 39, 40, 41], face_roi_landmark)
    right_eye_ratio = eye_aspect_ratio([42, 43, 44, 45, 46, 47], face_roi_landmark)
    eye_open_ratio = (left_eye_ratio + right_eye_ratio) / 2

    inner_lip_ratio = mouth_aspect_ratio([60, 62, 64, 66], face_roi_landmark)
    outer_lip_ratio = mouth_aspect_ratio([48, 51, 54, 57], face_roi_landmark)
    mouth_open_ratio = (inner_lip_ratio + outer_lip_ratio) / 2

    return eye_open_ratio, mouth_open_ratio

def gen():
    global global_frame, eye_open_ratio, mouth_open_ratio
    while True:
        if global_frame is not None:
            # Perform object detection on the frame
            detected_frame = detect_objects(global_frame)

            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', detected_frame)
            frame = buffer.tobytes()

            # Yield eye_open_ratio and mouth_open_ratio as part of the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   b'eye_open_ratio: {:.4f}, mouth_open_ratio: {:.4f}\r\n\r\n'.format(eye_open_ratio, mouth_open_ratio) +
                   frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    global global_frame, eye_open_ratio, mouth_open_ratio
    frame_data = request.data
    frame_np = np.frombuffer(frame_data, dtype=np.uint8)
    global_frame = cv2.imdecode(frame_np, flags=1)  # 1 means load color image

    # Detect face and calculate ratios
    faces = detector(cv2.cvtColor(global_frame, cv2.COLOR_BGR2GRAY))
    for face_roi in faces:
        landmark_list = predictor(cv2.cvtColor(global_frame, cv2.COLOR_BGR2GRAY), face_roi)
        eye_open_ratio, mouth_open_ratio = calculate_ratios(landmark_list)

    response_data = {
        'status': 'OK',
        'eye_open_ratio': eye_open_ratio,
        'mouth_open_ratio': mouth_open_ratio,
    }

    return jsonify(response_data)

@app.route('/preview')
def preview():
    return render_template('preview.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
