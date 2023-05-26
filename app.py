from flask import Flask, render_template, Response
import cv2
import numpy as np
import datetime
import math
from cvzone.PoseModule import PoseDetector as pm

app = Flask(__name__)
cap = cv2.VideoCapture(0)
detector = pm()

from flask import Flask, render_template, Response
import cv2
from cvzone.PoseModule import PoseDetector

app = Flask(__name__)
cap = cv2.VideoCapture(0)
detector = PoseDetector()

def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)  # Flip the image horizontally for mirror effect

        # Detect pose and get landmarks
        img, _, _, _ = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img)

        # Perform posture detection based on landmarks
        # You can add your custom logic here to analyze the pose and perform specific actions

        # Draw landmarks and bounding box
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_recording')
def stop_recording():
    global output_video

    if output_video is not None:
        output_video.release()  # Release the video writer
        output_video = None

    return 'Video recording stopped.'


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
