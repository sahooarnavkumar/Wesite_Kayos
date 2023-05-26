from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Perform posture detection on the frame
        frame = detector.findHands(frame)
        # ...
        # Perform other operations on the frame if needed
        # ...

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def stop_recording():
    # Code to stop the recording goes here
    pass

if __name__ == '__main__':
    app.run(debug=True)
