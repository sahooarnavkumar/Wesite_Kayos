from flask import Flask, render_template, Response
import cv2
import numpy as np
import datetime
import math
from cvzone.PoseModule import PoseDetector as pm

app = Flask(__name__)
cap = cv2.VideoCapture(0)
recording = False  # Flag to indicate if recording is in progress
output_video = None  # VideoWriter object to write the frames

detector = pm()

def generate_frames():
    x = datetime.datetime.now()
    date = '%d/%m/%Y'
    count = 0
    dir = 0
    per = 0
    bar = 0
    global output_video
    while True:
        success, img = cap.read()
        if not success:      
            print("Failed to read frame from camera")
            break
        if img.size == 0:
            print("Image has zero size")
        else:
            img = cv2.resize(img, (1280, 720))
        
        lmList, bboxInfo = detector.findPosition(img, draw=False)
        if lmList:
            p1 = lmList[11]
            p2 = lmList[13]
            p3 = lmList[15]

            x1, y1 = p1[1], p1[2]
            x2, y2 = p2[1], p2[2]
            x3, y3 = p3[1], p3[2]

            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

            color = (255, 0, 255)
            # Checking for dumble curls
            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0
            # print(count)
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                        color, 4)

            # Draw Curl Count
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                        (255, 0, 0), 25)
            actualCount = str(int(count))

            if angle < 0:
                angle += 360
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (650, 100))

        if recording and output_video is not None:
            output_video.write(img)

        ret, buffer = cv2.imencode('.jpg', img)
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
    global output_video, recording

    if output_video is not None:
        output_video.release()  # Release the video writer
        output_video = None

    recording = False

    return 'Video recording stopped.'


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
