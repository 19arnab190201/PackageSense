import cv2
import numpy as np
from utils import send_coordinates_to_serial
from object_detection import detect_and_crop_package
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import json
import base64

app = Flask(__name__)
socketio = SocketIO(app)

# Create a video capture object to capture video from your camera (0 indicates the default camera)
cap = cv2.VideoCapture(1)
label_camera = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Define the cropping dimensions (left, top, right, bottom)
            left = 50  # Number of pixels to crop from the left side
            top = 50   # Number of pixels to crop from the top
            right = 50 # Number of pixels to crop from the right side
            bottom = 50  # Number of pixels to crop from the bottom

            # Crop the frame
            frame = frame[top:-bottom, left:-right]

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')



# Add a new route for the label camera feed
@app.route('/label_camera_feed')
def label_camera_feed():
    def generate_label_camera():
        while True:
            ret, frame = label_camera.read()
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_label_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect_package')
def detect_package():
    while True:  # Loop to repeat the detection process
        print("STARTED")
        ret, frame = cap.read()

        # Define the cropping dimensions (left, top, right, bottom)
        left = 50  # Number of pixels to crop from the left side
        top = 50  # Number of pixels to crop from the top
        right = 50  # Number of pixels to crop from the right side
        bottom = 50  # Number of pixels to crop from the bottom

        # Crop the frame
        frame = frame[top:-bottom, left:-right]

        if ret:
            depth_map, detected_object, cropped_package, package_info = detect_and_crop_package(frame)

            if cropped_package is not None:
                # Convert the depthData NumPy array to an integer
                package_info['depthData'] = int(round(package_info['depthData']))
                print(package_info)
                # Create the display_data dictionary
                display_data = {
                    'depthData': str(package_info['depthData']),
                    'center': f'{package_info["center"][0]}, {package_info["center"][1]}',
                    'width': str(package_info['width']),
                    'height': str(package_info['height']),
                    'coordinates': f'{package_info["coordinates"][0]}, {package_info["coordinates"][1]}',
                    'has_label': str(package_info['has_label']),
                }

                # Convert the depth map and detected object images to base64-encoded strings
                _, buffer = cv2.imencode('.jpg', depth_map)  # Replace 'depth_map' with your depth map image
                display_data['depth_map'] = base64.b64encode(buffer).decode('utf-8')

                _, buffer = cv2.imencode('.jpg', detected_object)  # Replace 'detected_object' with your detected object image
                display_data['detected_object'] = base64.b64encode(buffer).decode('utf-8')

                # Send detection data to the client
                socketio.emit('detection_data', json.dumps(display_data))  # Send display_data as JSON

                send_coordinates_to_serial(package_info["coordinates"][0],
                                           package_info["coordinates"][1],
                                           package_info['depthData'],
                                           package_info['has_label'])

                _, buffer = cv2.imencode('.jpg', cropped_package)
                cropped_bytes = buffer.tobytes()
                return Response(cropped_bytes, content_type='image/jpeg')
            return ''


if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
