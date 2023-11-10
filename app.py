import cv2
from flask import Flask, Response, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

cap = cv2.VideoCapture(0)  # Capture video from the default webcam (change the index if needed)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
        data = {
            'frame': buffer.tobytes(),
            # You can add more data here if needed
        }
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        socketio.emit('data_updated', data)  # Emit data to the web interface

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('data_updated')
def handle_data(data):
    # Process the data as needed
    print("Received data from main.py:", data)

if __name__ == '__main__':
    socketio.run(app, debug=True)
