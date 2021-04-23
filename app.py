from flask import Flask
from flask import Response
from flask import render_template
from flask import jsonify
from flask import request
from threading import Thread
from threading import Lock
from datetime import datetime
from datetime import timedelta
from time import sleep
from copy import copy
from json import dumps
from webcamvideostream import WebcamVideoStream
from modifiers import *

# shared variables
face = 0
age = 0
gender = 0

current_fps = 0.0

outputFrame = None
lock = Lock()


app = Flask(__name__)

def gen():
	global outputFrame, lock, face, age, gender, current_fps
	frame = None
	while True:
		_start = datetime.now()
		
		# copy outputFrame to local frame
		with lock:
			if outputFrame is None:
				continue
			frame = outputFrame.copy()
		
		# modify frame and return as bytes
		frame = modify_frame(frame, face, age, gender)
		if frame is None:
			continue

		# calculating current_fps
		_end = datetime.now()
		tdiff = (_end-_start).total_seconds()
		estimated_fps = 1/tdiff
		current_fps = float( copy(estimated_fps) )

		# send updated frame back
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_fps():
	global current_fps
	while True:
		json_data = dumps({
			'time': datetime.now().strftime('%H:%M:%S'),
			'value': current_fps
		})
		yield f"data:{json_data}\n\n"
		sleep(1)


# route for home page
@app.route('/')
def index():
	"""Home Page"""
	return render_template('index.html')


# route for video data stream
@app.route('/video_feed')
def video_feed():
	"""Video streaming route. Put in the src attribute of an img tag"""
	return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')	


# route for setting face/age/gender model(s)
@app.route('/config', methods=['POST'])
def config():
	"""set global variables for model configuration"""
	global face, age, gender
	if request.method == 'POST':
		with lock:
			face = int( copy(request.form.get('face_detector')) )
			age = int( copy(request.form.get('age_detector')) )
			gender = int( copy(request.form.get('gender_detector')) )
		return ('[BACKEND] : ACK : configure received, implemented!!')


# route for current_fps data stream
@app.route('/fps_feed')
def fps_feed():
	"""FPS route. use this from a chart"""
	return Response(get_fps(), mimetype='text/event-stream')



# camera thread
def run_camera():
	global outputFrame, lock
	vs = WebcamVideoStream().start()
	while True:
		frame = vs.read()
		with lock:
			outputFrame = frame.copy()
	vs.stop()

if __name__ == '__main__':
	cam_thread = Thread(target=run_camera)
	cam_thread.daemon = True
	cam_thread.start()

	app.run(threaded=True)