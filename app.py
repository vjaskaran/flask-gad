from flask import Flask
from flask import Response
from flask import render_template
from flask import jsonify
from flask import request
from threading import Thread,Lock
from modifiers import *
from imutils.video import VideoStream
from copy import copy
import cv2
import imutils
import time

# shared variables
face = 0
age = 0
gender = 0
outputFrame = None
lock = Lock()

vs = VideoStream(src=0).start()
time.sleep(2.0)

def run_camera():
	global vs, outputFrame, lock
	while True:
		frame = vs.read()
		with lock:
			outputFrame = frame.copy()

app = Flask(__name__)

def gen():
	global outputFrame, lock, face, age, gender
	frame = None
	while True:
		with lock:
			if outputFrame is None:
				continue
			frame = outputFrame.copy()
			print('debug:gen face=', face, 'age=',age,' gender=', gender)
		frame = modify_frame(frame,face,age,gender)
		if frame is None:
			continue
		frame = cv2.imencode('.jpg', frame)[1].tobytes()
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
	"""Video streaming home page."""
	return render_template('index.html')

# Video streaming route. Put this in the src attribute of an img tag
@app.route('/video_feed')
def video_feed():
	"""Video streaming route. Put this in the src attribute of an img tag."""
	return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')	

@app.route('/config',methods=['POST'])
def config():
	"""set global variables of configuration"""
	global face, age, gender
	if request.method == 'POST':
		with lock:
			face = copy(request.form.get('face_detector'))
			age = copy(request.form.get('age_detector'))
			gender = copy(request.form.get('gender_detector'))
		return ('global face=' + face + ' age=' + age + ' gender=' + gender)

#@app.route('/docs/')
#def docs():
#	return "Work in Progress! To be completed by 2050. Stay Tuned!"

if __name__ == '__main__':
	cam = Thread(target=run_camera)
	cam.daemon = True
	cam.start()

	app.run(threaded=True)
	vs.stop()