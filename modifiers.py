import cv2
import numpy as np
import dlib
from mtcnn import MTCNN


ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
padding = 20

### load detector models

## FACE

# load opencv-haarcascades
face_cascade = cv2.CascadeClassifier('face_detector/opencv/haarcascade_frontalface_default.xml')

# load opencv-dnn face_detector
face_dnn_proto = "face_detector/opencv/dnn/deploy.prototxt.txt" #model structure
face_dnn_caffe = "face_detector/opencv/dnn/res10_300x300_ssd_iter_140000.caffemodel" #weights
face_dnn = cv2.dnn.readNetFromCaffe(face_dnn_proto,face_dnn_caffe)

# load mtcnn
face_mtcnn = MTCNN()

# load dlib-cnn
face_dlib_cnn = dlib.cnn_face_detection_model_v1('face_detector/dlib/cnn/mmod_human_face_detector.dat') #model weights

# load dlib-hog+svm
face_dlib_hog = dlib.get_frontal_face_detector()

## AGE

# load age detector model
age_prototxt = "age_detector/age_deploy.prototxt"
age_weights = "age_detector/age_net.caffemodel"
ageNet = cv2.dnn.readNet(age_prototxt, age_weights)

## GENDER

# load gender detector model
gender_prototxt = "gender_detector/gender_deploy.prototxt"
gender_weights = "gender_detector/gender_net.caffemodel"
genderNet = cv2.dnn.readNet(gender_prototxt, gender_weights)

def modify_frame(frame,face,age,gender):
	frame, faceBoxes = modify_frame_face(frame,face)
	frame, faceBoxes = modify_frame_age(frame,faceBoxes,age)
	frame = modify_frame_gender(frame,faceBoxes,gender)
	return modify_frame_bytes(frame)

def modify_frame_bytes(frame):
	if frame is not None:
		frame = cv2.imencode('.jpg', frame)[1].tobytes()
	return frame

def modify_frame_face(frame,face):
	faceBoxes = []
	
	# no modification
	if face == 0:
		pass

	# opencv : haarcascades
	if face == 1:
		#resize frame for better resolution
		frame = cv2.resize(frame, (600, 400))
		# convert to grayscale to avoid lighting effects
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detect faces
		faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
		# draw rectangle(s) around face(s)
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
			faceBoxes.append([x,y,x+w,y+h])

	# opencv : dnn
	if face == 2:
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300))
		face_dnn.setInput(blob)
		detections = face_dnn.forward()
		conf_threshold = 0.3
		for faceIndex in range(0, detections.shape[2]):
			confidence = detections[0, 0, faceIndex, 2]
			if confidence > conf_threshold:
				(x1, y1, x2, y2) = ( detections[0, 0, faceIndex, 3:7] * np.array([w, h, w, h]) ).astype("int")
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255,0), 2)
				faceBoxes.append([x1,y1,x2,y2])
				cv2.putText(frame, "{:.2f}%".format(confidence * 100),(x1, y1-10),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 2), 2)

	# mtcnn
	if face == 3:
		frame = cv2.resize(frame, (600, 400))
		boxes = face_mtcnn.detect_faces(frame)
		if boxes:
			box = boxes[0]['box']
			conf = boxes[0]['confidence']
			x, y, w, h = box[0], box[1], box[2], box[3]
			if conf > 0.5:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
				faceBoxes.append([x,y,x+w,y+h])

	# dlib : CNN
	if face == 4:
		# apply face detection (cnn)
		faces = face_dlib_cnn(frame, 1)
		# loop over detected faces
		for face in faces:
			x1,y1 = face.rect.left(), face.rect.top()
			x2,y2 = face.rect.right(), face.rect.bottom()
			# draw box over face
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
			faceBoxes.append([x1,y1,x2,y2])
	
	# dlib : hog
	if face == 5:
		# apply face detection (hog)
		faces = face_dlib_hog(frame, 1)
		for face in faces:
			x1, y1 = face.left(), face.top()
			x2, y2 = face.right(), face.bottom()
	    	# draw box over face
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
			faceBoxes.append([x1,y1,x2,y2])

	# return the frame and faceBoxes
	return frame, faceBoxes


# predict age of face(s) in frame
def modify_frame_age(frame,faceBoxes,age):
	if age == 0:
		return frame, faceBoxes
	if not faceBoxes:
		return frame, None

	global ageList, MODEL_MEAN_VALUES, padding
	if age == 1:
		for faceBox in faceBoxes:
			face = frame[max(0, faceBox[1]-padding) : min(faceBox[3]+padding, frame.shape[0]-1),
						max(0, faceBox[0]-padding) : min(faceBox[2]+padding, frame.shape[1]-1)]
			blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
			
			ageNet.setInput(blob)
			agePreds = ageNet.forward()
			age = ageList[agePreds[0].argmax()]
			cv2.putText(frame, f'{age}', (faceBox[2], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
	
	# return frame, faceBoxes
	return frame, faceBoxes


# predict gender of face(s) in frame
def modify_frame_gender(frame,faceBoxes,gender):
	if gender == 0:
		return frame
	if not faceBoxes:
		return frame

	global genderList, MODEL_MEAN_VALUES, padding
	if gender == 1:
		for faceBox in faceBoxes:
			face = frame[max(0, faceBox[1]-padding) : min(faceBox[3]+padding, frame.shape[0]-1),
						max(0, faceBox[0]-padding) : min(faceBox[2]+padding, frame.shape[1]-1)]
			blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
			genderNet.setInput(blob)
			genderPreds = genderNet.forward()
			gender = genderList[genderPreds[0].argmax()]
			cv2.putText(frame, f'{gender}', (faceBox[0], faceBox[2]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
	# return the frame
	return frame