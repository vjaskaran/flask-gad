#Face Detector : Haar Cascades=1,DNN=2,MTCNN=3
#Age Detector : =1,=2
#Gender Detector : =1,=2

import cv2

# load opencv-haarcascades
face_cascade = cv2.CascadeClassifier('face_detector/haarcascade_frontalface_default.xml')

# load opencv-dnn
face_dnn_proto = "face_detector/dnn/deploy.prototxt.txt" #model structure
face_dnn_caffe = "face_detector/dnn/res10_300x300_ssd_iter_140000.caffemodel" #weights
face_dnn = cv2.dnn.readNetFromCaffe(face_dnn_proto,face_dnn_caffe)

def modify_frame(frame,face,age,gender):
	print('debug:modify_frame')
	frame = modify_frame_face(frame,face)
	frame = modify_frame_age(frame,age)
	frame = modify_frame_gender(frame,gender)
	return frame

def modify_frame_face(frame,face):
	# no modification
	if face == 0:
		return frame
	# opencv : haarcascades
	if face == 1:
		# convert to grayscale
		gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		# detect faces
		faces = face_cascade.detectMultiScale(gray_frame,1.1,4)
		# draw rectangle(s) around face(s)
		for(x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		return frame
	# opencv : dnn
	if face == 2:
		# get image height and width
		(h, w) = frame.shape[:2] 
		# convert image to blob
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300))
		# use detector to fetch all faces
		face_dnn.setInput(blob)
		detections = face_dnn.forward()
		# draw rectangle around each face(s) if confidence is >= 30%
		conf_threshold = 0.3
		for faceIndex in range(0, detections.shape[2]):
			confidence = detections[0, 0, faceIndex, 2]
			#filter the face confidence percentage
			if confidence > conf_threshold:
				# computer (x, y) coordinates
				(startX, startY, endX, endY) = ( detections[0, 0, faceIndex, 3:7] * np.array([w, h, w, h]) ).astype("int")
				# draw rectangle
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255,0), 2)
				# draw confidence percentage
				cv2.putText(frame, "{:.2f}%".format(confidence * 100),(startX, startY-10),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 2), 2)
	# mtcnn
	if face == 3:
		pass		

def modify_frame_age(frame,age):
	#if age == 0:
	return frame

def modify_frame_gender(frame,gender):
	#if gender == 0:
	return frame