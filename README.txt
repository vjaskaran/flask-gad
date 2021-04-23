## Development Environment : UBUNTU ##

#check if python and pip is installed and install it
sudo apt-get install python3 python3-pip

# flask : web framework for python
pip3 install flask
pip3 install numpy
pip3 install opencv-python
pip3 install dlib
pip3 install tensorflow
pip3 install mtcnn

OpenCV Haar Cascade : https://www.mlcrunch.com/face/face-haar-cascade
OpenCV DNN: https://www.mlcrunch.com/face/opencv-dnn-caffe
dlib CNN : https://towardsdatascience.com/cnn-based-face-detector-from-dlib-c3696195e01c
dlib HOG+SVM : https://towardsdatascience.com/cnn-based-face-detector-from-dlib-c3696195e01c






# Theoretical Understanding:

Gender and Age Detection System

The age system can be divided further into two sub-systems :
1. Age Detection System
2. Gender Detection System

#1. Age Detection System :

Age detection can be done in following two stages:
##1. Face detection system to get face data from image / video stream.
##2. Extracting face data from ROI (Region of Interest) and applying age detection algorithm to predict the age.

##1 : Face detection
	(a) Using Haar Cascades : Haar cascades will be very fast and capable of running in real-time on embedded devices — the problem is that they are less accurate and highly prone to false-positive detections.
	(b) Using HOG + Linear SVM models are more accurate than Haar cascades but are slower. They also aren’t as tolerant with occlusion (i.e., not all of the face visible) or viewpoint changes (i.e., different views of the face)
	(c) Deep learning-based face detectors are the most robust and will give you the best accuracy, but require even more computational resources than both Haar cascades and HOG + Linear SVMs

	Face detector will produce the bounding box coordinates of the face in the image/video stream, then we can move on to Stage ##2 — identifying the age of the person.

##2 : Age Detection
	Given the bounding box (x, y)-coordinates of the face, we will first extract the face ROI, ignoring the rest of the image/frame.
	(Doing this allows the age detector to focus solely on the person’s face and not any other irrelevant “noise” in the image.)
	The face ROI is then passed through the model, yielding the actual age prediction.
	There are a number of age detector algorithms, but the most popular and useful are deep learning-based age detectors — we’ll be using such a deep learning-based age detector

	We divide age into 8 ranges as follows:
	1. 0-2
	2. 4-6
	3. 8-12
	4. 15-20
	5. 25-32
	6. 38-43
	7. 48-53
	8. 60-100

	(Note : Non-continuous age ranges are used because it is dependent on Adience database which has such non-continuous age ranges.)


#2. Gender Detection System :

