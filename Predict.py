import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# Load Model
model=model_from_json(open("Trained Model/trainedModel.json","r").read())
# Load Weights
model.load_weights("Trained Model/trainedModel.h5")


face_haar_cascade=cv2.CascadeClassifier('Cascade Classifier/haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

while True:
	# Capture Image And Returns a Bool value
	ret,test_img=cap.read()
	if not ret:
		continue
	gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)

	faces_detected=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=5)


	for(x,y,w,h) in faces_detected:
		cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=2)
		roi_gray=gray_img[y:y+w,x:x+h] # Cropping Face From The Frame Captured
		roi_gray=cv2.resize(roi_gray,(48,48))
		img_pixels=image.img_to_array(roi_gray)
		img_pixels=np.expand_dims(img_pixels,axis=0)
		img_pixels/=255

		predictions=model.predict(img_pixels)

		# Finding Emotion Index
		max_index=np.argmax(predictions[0])

		emotions=('Angry','Disgust','Fear','Happy','Sad','Surprise','Nuetral')
		predicted_emotion=emotions[max_index]

		cv2.putText(test_img,predicted_emotion,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

	resized_img=cv2.resize(test_img,(1000,700))
	cv2.imshow('FACIAL EMOTION DETECTION',resized_img)


	# Press q to exit
	if cv2.waitKey(10)==ord('q'):
		break
cap.release()