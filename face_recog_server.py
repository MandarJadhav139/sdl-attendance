import os.path
import json
import numpy as np
from flask import Flask,request, Response
import uuid
import cv2
import face_recognition
def encode(test_image):
	image_of_bill=face_recognition.load_image_file('./res/train_bill.jpeg')
	bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

	image_of_steve=face_recognition.load_image_file('./res/train_steve.jpeg')
	steve_face_encoding = face_recognition.face_encodings(image_of_steve)[0]

	image_of_jeff=face_recognition.load_image_file('./res/train_jeff.jpeg')
	jeff_face_encoding = face_recognition.face_encodings(image_of_jeff)[0]

	image_of_mark=face_recognition.load_image_file('./res/train_mark.jpeg')
	mark_face_encoding = face_recognition.face_encodings(image_of_mark)[0]

	image_of_warren=face_recognition.load_image_file('./res/train_warren.jpeg')
	warren_face_encoding = face_recognition.face_encodings(image_of_warren)[0]

	image_of_sid=face_recognition.load_image_file('./res/train_sid.jpeg')
	sid_face_encoding = face_recognition.face_encodings(image_of_sid)[0]


	image_of_dennis=face_recognition.load_image_file('./res/train_dennis.jpeg')
	dennis_face_encoding = face_recognition.face_encodings(image_of_dennis)[0]

	known_face_encodings=[
	bill_face_encoding,
	steve_face_encoding,
	jeff_face_encoding,
	mark_face_encoding,
	warren_face_encoding,
	dennis_face_encoding,
	sid_face_encoding
	]

	known_face_names=[
	'Bill',
	'Steve',
	'jeff',
	'mark',	
	'warren',
	'dennis',
	'sid'
	]
	face_locations=face_recognition.face_locations(test_image)
	face_encodings=face_recognition.face_encodings(test_image,face_locations)
	flist=[]
	for (top,right,bottom,left),face_encoding in zip(face_locations,face_encodings):		
		matches=face_recognition.compare_faces(known_face_encodings,face_encoding,0.5)
		name="unknown"		
		if True in matches:
			first_match_index=matches.index(True)
			name=known_face_names[first_match_index]
		print (name)

	return face_locations

def faceDetect(img):
	face_cascade=cv2.CascadeClassifier('face_cascade.xml')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	path_file=('static/%s.jpg'%uuid.uuid4().hex)
	# cv2.imwrite(path_file,gray)	
	faces=encode(img)
	
	# for(x,y,w,h) in faces:
	# 	img=cv2.rectangle(img,(x,y),(w,h),(0,255,0))
	cv2.imwrite(path_file,img)	
	return json.dumps(path_file)
app=Flask(__name__)

@app.route('/api/upload',methods=['POST'])

def upload():	
	img = cv2.imdecode(np.fromstring(request.files['image'].read(),np.uint8),cv2.IMREAD_UNCHANGED)
	img_processed=faceDetect(img)
	print ("HELLO"+img_processed)
	return Response(response=img_processed,status=200,mimetype="application/json")
app.run(host="192.168.43.114",port=5000)
