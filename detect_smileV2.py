from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os


os.chdir('/home/pol/faceD')

def angle(point1,point2):
	x1 = point1[0]
	y1 = point1[1]
	x2 = point2[0]
	y2 = point2[1]
	ang = int(np.arctan((y2-y1)/(x2-x1))*180/np.pi)
	return(ang)

def smile(mouth):
	A = dist.euclidean(mouth[3], mouth[9])
	B = dist.euclidean(mouth[2], mouth[10])
	C = dist.euclidean(mouth[4], mouth[8])
	avg = (A+B+C)/3
	D = dist.euclidean(mouth[0], mouth[6])
	mar=avg/D
	return mar

def showImg(img):
	cv2.imshow('temp',img)
	cv2.waitKey(0)


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
# (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

vs = cv2.VideoCapture(0)
while True:

	# fileName = './dataset/'+imgFiles[0]
	# frame = cv2.imread(fileName,3)
	ret, frame = vs.read()
	# frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	if len(rects) ==0:
		print('no mouse')

	frame2 = gray.copy()
	frame2 *=0 #create empty matrix
	for rect in rects:#for each detected face
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		mouth = shape[mStart : mEnd] #get the mouse
		mouthHull = cv2.convexHull(mouth)  #get convexhull of the mouse

		mouseX = [mouthHull[i][0][0] for i in range(mouthHull.shape[0])]
		mouseY = [mouthHull[i][0][1] for i in range(mouthHull.shape[0])]
		x = np.min(mouseX)
		y = np.min(mouseY)
		w = np.max(mouseX) - x
		h = np.max(mouseY) - y

		cv2.fillPoly(frame2, [mouthHull], color=(255,255,255))#fill the convexhull area
		mouseC = frame2[y:y+h,x:x+w] #crop

		cv2.drawContours(frame, [mouthHull], -1, (255, 255, 0), 1) #draw around the mouth


		ang = angle(mouth[0], mouth[6])#determine angle between the mouse's corne
		mouseCR = imutils.rotate_bound(mouseC, -ang)#rotate the mouse(Adrain's method)
		s = np.array( [sum(mouseCR[i,:]) for i in range(mouseCR.shape[0])])#I like np array

		maxIndex = np.argmax(s)
		maxIndexP = np.round(maxIndex/mouseCR.shape[0]*100) #determine the mouse's corner
		# cv2.line(mouseCR,(0,maxIndex),(5000,maxIndex),(0,0,0),5)#just debuging

		mouseCR = mouseCR[s!=0,:] #crop top/bottom
		cv2.imshow('mouse',mouseCR)
		mar = smile(mouth) #thanks for the Idea of https://www.freecodecamp.org/news/smilfie-auto-capture-selfies-by-detecting-a-smile-using-opencv-and-python-8c5cfb6ec197/

		ans ='natural'
		if mar>0.8:
			ans = 'oh'
		elif mar<0.3:#assume that the mouse horizontal expanding
			if maxIndexP> 65:
				ans = 'angry'
			elif maxIndexP< 45:
				ans = 'smile'
		else:
			if maxIndexP< 30:
				ans = 'big Smile'

		# for point in mouth:
		# 	cv2.rectangle(frame,(point[0],point[1]),(point[0]+5,point[1]+5),(0,255,0),3)
		# 	# cv2.rectangle(frame,(point[0][0],point[0][1]),(point[0][0]+5,point[0][1]+5),(0,255,0),3)

		#clean version
		cv2.putText(frame,ans,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),5)
		#debugging version
		# cv2.putText(frame,ans+' '+str(maxIndexP)+' '+str(mar),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),5)

	# show the frame
	cv2.imshow("Frame", frame)
	cv2.imshow("Frame2", frame2)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

vs.release()
cv2.destroyAllWindows()
