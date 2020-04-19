# USAGE
# python ref.py --video videos/ref.mp4

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
from cv2 import aruco
import cv2
import time
from matplotlib import pyplot as plt
#
size_of_marker =  0.03 # side lenght of the marker in meter
length_of_axis = 0.015
refID = 0
checkID = 3

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

#f_name = args.get('video').replace('.mp4','_tracking.txt')
#f_out = open(f_name, 'w')

# camera set
mtx = np.loadtxt("camCalMtx.csv")
dist = np.loadtxt("camCalDist.csv")
mtx = np.array([[ 2940.,    0., 1080.],
                [    0., 2940., 1920.],
                [    0.,    0.,    1.]])

dist = np.zeros((5,1))

# keep looping
tHis = np.zeros((int(vs.get(cv2.CAP_PROP_FRAME_COUNT)),3))
aHis = np.zeros((int(vs.get(cv2.CAP_PROP_FRAME_COUNT)),1))
timeHist = np.zeros((int(vs.get(cv2.CAP_PROP_FRAME_COUNT)),1))
count = -1
while True:
	count += 1
	current_time = vs.get(cv2.CAP_PROP_POS_MSEC)/1000
	
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# rotate the current frame
	frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
	  

	# blur it, and convert it to the HSV
	# color space
	# frame = imutils.resize(frame, height=600)
	#blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	parameters =  aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

	# draw markers
	frame = aruco.drawDetectedMarkers(frame, corners, ids)
	
	
	timeHist[count] = current_time

	if corners:

		rvecs,tVecs, trash = aruco.estimatePoseSingleMarkers(corners, size_of_marker , mtx, dist)


		# get reference coordinates
		if refID in ids:
			idx = np.where(ids == refID)
			rRefMtx,_jacob = cv2.Rodrigues(rvecs[idx])
			tRefVec = tVecs[idx]
		else:
			print("ref Marker not found!!!!!!!!!!!!!!!!!!!!!!!")
			aHis[count,:] = aHis[count-1,:]
			tHis[count,:] = tHis[count-1,:]
			# continue
			break
		# process all markers
		if checkID not in ids:
			print('checkID not found !!!!!!!!!!!!!!!!!!!!!!!!!!!')
			aHis[count,:] = aHis[count-1,:]
			tHis[count,:] = tHis[count-1,:]

		for i in range(len(tVecs)):
			markerID = ids[i,0]
			rMtx,_jacob = cv2.Rodrigues(rvecs[i])
			tVec = tVecs[i]
			frame = aruco.drawAxis(frame, mtx, dist, rMtx, tVec, length_of_axis)
			if markerID == checkID:
				transition = rRefMtx.T@(tVec-tRefVec)[0]
				# transition = tVec
				angle = np.rad2deg(np.arccos(rRefMtx.T@rMtx@[1,0,0]@[1,0,0]))
				aHis[count,:] = angle
				tHis[count,:] = transition*1000
				print('checkID = {0}'.format(markerID))
				print('time = {0}'.format(current_time))
				print('tVec = {0}'.format(transition*1000))
				print('angle = {0}'.format(angle))
				# plt.plot([count]*3, transition)
		print('================================')
		

	# show the frame to our screen
	frame = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4)))
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

#f_out.close()

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()

plt.subplot(1,2,1)
plt.title('transition of marker 3')
plt.xlabel('time (s)')
plt.ylabel('transition (mm)')
plt.plot(timeHist,tHis)
plt.legend(['x','y','z'])

plt.subplot(1,2,2)
plt.title('rotation of x axis')
plt.xlabel('time (s)')
plt.ylabel('angle (degree)')
plt.plot(timeHist,aHis)
plt.show()