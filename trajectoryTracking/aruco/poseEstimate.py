# USAGE
# python trajectoryTracking/aruco/poseEstimate.py --video trajectoryTracking/aruco/videos/box.mp4

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
from cv2 import aruco
import cv2
import imutils
import time

#
size_of_marker =  0.030 # side lenght of the marker in meter


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
mtx = np.loadtxt("trajectoryTracking/aruco/camCalMtx.csv")
dist = np.loadtxt("trajectoryTracking/aruco/camCalDist.csv")

# keep looping
while True:
	# time.sleep(0.1)
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
	frame = imutils.rotate(frame, -90)

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, height=600)
	#blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	parameters =  aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

	frame_markers = frame
	# draw markers
	# frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
	
	if corners:

		rvecs,tvecs, trash = aruco.estimatePoseSingleMarkers(corners, size_of_marker , mtx, dist)


		# draw axis
		length_of_axis = 0.02
		tAverage = None
		rAverage = None
		for i in range(len(tvecs)):
		# for i in range(1):
			rotMatrix,_jacob = cv2.Rodrigues(rvecs[i])
			offset = rotMatrix@[0,0,-0.02]
			tvec = tvecs[i] + offset
			markerID = ids[i,0]
			if markerID in (0,2,3):
				rotMatrix = rotMatrix@[[-1,0,0],[0,-1,0],[0,0,1]]
			elif markerID == 5:
				rotMatrix = rotMatrix@[[1,0,0],[0,0,-1],[0,1,0]]
			if markerID == 0:
				rotMatrix = rotMatrix@[[-1,0,0],[0,1,0],[0,0,-1]]
			elif markerID == 2:
				rotMatrix = rotMatrix@[[0,0,1],[0,1,0],[-1,0,0]]
			elif markerID == 3:
				rotMatrix = rotMatrix@[[0,0,-1],[0,1,0],[1,0,0]]
			rvec,_jacob = cv2.Rodrigues(rotMatrix)
			if tAverage is None:
				tAverage = tvec
				rAverage = rvec
			else:		
				tAverage += tvec
				rAverage += rvec
		tAverage/= len(tvecs)
		rAverage/= len(tvecs)
		# print(tAverage)
		frame_markers = aruco.drawAxis(frame_markers, mtx, dist, rAverage, tAverage, length_of_axis)


	# show the frame to our screen
	cv2.imshow("Frame", frame_markers)
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