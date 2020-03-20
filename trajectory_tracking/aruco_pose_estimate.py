# USAGE
# python3 aruco_pose_estimate.py --video ./Videos/aruco_video_single.mp4

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
from cv2 import aruco
import cv2
import imutils
import time



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize thee
# list of tracked points
yellowLower = (30/360*255, 50*2.55, 30*2.55)
yellowUpper = (45/360*255,100*2.55,100*2.55)
greenLower = (70/360*255, 50*2.55, 30*2.55)
greenUpper = (120/360*255, 100*2.55, 100*2.55)
blueLower = (230/360*255, 50*2.55, 30*2.55)
blueUpper = (250/360*255, 100*2.55, 100*2.55)
pts_g = deque(maxlen=args["buffer"])
pts_y = deque(maxlen=args["buffer"])
pts_b = deque(maxlen=args["buffer"])
n_color = 2
colorLowers = (yellowLower,greenLower,blueLower)
colorUppers = (yellowUpper,greenUpper,blueUpper)
pts_color = (pts_y,pts_g,pts_b)

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

f_name = args.get('video').replace('.mp4','_tracking.txt')
f_out = open(f_name, 'w')

# camera set
mtx = np.loadtxt("camera_cali/calib_mtx_webcam.csv")
dist = np.loadtxt("camera_cali/calib_dist_webcam.csv")

# keep looping
while True:
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
	#frame = imutils.rotate(frame, -90)

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, height=800)
	#blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	parameters =  aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
	
	if corners:

		size_of_marker =  0.0345 # side lenght of the marker in meter
		rvecs,tvecs, trash = aruco.estimatePoseSingleMarkers(corners, size_of_marker , mtx, dist)

		# draw axis
		length_of_axis = 0.01
		for i in range(len(tvecs)):
			frame_markers = aruco.drawAxis(frame_markers, mtx, dist, rvecs[i], tvecs[i], length_of_axis)


	# show the frame to our screen
	cv2.imshow("Frame", frame_markers)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

f_out.close()

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()