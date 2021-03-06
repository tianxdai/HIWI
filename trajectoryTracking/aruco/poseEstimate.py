# USAGE
# python poseEstimate.py --video videos/box.mp4

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
from cv2 import aruco
import cv2
# import imutils
import time

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
	# frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

	# blur it and convert it to the HSV
	# color space
	#blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	parameters =  aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

	# draw markers
	frame = aruco.drawDetectedMarkers(frame, corners, ids)
	
	if corners:

		rvecs,tvecs, trash = aruco.estimatePoseSingleMarkers(corners, size_of_marker , mtx, dist)


		tAverage = None
		rAverage = None


		# get reference coordinates
		if refID in ids:
			idx = np.where(ids == refID)
			rRefMtx,_jacob = cv2.Rodrigues(rvecs[idx])
			tRefVec = tvecs[idx]
			frame = aruco.drawAxis(frame, mtx, dist, rRefMtx, tRefVec, length_of_axis)
		else:
			print("ref Marker not found!!!!!!!!!!!!!!!!!!!!!!!")
			break

		
		nMarker = len(tvecs)
		trans = np.zeros((nMarker,3))
		rots = np.zeros((nMarker,3,3))
		weights = np.ones(nMarker)

		for i in range(nMarker):
			ID = ids[i,0]
			idx = np.where(ids == refID)
			rMtx,_jacob = cv2.Rodrigues(rvecs[i])
			tVec = tvecs[i] + rMtx@[0,0,-0.02]*0

			if abs(rMtx[:,-1]@[0,0,1])<1 and ID == 3:
				weights[i] = 0

			if ID == refID:
				weights[i] = 0
				continue
			elif ID == 2:
				rMtx = rMtx@cv2.Rodrigues(np.array([np.deg2rad(-90),0,0]))[0]
				rMtx = rMtx@cv2.Rodrigues(np.array([0,0,np.deg2rad(90)]))[0]
				# frame = aruco.drawAxis(frame, mtx, dist, rMtx, tVec+offset, length_of_axis)
			elif ID == 3:
				rMtx = rMtx@cv2.Rodrigues(np.array([np.deg2rad(-90),0,0]))[0]
				rMtx = rMtx@cv2.Rodrigues(np.array([0,0,np.deg2rad(-90)]))[0]
				# frame = aruco.drawAxis(frame, mtx, dist, rMtx, tVec+offset, length_of_axis)
			elif ID == 4:
				rMtx = rMtx@cv2.Rodrigues(np.array([np.deg2rad(-90),0,0]))[0]
				# frame = aruco.drawAxis(frame, mtx, dist, rMtx, tVec+offset, length_of_axis)
			elif ID == 5:
				rMtx = rMtx@cv2.Rodrigues(np.array([np.deg2rad(-90),0,0]))[0]
				rMtx = rMtx@cv2.Rodrigues(np.array([0,0,np.deg2rad(180)]))[0]
			else:
				pass 

			trans[i] = tVec
			rots[i] = rMtx
			frame = aruco.drawAxis(frame, mtx, dist, rMtx, tVec, length_of_axis)

		# frame = aruco.drawAxis(frame, mtx, dist, np.average(rots,axis=0,weights=weights), np.average(trans,axis=0,weights=weights), length_of_axis)
			


	# show the frame to our screen
	frame = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4)))
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# time.sleep(.1)

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