import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
#matplotlib nbagg



##########  Create Markers  ##########
# settings
imagesFolder = "aruco_tracking/images/"
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

fig = plt.figure()
nx = 8
ny = 6
for i in range(1, nx*ny+1):
    ax = fig.add_subplot(ny,nx, i)
    img = aruco.drawMarker(aruco_dict,i-1, 700)
    plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")

plt.savefig(imagesFolder + "markers.pdf")    
plt.show()
#plt.close()
#####################################


########## create marker board for calibration ##########
# settings
imagesFolder = "aruco_tracking/images/"
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(3, 3, 1, 0.8, aruco_dict)
imboard = board.draw((4000, 4000))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
ax.axis("off")
cv2.imwrite(imagesFolder + "camera_calibration.tiff",imboard)
#plt.savefig(imagesFolder + "camera_calibration.pdf")   
print("->  print it out for calibration!")
plt.grid()
plt.show()
########################################################


######### camera calibration ##########
# Settings
imagesFolder = "aruco_tracking/images/camera_calibation/"
videoFile = "aruco_tracking/videos/camera_calibration.mp4"
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(3, 3, 1, 0.8, aruco_dict)
# video to images
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5)  #frame rate
while(cap.isOpened()):
    frameId = cap.get(1)  #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId <100):
        filename = imagesFolder + "image_" +  str(int(frameId)) + ".jpg"
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")


def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    
    imsize = None
    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.aruco.detectMarkers(gray, aruco_dict)

        imsize = gray.shape

        if len(res[0])>0:
            res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board)        
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])              

        decimator+=1   

    
    
    print("processing image finished")
    return allCorners,allIds,imsize

images = [imagesFolder + f for f in os.listdir(imagesFolder) if f.startswith("image_")]
allCorners,allIds,imsize=read_chessboards(images)


def calibrate_camera(allCorners,allIds,imsize):   
    """
    Calibrates the camera using the dected corners.
    """
    
    cameraMatrixInit = np.array([[ 2000.,    0., imsize[0]/2.],
                                 [    0., 2000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL) 
    (ret, camera_matrix, distortion_coefficients0, 
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics, 
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

print("CAMERA CALIBRATION")
ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners,allIds,imsize)
print("calibration finished")

np.savetxt("aruco_tracking/cam_calib_mtx.csv", mtx)
np.savetxt("aruco_tracking/cam_calib_dist.csv", dist)
######################################


########## check calibration ##########
# setting
imagesFolder = "aruco_tracking/images/camera_calibation/"
mtx = np.loadtxt("aruco_tracking/cam_calib_mtx.csv")
dist = np.loadtxt("aruco_tracking/cam_calib_dist.csv")
i=60 # select image id
plt.figure()
frame = cv2.imread(imagesFolder + "image_%d.jpg"%(i))
img_undist = cv2.undistort(frame,mtx,dist,None)
plt.subplot(211)
plt.imshow(frame)
plt.title("Raw image")
plt.axis("off")
plt.subplot(212)
plt.imshow(img_undist)
plt.title("Corrected image")
plt.axis("off")
plt.show()
#######################################


