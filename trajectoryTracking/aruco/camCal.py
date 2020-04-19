import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import imutils
#matplotlib nbagg




aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
imagesFolder = "images/"



########## create marker board for calibration ##########
# settings
board = aruco.CharucoBoard_create(4, 4, 0.04, 0.03, aruco_dict)
imboard = board.draw((4000, 4000))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
ax.axis("off")
cv2.imwrite(imagesFolder + "chessboard.tiff",imboard)
#plt.savefig(imagesFolder + "camCal.pdf")   
print("->  print it out for calibration!")
plt.grid()
plt.show()



# ######### convert video to images ##########
# # Settings
# videoFile = "videos/camCal.mp4"
# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
# board = aruco.CharucoBoard_create(7, 5, 0.020, 0.016, aruco_dict)
# # video to images
# cap = cv2.VideoCapture(videoFile)
# frameRate = cap.get(60)  #frame rate
# while(cap.isOpened()):
#     frameId = cap.get(1)  #current frame number
#     ret, frame = cap.read()
#     if (ret != True):
#         break
#     if (frameId%10==0):
#         frame = imutils.rotate(frame, -90)
#         filename = imagesFolder + "camCal/image_" +  str(int(frameId/10)) + ".jpg"
#         cv2.imwrite(filename, frame)
# cap.release()
# print ("Done!")



######### camera calibration ##########
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


images = [imagesFolder + 'camCal/' + f for f in os.listdir(imagesFolder + 'camCal/' )]
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

np.savetxt("camCalMtx.csv", mtx)
np.savetxt("camCalDist.csv", dist)




########## check calibration ##########
# setting
mtx = np.loadtxt("camCalMtx.csv")
dist = np.loadtxt("camCalDist.csv")
plt.figure()
frame = cv2.imread(imagesFolder+'pos2.jpg')
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



