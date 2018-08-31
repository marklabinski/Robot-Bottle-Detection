'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html
'''

import numpy as np
import cv2

def disparity(self):
        matcher = cv2.StereoBM_create(1024, 7)
        disparity = matcher.compute(cv2.cvtColor(self.images[0], cv2.COLOR_BGR2GRAY),
                                    cv2.cvtColor(self.images[1], cv2.COLOR_BGR2GRAY))
        self.process_output(disparity)

def readyStereoBM(roi1, roi2):
    stereobm = cv2.StereoBM_create(numDisparities=112, blockSize=31)
    stereobm.setPreFilterSize(31)#41
    stereobm.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
    stereobm.setPreFilterCap(31)
    stereobm.setTextureThreshold(10)
    stereobm.setMinDisparity(0)
    stereobm.setSpeckleWindowSize(100)
    stereobm.setSpeckleRange(64)
    stereobm.setUniquenessRatio(0)
    stereobm.setROI1(roi1)
    stereobm.setROI1(roi2)
    return stereobm 

# Load img directory 2
img_dir2 = 'C:/Users/markl/startup/raw-images/Disparity/'

# Load the left and right images in gray scale
imgLeft = cv2.imread(img_dir2+'L_Image18.jpg',0)
imgRight = cv2.imread(img_dir2+'R_Image18.jpg',0)

# Initialize the stereo block matching object
stereo = cv2.StereoBM_create(numDisparities=16*8, blockSize=5)

# Compute the disparity image
disparity = stereo.compute(imgLeft, imgRight)

# Normalize the image for representation
min = disparity.min()
max = disparity.max()
disparity = np.uint8(255 * (disparity - min) / (max - min))

# Display the result
cv2.imshow('disparity', np.hstack((imgLeft, imgRight, disparity)))
cv2.waitKey(0)
cv2.destroyAllWindows()
