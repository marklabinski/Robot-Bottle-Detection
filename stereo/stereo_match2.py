#!/usr/bin/env python



# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


if __name__ == '__main__':
    for i in range(68,210,2):
        print('loading images...')
        #imgL = cv.pyrDown( cv.imread('C:/Users/markl/startup/raw-images/RobotImages8-29/L_Image198.jpg') )  # downscale images for faster processing
        #imgR = cv.pyrDown( cv.imread('C:/Users/markl/startup/raw-images/RobotImages8-29/R_Image28.jpg') )
        imgL = cv.imread('C:/Users/markl/startup/raw-images/Baseline/L_Image{}.jpg'.format(i),1)
        imgR = cv.imread('C:/Users/markl/startup/raw-images/Baseline/R_Image{}.jpg'.format(i),1)

        # disparity range is tuned for 'aloe' image pair
        window_size = 5
        min_disp = 32
        num_disp = 112-min_disp
        stereo = cv.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = 8,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32
            #fullDP = False,
            #SADWindowSize = window_size,
        )

        # morphology settings
        kernel = np.ones((12,12),np.uint8)

        print('computing disparity...')
        disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        disparity = (disparity-min_disp)/num_disp

        # apply threshold
        threshold = cv.threshold(disparity, 0.2, 1.0, cv.THRESH_BINARY)[1]

        # apply morphological transformation
        morphology = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel)

        #print('generating 3d point cloud...',)
        #h, w = imgL.shape[:2]
        #f = 0.8*w                          # guess for focal length
        #Q = np.float32([[1, 0, 0, -0.5*w],
        #                [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
        #                [0, 0, 0,     -f], # so that y-axis looks up
        #                [0, 0, 1,      0]])
        #points = cv.reprojectImageTo3D(disp, Q)
        #colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
        #mask = disp > disp.min()
        #out_points = points[mask]
        #out_colors = colors[mask]
        #out_fn = 'out.ply'
        #write_ply('out.ply', out_points, out_colors)
        #print('%s saved' % 'out.ply')

        cv.imshow('left', imgL)
        cv.imshow('right',imgR)
        #cv.imshow('disparity', (disp-min_disp)/num_disp)
        cv.imshow('disparity',disparity)
        #cv.imshow('threshold', threshold)
        #cv.imshow('morphology', morphology)
        cv.waitKey()
        cv.destroyAllWindows()
