import numpy as np
import cv2 as cv
import argparse

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
video = cv.VideoCapture("./videoT.mp4")
ret, im1 = video.read()
canalUsado = "green"

if(canalUsado=="gray"):
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    canalCorIM1 = im1Gray
else:
    rgb_im1 = cv.cvtColor(im1, cv.COLOR_BGR2RGB)
    im1Red, im1Green, im1Blue = cv.split(rgb_im1)
    if(canalUsado=="red"):
        canalCorIM1 = im1Red
    elif(canalUsado=="green"):
        canalCorIM1 = im1Green
    elif(canalUsado=="blue"):
        canalCorIM1 = im1Blue

p0 = cv.goodFeaturesToTrack(canalCorIM1, mask = None, **feature_params)

mask = np.zeros_like(im1)

while(1):
    ret, im2 = video.read()
    if not ret:
        print('Sem frames detectados.')
        break

    if(canalUsado=="gray"):
        im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
        canalCorIM2 = im2Gray
    else:
        rgb_im2 = cv.cvtColor(im2, cv.COLOR_BGR2RGB)
        im2Red, im2Green, im2Blue = cv.split(rgb_im2)
        if(canalUsado=="red"):
            canalCorIM2 = im2Red
        elif(canalUsado=="green"):
            canalCorIM2 = im2Green
        elif(canalUsado=="blue"):
            canalCorIM2 = im2Blue

    p1, st, err = cv.calcOpticalFlowPyrLK(canalCorIM1, canalCorIM2, p0, None, **lk_params)
    
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (color[i].tolist()), 2)
        im2 = cv.circle(im2, (int(a), int(b)), 5, color[i].tolist(), -1)
    
    img = cv.add(im2, mask)

    cv.imshow('frame', img)
    cv.waitKey(0)

    canalCorIM1 = canalCorIM2.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
