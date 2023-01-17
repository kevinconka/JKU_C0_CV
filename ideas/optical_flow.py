import os
import time
import numpy as np
import cv2 as cv

# get directory of this file
dir_path = os.path.dirname(os.path.realpath(__file__))
vid_path = os.path.join(dir_path, "..", "datasets", "data", "video_000.mp4")
cap = cv.VideoCapture(vid_path)
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

i = 1
fps = np.round(cap.get(cv.CAP_PROP_FPS))
print('fps: ', fps)
while(1):
    ret, frame = cap.read()
    i += 1
    if i % 2 != 0: # skip every other frame
        continue
    
    # clip frame2 to 150-255
    frame2 = np.clip(frame, 0, 255)

    if not ret:
        print('No frames grabbed!')
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    
    hsv[..., 0] = ang*180/np.pi/2
    # hsv[..., 1] = next
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr_flow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    
    # multiply next and hsv[..., 2] to get a bgr image
    bgr = mag.astype(np.float64) * next.astype(np.float64)
    bgr = cv.normalize(bgr, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(bgr.astype(np.uint8), cv.COLOR_GRAY2BGR)

    # find the max magnitude
    max_mag = np.max(mag)
    # draw circle at the max magnitude
    max_mag_loc = np.where(mag == max_mag)
    center = (max_mag_loc[1][0], max_mag_loc[0][0])
    # draw circle with no fill
    cv.circle(bgr, center, 20, (0, 0, 255), 2)

    #cv.imshow('original', frame)
    cv.imshow('bgr', bgr)
    cv.imshow('optical flow', bgr_flow)
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)

    time.sleep(0.1)
    prvs = next
    
cv.destroyAllWindows()