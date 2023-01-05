import numpy as np
import cv2 as cv

vid_path = "./data_target_tracking/video_023.mp4"
cap = cv.VideoCapture(vid_path)
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while(1):
    ret, frame = cap.read()
    
    # clip frame2 to 150-255
    frame2 = np.clip(frame, 150, 255)

    if not ret:
        print('No frames grabbed!')
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # find the max magnitude
    max_mag = np.max(mag)
    # draw circle at the max magnitude
    max_mag_loc = np.where(mag == max_mag)
    center = (max_mag_loc[1][0], max_mag_loc[0][0])
    # draw circle with no fill
    cv.circle(frame, center, 20, (0, 0, 255), 2)
    

    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('original', frame)
    cv.imshow('clipped', cv.normalize(frame2, None, 0, 255, cv.NORM_MINMAX))
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next
    
cv.destroyAllWindows()