import numpy as np
import cv2 as cv

vid_path = "./data_target_tracking/video_023.mp4"
cap = cv.VideoCapture(vid_path)

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    
    # convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # take fft of frame
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    # get HPF mask
    rows, cols = gray.shape[:2]
    crow, ccol = rows//2, cols//2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-15:crow+15, ccol-15:ccol+15] = 0
    # apply mask and inverse DFT
    fshift = fshift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX)

    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    cv.imshow('HPF', img_back)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
cv.destroyAllWindows()