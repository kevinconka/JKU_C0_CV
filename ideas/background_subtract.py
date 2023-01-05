import cv2 as cv

vid_path = "./data_target_tracking/video_023.mp4"
cap = cv.VideoCapture(vid_path)

backSub = cv.createBackgroundSubtractorMOG2()
#backSub = cv.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    
    #fgMask = backSub.apply(frame)
    
    # use fastNlMeansDenoisingMulti to denoise the frame
    h = 4
    templateWindowSize = 7
    searchWindowSize = 21
    denoised = cv.fastNlMeansDenoisingMulti(frame, 2, 5, None, h, templateWindowSize, searchWindowSize)
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    # cv.imshow('FG Mask', fgMask)
    cv.imshow('Denoised', denoised)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
cv.destroyAllWindows()