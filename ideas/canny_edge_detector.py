import cv2 as cv

vid_path = "./data_target_tracking/video_023.mp4"
cap = cv.VideoCapture(vid_path)

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    
    # canny edge detection
    edges = cv.Canny(frame, 100, 200)
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', edges)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
cv.destroyAllWindows()