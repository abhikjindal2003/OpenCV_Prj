import cv2, time

camr = cv2.VideoCapture(0)
static_back = None

while True:
    check,frame = camr.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray = cv2.GaussianBlur(gray, (21, 21), 0) 

    if static_back is None:
        static_back = gray
        continue
    diff_frame = cv2.absdiff(static_back, gray)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1] 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 
    cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    for contour in cnts: 
        if cv2.contourArea(contour) < 10000: 
            continue
  
        (x, y, w, h) = cv2.boundingRect(contour) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 

    cv2.imshow("face_detect", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

camr.release() 
cv2.destroyAllWindows() 