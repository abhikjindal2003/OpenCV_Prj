import cv2, time
import numpy as np

camr = cv2.VideoCapture(0)

while True:
    check,frame = camr.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_l = np.array([136, 87, 111], np.uint8) 
    red_u = np.array([180, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsv, red_l, red_u)

    green_l = np.array([25, 52, 72], np.uint8) 
    green_u = np.array([102, 255, 255], np.uint8) 
    green_mask = cv2.inRange(hsv, green_l, green_u) 

    blue_l = np.array([94, 80, 2], np.uint8) 
    blue_u = np.array([120, 255, 255], np.uint8) 
    blue_mask = cv2.inRange(hsv, blue_l, blue_u) 

    kernel = np.ones((5, 5), np.uint8) 
      
    # For red color 
    red_mask = cv2.dilate(red_mask, kernel) 
    res_red = cv2.bitwise_and(frame, frame, mask = red_mask) 
      
    # For green color 
    green_mask = cv2.dilate(green_mask, kernel) 
    res_green = cv2.bitwise_and(frame, frame, mask = green_mask) 
      
    # For blue color 
    blue_mask = cv2.dilate(blue_mask, kernel) 
    res_blue = cv2.bitwise_and(frame, frame, mask = blue_mask) 

    cnts_red,_ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    for _,contour in enumerate(cnts_red): 
        if cv2.contourArea(contour) < 300: 
            continue
  
        (x, y, w, h) = cv2.boundingRect(contour) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3) 

    cnts_green,_ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    for _,contour in enumerate(cnts_green): 
        if cv2.contourArea(contour) < 300: 
            continue
  
        (x, y, w, h) = cv2.boundingRect(contour) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 

    cnts_blue,_ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    for _,contour in enumerate(cnts_blue): 
        if cv2.contourArea(contour) < 300: 
            continue
  
        (x, y, w, h) = cv2.boundingRect(contour) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3) 

    cv2.imshow("Colours", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

camr.release() 
cv2.destroyAllWindows() 