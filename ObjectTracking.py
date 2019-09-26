#Francesco Pecora -- 01/30/2019

'''
This program tracks the position of an object using its color.
Two variables define the range of the color in HSV mode. The 
color initialized will be the one tracked by the camera. 
'''

import cv2

#initializing the range of the color to be recognised
lower_green = (23, 50, 50)
upper_green = (37, 255, 255)

#cap rapresents our USB camera
cap = cv2.VideoCapture(0)

def process_frame(frame):
    
    #resizing the frame to make the processing faster
    resize_frame = cv2.pyrDown(frame.copy())
    
    #removing some noise blurring the image
    blurred = cv2.GaussianBlur(resize_frame.copy(), (11,11), 0)
    
    #turning the bgr image into hsv image
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    #finding the mask of the object (the camera only sees its shape)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    #removing noise from the mask by eroding and dilating the shape
    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 2)
    
    #scaling the frame back up
    resize_frame = cv2.pyrUp(mask)
    
    return resize_frame
     

while True:
    
    ok, frame = cap.read()
    
    while not ok:
        print("Frame not found.")
        ok, frame = cap.read()
    
    #processing the frame using the function previously built
    processed_frame = process_frame(frame)
    
    #finding the contours of the object
    video_contours, hierarchy = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
    #sorting the contours to only show the bigger ones
    sorted_contours = sorted(video_contours, key = cv2.contourArea, reverse = True)
    
    #if the contour is not found, then we just display the frame as it is
    if len(video_contours) < 1:
        cv2.imshow("Live", frame)
    
    #if we find a contour, then we draw it on the frame
    else:        
        cv2.drawContours(frame, sorted_contours[:2], -1, (255,255,0), 4)        
        cv2.imshow("Live", frame)
        
    
    #if we press the key "q", we break from the while loop
    key = cv2.waitKey(1) & 0xFF    
    if key == ord("q"):
        break
    
  
cap.release()
cv2.destroyAllWindows()
























