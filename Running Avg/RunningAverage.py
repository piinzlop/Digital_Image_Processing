import cv2 as cv
import numpy as np
 
def handler_thresh(x):
    thresh_value = cv.getTrackbarPos('Threshold','motion')
    alpha_percent = cv.getTrackbarPos('Alpha','motion')
    print(f"thresh_value:{thresh_value} / alpha_percent:{alpha_percent}")
 
def main():
    thresh_value = 4 
    alpha_percent = 50  

    cv.namedWindow('motion',cv.WINDOW_NORMAL)  
    cv.createTrackbar('Threshold', 'motion', thresh_value, 255, handler_thresh) 
    cv.createTrackbar('Alpha', 'motion', alpha_percent, 100, handler_thresh)  

    vdofile = 'depth.avi'  
    cap = cv.VideoCapture(vdofile)  
    if not cap.isOpened():
        print("Cannot open vdo")
        exit()
    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    background = frame.copy()
    
    while True:
        thresh_value = cv.getTrackbarPos('Threshold','motion')
        alpha_percent = cv.getTrackbarPos('Alpha','motion')
        alpha = alpha_percent/100
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        motion_no_thres = np.abs(frame - background) #  |current - old[t-xx]| > Th -> is Motion parts
        ret,motion = cv.threshold(motion_no_thres, thresh_value, 255, cv.THRESH_BINARY)
        
        # Running average ใช้สมการจำค่าเฟรมที่เข้ามา ไปเป็น background
        background = cv.addWeighted(frame, alpha, background, (1-alpha), 0)  

        cv.imshow('frame', frame)
        cv.imshow('motion', motion)
        cv.imshow('background', background)
        if cv.waitKey(20) == 27:
            break
    cap.release()
 
if __name__ == "__main__":
    main()