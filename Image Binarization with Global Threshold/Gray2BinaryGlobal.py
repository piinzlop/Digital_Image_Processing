import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot as plt

# Global Variable
threshold_value = 180

source_img = np.zeros((10,10,3), dtype=np.uint8)
adjusted_img = np.zeros((10,10,3), dtype=np.uint8)
hist_img = np.zeros((10,10,3), dtype=np.uint8)
 
def handler_adjustThreshold(x):
    global threshold_value
    global source_img,adjusted_img,hist_img
    threshold_value = cv.getTrackbarPos('threshold','Binary')
    print(f"Threshold Value = {threshold_value}")
   
   # ปรับภาพโดยใช้ภาพต้นฉบับมาเก็บไว้ในตัวแปร adjusted_img ใช้ cv.เข้ามาช่วย (รูปต้นฉบับ, ค่าตัวแปรเริ่มต้น, ค่าตัวแปรสูงสุด, โหมดไบนารี่ ขาว/ดำ)
    retve, adjusted_img = cv.threshold(source_img, threshold_value, 255, cv.THRESH_BINARY) 

    # Update histogram
    histSize = 256
    histRange = (0, 256) # the upper boundary is exclusive
    accumulate = False
    gray_hist = cv.calcHist(source_img, [0], None, [histSize], histRange, accumulate=accumulate)
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv.normalize(gray_hist, gray_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    for i in range(1, histSize):
        cv.line(hist_img, ( bin_w*(i-1), hist_h - int(gray_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(gray_hist[i]) ),
                ( 255, 0, 0), thickness=2)
    cv.line(hist_img,(threshold_value*2,0),(threshold_value*2,hist_h-1),(255,255,255),3)
 
def main():
    global threshold_value
    global source_img,adjusted_img,hist_img
 
    if(len(sys.argv)>=2):
        source_img = cv.imread(str(sys.argv[1]))
    else :
        source_img = cv.imread("./test.jpg", 1)
 
    source_img = cv.cvtColor(source_img,cv.COLOR_BGR2GRAY) # convert to GrayScale
 
    #named windows
    cv.namedWindow("Original", cv.WINDOW_NORMAL)
    cv.namedWindow("Binary", cv.WINDOW_NORMAL)
    cv.namedWindow("Histogram", cv.WINDOW_NORMAL)
 
    #create trackbar
    cv.createTrackbar('threshold', 'Binary', threshold_value, 255, handler_adjustThreshold)

    adjusted_img  = source_img.copy()
    handler_adjustThreshold(-1); # ตั้งค่าให้เมื่อเปิดโปรแกรมขึ้นมาจะทำการคำนวณเลย
    
    while(True):
        cv.imshow("Original",source_img)
        cv.imshow("Binary",adjusted_img)
        cv.imshow("Histogram",hist_img)
        key = cv.waitKey(100)
        if(key==27): #ESC = Exit Program
            break
 
    cv.destroyAllWindows()
 
if __name__ == "__main__":
    main()