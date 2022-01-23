import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot as plt

# Global Variable
threshold_value = 180 # ค่า threshold เริ่มต้น
window_size = 1 # window_size * 2 + 1
c_value = 10 

source_img = np.zeros((10,10,3), dtype=np.uint8)
adjusted_img = np.zeros((10,10,3), dtype=np.uint8) # Global Threshold
adjusted_mean_img = np.zeros((10,10,3), dtype=np.uint8) # Local Threshold Mean ลบ c 
adjusted_gaussian_img = np.zeros((10,10,3), dtype=np.uint8) # Global Threshold GaussianWeightSum ลบ c 
hist_img = np.zeros((10,10,3), dtype=np.uint8)

def handler_adjustThreshold(x):
    global threshold_value,window_size,c_value
    global source_img,adjusted_img,hist_img,adjusted_mean_img,adjusted_gaussian_img
    threshold_value = cv.getTrackbarPos('threshold','Binary')
    # 3 5 7 9 11 13 15 17 ..... 41 <- block size 
    window_size = cv.getTrackbarPos('window_size','Binary') * 2 + 1 # ทำให้เป็นเลขคี่โดยการเอาค่ามา *2 +1 = เลขคี่
    c_value = cv.getTrackbarPos('c_value','Binary') - 10 #เอาค่าตัวแปรมาลบ 10
    print(f"Global Threshold Value = {threshold_value}")
    print(f"window_size = {window_size}")
    print(f"c_value = {c_value}")

   # ปรับภาพโดยใช้ภาพต้นฉบับมาเก็บไว้ในตัวแปร adjusted_img ใช้ cv.เข้ามาช่วย (รูปต้นฉบับ, ค่าตัวแปรเริ่มต้น, ค่าตัวแปรสูงสุด, โหมดไบนารี่ ขาว/ดำ)
    retve, adjusted_img = cv.threshold(source_img, threshold_value, 255, cv.THRESH_BINARY) 
    adjusted_mean_img = cv.adaptiveThreshold(source_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, window_size, c_value)
    adjusted_gaussian_img = cv.adaptiveThreshold(source_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, window_size, c_value)

    # ทำกราฟ histogram
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
    global threshold_value,window_size,c_value
    global source_img,adjusted_img,hist_img,adjusted_mean_img,adjusted_gaussian_img
 
    if(len(sys.argv)>=2):
        source_img = cv.imread(str(sys.argv[1]))
    else :
        source_img = cv.imread("./test.jpg", 1)
 
    source_img = cv.cvtColor(source_img,cv.COLOR_BGR2GRAY) # convert to GrayScale
 
    #สร้างหน้าต่างมารองรอบ
    cv.namedWindow("Original", cv.WINDOW_NORMAL)
    cv.namedWindow("Binary", cv.WINDOW_NORMAL)
    cv.namedWindow("BinaryMean", cv.WINDOW_NORMAL)
    cv.namedWindow("BinaryGaussian", cv.WINDOW_NORMAL)
    cv.namedWindow("Histogram", cv.WINDOW_NORMAL)
 
    #สร้าง trackbar
    cv.createTrackbar('threshold', 'Binary', threshold_value, 255, handler_adjustThreshold)
    cv.createTrackbar('window_size', 'Binary', window_size, 20, handler_adjustThreshold)
    cv.createTrackbar('c_value', 'Binary', c_value, 20, handler_adjustThreshold)
    adjusted_img = source_img.copy()
    handler_adjustThreshold(-1); # ตั้งค่าให้เมื่อเปิดโปรแกรมขึ้นมาจะทำการคำนวณเลย
    
    while(True):
        cv.imshow("Original",source_img)
        cv.imshow("Binary",adjusted_img)
        cv.imshow("BinaryMean",adjusted_mean_img)
        cv.imshow("BinaryGaussian",adjusted_gaussian_img)
        cv.imshow("Histogram",hist_img)
        key = cv.waitKey(100)
        if(key==27): #ESC = Exit Program
            break
 
    cv.destroyAllWindows()
 
if __name__ == "__main__":
    main()