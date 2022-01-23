# Brightness/Contrast Adjustment -> https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
# Trackbar -> https://www.life2coding.com/change-brightness-and-contrast-of-images-using-opencv-python/
# Python -> https://www.tutorialspoint.com/python/python_command_line_arguments.htm

import cv2 as cv  # import library opencv โดยใช้ชื่อเล่นว่า " cv "
import numpy as np # import library numpy โดยใช้ชือเล่นว่า " np "
import sys # import library sys ใช้ในการรับ argument ภายนอกจาก command line หรือ คือการรับ path ของภาพ 
from matplotlib import pyplot as plt # import library matplotlib ในส่วนย่อย pyplot โดยใช้ชื่อเล่นว่า " plt "
# Global Variables
beta_brightness_value = 100 # เป็นตัวแปรใช้ปรับค่าความสว่าง (brightness)  100 = ค่าจริงคือ 0 / ถ้า 0 = ค่าจริงคือ -100
alpha_contrast_value = 10 # เป็นตัวแปรใช้ในการปรับค่าความต่างของสี (contrast)
source_img = np.zeros((10,10,3), dtype=np.uint8) # ตัวแปร Array ใช้ library numpy 2 มิติ (10,10,3)-> 10col,10row,3แผ่น-RGB ใช้เก็บภาพที่นำเข้ามา ส่วน dtype ย่อมาจาก data type 
adjusted_img = np.zeros((10,10,3), dtype=np.uint8) # ตัวแปรใช้เก็บภาพ result ที่ปรับ beta กับ alpha
hist_img = np.zeros((10,10,3), dtype=np.uint8) # ตัวแปรใช้ใช้เก้บภาพ histogram

def handler_adjustAlphaBeta(x): # def ย่อมาจาก ดีฟายฟังก์ชั่น เป็นการใช้ฟังก์ชั่น handler_adjustAlphaBeta  แต่ตัวแปร (x) ในฟังก์ชั่นใส่ไว้ให้ถูกตามหลัก syntax ถ้าไม่มีจะ error 
    global beta_brightness_value,alpha_contrast_value # ประกาศตัวแปรนอกฟังก์ชั่น 2 ตัวแปร
    global source_img,adjusted_img,hist_img # ประกาศตัวแปรนอกฟังก์ชั่น 3 ตัวแปร
    beta_brightness_value = cv.getTrackbarPos('beta','BrightnessContrast') # คือการนำค่าที่อยู่บน trackbar ของ beta ที่อยู่ในหน้าต่าง BrightnessContrast เข้าไปเก็บในตัวแปร beta_brightness_value หรือคือการปรับค่า ความส่าง
    alpha_contrast_value = cv.getTrackbarPos('alpha','BrightnessContrast') # คือการนำค่าที่อยู่บน trackbar ของ alpha ที่อยู่ในหน้าต่าง BrightnessContrast เข้าไปเก็บในตัวแปร alpha_contrast_value หรือคือการปรับค่า ความเข้มของสี
    alpha = alpha_contrast_value / 10 # เป็นการคำนวนการปรับค่าของตัวแปร alpha_contrast_value หาร 10 แล้วที่เก็บไว้ในตัวแปร alpha
    beta = int(beta_brightness_value - 100) # เป็นการคำนวนการปรับค่าของตัวแปร beta_brightness_value ลบ 100 แล้วที่เก็บไว้ในตัวแปร beta
    print(f"alpha={alpha} / beta={beta}")
    
    ## loop access each pixel -> too slow      จะเป็นการประมวลผลที่เข้าถึงทีละตัวแปรที่ละพิกเซล จึงทำให้การประมวลผลช้ามาก
    ''' for y in range(source_img.shape[0]): 
        for x in range(source_img.shape[1]):
            for c in range(source_img.shape[2]):
                adjusted_img[y,x,c] = np.clip( alpha * source_img[y,x,c] + beta , 0, 255)
    '''
    # for better performance, pls use -> dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    adjusted_img = cv.convertScaleAbs(source_img, alpha=alpha, beta=beta)
                        
    # Update histogram   การสร้าง histogram
    bgr_planes = cv.split(adjusted_img)  # คำสั่งแยกภาพให้มี 3แผ่น -> RGB
    histSize = 256  
    histRange = (0, 256) # the upper boundary is exclusive
    accumulate = False  
    b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate) # คำสั่งในการคำนวณ histogram ในแชแนลแผ่นสีน้ำเงิน B 
    g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate) # คำสั่งในการคำนวณ histogram ในแชแนลแผ่นสี้ขียว G
    r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate) # คำสั่งในการคำนวณ histogram ในแชแนลแผ่นสีแดง R
    hist_w = 512 # กำหนดความหนา weight
    hist_h = 400 # กำหนดความสูง hight
    bin_w = int(round( hist_w/histSize ))
    hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX) # คำสั่งการทำ normalize -> 0 - 1 ในแชแนลแผ่นสีน้ำเงิน B ให้ขนาดของกราฟไม่หลุดขอบของหน้าต่างแสดงผล 
    cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX) # คำสั่งการทำ normalize -> 0 - 1 ในแชแนลแผ่นสี้ขียว G ให้ขนาดของกราฟไม่หลุดขอบของหน้าต่างแสดงผล 
    cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX) # คำสั่งการทำ normalize -> 0 - 1 ในแชแนลแผ่นสีแดง R ให้ขนาดของกราฟไม่หลุดขอบของหน้าต่างแสดงผล 
    for i in range(1, histSize):
        cv.line(hist_img, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(b_hist[i]) ),
                ( 255, 0, 0), thickness=2)
        cv.line(hist_img, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(g_hist[i]) ),
                ( 0, 255, 0), thickness=2)
        cv.line(hist_img, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(r_hist[i]) ),
                ( 0, 0, 255), thickness=2)


def main(): # ในฟังก์ชั่น main() ใช้จะบอกรายการของตัวแปรที่เรียกใช้ใน global นอกฟังก์ชั่น คือมี 5 ตัวแปร
    global beta_brightness_value,alpha_contrast_value # เรียกใช้ 2 ตัวแปร
    global source_img,adjusted_img,hist_img # เรียกใช้ 3 ตัวแปร

    if(len(sys.argv)>=2): # การ input ภาพ เพื่อใช้เปิดภาพ ที่ถูกครอบด้วยตัวแปร len หรือ เล้ง
        source_img = cv.imread(str(sys.argv[1])) # จะเพิ่มมาต่อเมื่อเรากด space bar แล้วพิมพ์ชื่อ ภาพ/ไฟล์ จะเป็นการเรียกใช้รูปภาพจากชื่อภาพที่ไหนก็ได้
    else : # แต่ถ้า ไม่มี argv[1] จะไปบรรทัดข้างล่าง 
        source_img = cv.imread("./test.jpg", 1) # อ่านจาก string ที่ใส่ชื่อใน Code คือ ./output.png จะเป็นการเรียกใช้รูปภาพจากชื่อภาพที่อยู่ใน Code             

    #source_img = cv.cvtColor(source_img,cv.COLOR_BGR2GRAY) # convert to GrayScale

    #named windows
    cv.namedWindow("Original", cv.WINDOW_NORMAL) # สร้างหน้าต่าง (WINDOW_NORMAL ปรับขนาด Auto size หรือ สามารภปรับหน้าต่างขยับ ยืด หด ได้)
    cv.namedWindow("BrightnessContrast", cv.WINDOW_NORMAL) # สร้างหน้าต่าง
    cv.namedWindow("Histogram", cv.WINDOW_NORMAL) # สร้างหน้าต่าง

    #create trackbar
    # ใช้สร้าง trackbar ปรับค่า beta ที่อยู่ในหน้าต่าง BrightnessContrast , beta_brightness_value คือการปรับค่าของ Brightness , 200 คือค่าเริ่มต้นของความสว่าง , เมื่อ trackbar มีการเปลี่ยนแปรงจะเรียกใช้ฟังก์ชั่น handler_adjustAlphaBeta
    cv.createTrackbar('beta', 'BrightnessContrast', beta_brightness_value, 200, handler_adjustAlphaBeta) 

    # ใช้สร้าง trackbar ปรับค่า alpha ที่อยู่ในหน้าต่าง BrightnessContrast , beta_brightness_value คือการปรับค่าของ contrast , 50 คือค่าเริ่มต้นของความต่างสี , เมื่อ trackbar มีการเปลี่ยนแปรงจะเรียกใช้ฟังก์ชั่น handler_adjustAlphaBeta
    cv.createTrackbar('alpha', 'BrightnessContrast', alpha_contrast_value, 50, handler_adjustAlphaBeta) 

    adjusted_img = source_img.copy() # จะเป็นการสร้างภาพอีกตัวขึ้นมาที่คล้ายคลึงกับรูปต้นฉบับ โดยใช้คำสั่ง .copy() มาในตัวแปร adjusted_img จะไม่ยุ่งเกี่ยวกับภาพต้นฉบับ **แต่ถ้าหากไม่มีคำสั่ง .copy() จะเรียกใช้ตัวแปรภาพต้นฉบับแทน

    while(True): # ใช้วน Loop ไม่มีที่สิ้นสุด เป็นคำสั่ง output ใช้แสดงภาพ/ค่าต่างๆ 
        cv.imshow("Original",source_img) # การแสดงตัวแปร source_img ในหน้าต่าง Original
        cv.imshow("BrightnessContrast",adjusted_img) # การแสดงตัวแปร adjusted_img ในหน้าต่าง BrightnessContrast
        cv.imshow("Histogram",hist_img) # การแสดงตัวแปร hist_img ในหน้าต่าง Histogram
        key = cv.waitKey(100) # เป็นทั้งคำสั่ง plot ของ imshow ก่อนหน้าถ้าไม่มีคำสั่งนี้จจะไม่แสดงภาพจากตัวแปรก่อนหน้า และยังเป็นคำสั่งที่รอผู้ใช้กดคีย์ 100ms ถ้าไม่กดก็จะลงไปยังคำสั่งข้างล่าง
        if(key==27): #ESC = Exit Program  27 คือ ปุ่ม esc ในภาษาแอสกี้ 
            break # เมื่อกดแล้วจะเป้นคำสั่งปิดโปรแกรม

    cv.destroyAllWindows() # คำสั่งทำลายหน้าต่างที่เกี่ยวข้องของโปรแกรมนี้ทั้งหมด เป็นคำสั่งลบหน่วยความจำที่โปรแกรมใช้

if __name__ == "__main__": # ถ้าตัวแปร __name__ เท่ากับตัวแปร "__main__" จะใช้ฟังก์ชั่น main()
    main()