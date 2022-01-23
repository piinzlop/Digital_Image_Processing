import cv2 as cv

# BG Substraction Algorithm
backSub = cv.createBackgroundSubtractorMOG2(history=None,varThreshold=None,detectShadows=True)
#backSub = cv.createBackgroundSubtractorKNN()


capture = cv.VideoCapture('CarsDrivingUnderBridge.mp4')
#capture = cv.VideoCapture('depth.avi')

#Print default value
print("===========SHOW DEFAULT PARAMETER===========")
print(f"getHistory={backSub.getHistory()}")
print(f"getNMixtures={backSub.getNMixtures()}")
print(f"getDetectShadows={backSub.getDetectShadows()}")
print(f"varThreshold={backSub.getVarThreshold()}")
#backSub.setHistory(600) # Amount of mean sigma of BG
#backSub.setNMixtures(2) # Number of Gaussian Distribution
backSub.setDetectShadows(True) 
backSub.setVarThreshold(100) 

ret, frame = capture.read()
w = frame.shape[1]
h = frame.shape[0]
fps = 20   
fourcc = cv.VideoWriter_fourcc('d','i','v','x')   
VDO_writer = cv.VideoWriter("lab17.avi", fourcc, 20, (w, h))  

while True:         
    ret, frame = capture.read()
    id_frame = capture.get(cv.CAP_PROP_POS_FRAMES)
    if frame is None:
        break

    fgMask = backSub.apply(frame,learningRate=0.005) # learningRate
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(id_frame), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    _,fgMask = cv.threshold(fgMask, 200, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9,9))
    fgMask = cv.dilate(fgMask, kernel, iterations=1)

    fgMask3C = cv.cvtColor(fgMask, cv.COLOR_GRAY2BGR)
    VDO_writer.write(fgMask3C);

    cv.imshow('Frame', frame)   
    cv.imshow('FG Mask', fgMask)  
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

VDO_writer.release()  