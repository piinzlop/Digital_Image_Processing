import math
import cv2 as cv
import numpy as np
from numpy import random

capture = cv.VideoCapture('CarsDrivingUnderBridge.mp4')
#capture = cv.VideoCapture('circle_move.mp4')


backSub = cv.createBackgroundSubtractorMOG2(history=None,varThreshold=None,detectShadows=True)
#Print default value
print("===========SHOW DEFAULT BackgroundSubtractor PARAMETER===========") 
print(f"getHistory={backSub.getHistory()}\ngetNMixtures={backSub.getNMixtures()}\ngetDetectShadows={backSub.getDetectShadows()}\nvarThreshold={backSub.getVarThreshold()}")
#backSub.setHistory(600) # Amount of mean sigma of BG
#backSub.setNMixtures(2) # Number of Gaussian Distribution
backSub.setDetectShadows(True) # Detect Shadow
backSub.setVarThreshold(100) # Threshold Motion

class TrackedBlob:
    id = None
    color = None
    alive = True # if this flag false = marked to remove
    def __init__(self,blob_id,xywh):
        self.id = blob_id
        (self.x,self.y,self.w,self.h) = xywh
        c = random.randint(255, size=(3))
        self.color = [c[0].item(),c[1].item(),c[2].item()]
        self.path = []
        self.nlost = 0
        self.allLife = 1
        self.matchedStatus = True   ## สถานะของวัตถุที่ถูกแมต
        self.alreadyCounted = False # use for count function
    def XYWH(self): ## ฟังก์ชันส่งค่า x y w h ที่ถูกแท็ก ว่าวัตถุอยู่ตรงไหน
        return [self.x,self.y,self.w,self.h]
    def calDiffBlobWithContour(self, contour_xywh, weightPosition = 0.5, weightScale = 0.5):    ## หาค่าความแตกต่างระหว่าง Blob
        (xContour,yContour,wContour,hContour) = contour_xywh # incoming object
        return (math.sqrt((xContour - self.x)**2 + (yContour - self.y)**2) * weightPosition) + (math.sqrt((wContour - self.w)**2 + (hContour - self.h)**2) * weightScale)
    def updateBlob(self, status, xywh=[0,0,0,0]):
        if (status): # Matched
            self.nlost = 0
            self.allLife += 1
            self.path.append(xywh)
            (self.x,self.y,self.w,self.h) = xywh # update corrent position
        else : # Not Matched
            self.nlost += 1
            if(self.nlost >= 4): # if not matched xx frame will be marked to remove ไม่มีชีวิต จะใช้ลบวัตถุที่ตาย
                self.alive = False      
    def setMatched(self):
        self.matchedStatus = True
    def resetMatched(self):
        self.matchedStatus = False
    def getMatched(self):
        return self.matchedStatus
    def getNLost(self):     ## ทราบว่าวัตถุหายไปกี่ครั้งแล้ว
        return self.nlost
    def setCounted(self):   ## วัตถุถูกนับแล้วยัง
        self.alreadyCounted = True
    def getCounted(self): 
        return self.alreadyCounted
    def getAllLife(self): 
        return self.allLife
    def getPath(self): 
        return self.path

class BlobTracker:
    def __init__(self, distanceThreshold = 60): ## ฟังก์ชันใช้ติดตาม
        self.TrackedBlob_Table = []     ## ทำการเก็บคุณสมบัตวัตถุที่เคยแท็กแล้ว
        self.distanceThreshold = distanceThreshold
        self.lastID = 0
    def setDistanceThreshold(self, distanceThreshold = 60): ## ค่าระยะห่าง ถ้าตั้งค่าน้อยจะสร้างวัตถุใหม่ , แต่ถ้ามากจะไม่สร้างเป็นวัตถุใหม่ นับอยู่ที่เดิม 
        self.distanceThreshold = distanceThreshold
    def getTrackedBlob_Table(self):
        return self.TrackedBlob_Table
    def trackXYWHs(self, ContourXYWHs): # ContourXYWHs -> Incomming Objects Table
        # if no TrackedBlob in Table
        if(len(self.TrackedBlob_Table)==0): ## หาความยาวความกว้าง วนเป็นลูป และนับวัตถุ
            for XYWH in ContourXYWHs:
                self.addNewTrackedBlob(XYWH)
        # Matching Incomming Objects with all   วนลูปแมตทุกๆอัน
        else :  ## แต่ถ้าเจอวันถุที่ถูกแมตแล้ว 
            for XYWH in ContourXYWHs: # accessing (incoming object) in each contour to update or new tracked blob
                minDistance = 1000000000 #         -1 = not match with anyone
                minDistanceTrackedBlobID = -1 #    -1 = not match with anyone
                for TrackedBlobIter in self.TrackedBlob_Table:
                    if(not TrackedBlobIter.getMatched()): # if didnot match yet
                       diffValue = TrackedBlobIter.calDiffBlobWithContour(XYWH)
                       #print(f'diffValue {diffValue} of Blob={XYWH} with TrackedBlob#{TrackedBlobIter.id}={TrackedBlobIter.XYWH()}({TrackedBlobIter.color})')
                       if(diffValue <= self.distanceThreshold and diffValue < minDistance): # ถ้าเข้าเกณฑ์ และมีค่าน้อยสุดก็ marked ไว้ก่อน
                           minDistanceTrackedBlobID = TrackedBlobIter.id
                           minDistance = diffValue
                # if matched some TrackedBlob -> update
                if(minDistanceTrackedBlobID!=-1):
                    self.updateTrackedBlob(minDistanceTrackedBlobID, XYWH)
                else:
                    self.addNewTrackedBlob(XYWH)
        # if some TrackedBlob has not be updated -> set Lost
        for TrackedBlobIter in self.TrackedBlob_Table:
            if(not TrackedBlobIter.getMatched()): # if didnot match yet
                TrackedBlobIter.updateBlob(False)   ## set ให้หายไปแล้วกี่เฟรม จะยังไม่ถูกลบโดยทันที
        # remove all dead (not alive)
        numTrackedBlob = len(self.TrackedBlob_Table)    ## ลบวัตถุที่มันที่ตายไปแล้ว หรือหายไปจากเฟรมนานๆ  ตัวแปร alive = false
        i=0
        while i < numTrackedBlob:
            if(not self.TrackedBlob_Table[i].alive): # if didnot match yet
                self.TrackedBlob_Table.pop(i) # remove
                numTrackedBlob-=1 # move down counter
                i-=1 # move down iterator
            i+=1
        # reset all Match Status
        for TrackedBlobIter in self.TrackedBlob_Table:
            TrackedBlobIter.resetMatched()
    def addNewTrackedBlob(self, XYWH):
        self.TrackedBlob_Table.append(TrackedBlob(self.lastID,XYWH))
        self.lastID += 1    ## บวก lastID เพิ่ม
    def updateTrackedBlob(self, id, XYWH):
        for TrackedBlobIter in self.TrackedBlob_Table:
            if TrackedBlobIter.id == id:
                TrackedBlobIter.updateBlob(True,XYWH) # update
                TrackedBlobIter.setMatched() # marked already Mateched
                break
    def drawTrackedBlobs(self, image, fontSize=1.0, thickness=2, drawPath=True):    ## ฟังก์ชัน วาดกรอบและเขียนเลขบนกรอบของวัตถุที่ถูกแท็ก
        for TrackedBlobIter in self.TrackedBlob_Table:
            if TrackedBlobIter.getNLost()==0 and TrackedBlobIter.alive : # and TrackedBlobIter.getAllLife()>1'''
                color = TrackedBlobIter.color
                (x,y,w,h) = TrackedBlobIter.XYWH()
                cv.rectangle(image, (x,y), (x+w,y+h), color, thickness)
                cv.putText(image, str(TrackedBlobIter.id), (x+2,y+2), cv.FONT_HERSHEY_SIMPLEX, fontSize, color, thickness)
                # plot path
                if(drawPath):
                    histXYWHs = TrackedBlobIter.getPath()
                    histCenter = []
                    for i in range(len(histXYWHs)) :
                        (xHist,yHist,wHist,hHist) = histXYWHs[i]
                        histCenter.append((xHist+(wHist//2),yHist+(hHist//2)))
                        if(i!=0):
                            cv.line(image,histCenter[i-1],histCenter[i],color)

class BlobExtractor:
    def __init__(self):
        self.contours = None
        self.hierarchy = None
        self.XYWHs = None
        self.colors = None
    def execute(self,segmented_bin_img):    ## ให้ execute
        edge16S_img = cv.Laplacian(segmented_bin_img, cv.CV_16S, ksize=3)
        edge_img = cv.convertScaleAbs(edge16S_img)
        self.contours, self.hierarchy = cv.findContours(edge_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.XYWHs = [ cv.boundingRect(contour) for contour in self.contours]
    def filterMinArea(self,min):
        ''' filter out(remove) countours that  have area < min'''
        temp_contours = []
        for i,cnt in enumerate(self.contours):
            (x,y,w,h) = cv.boundingRect(cnt)
            if((w*h)>=min):
                temp_contours.append(cnt) # [x,y,w,h]
        self.contours = temp_contours.copy()
        self.XYWHs = [ cv.boundingRect(contour) for contour in self.contours]
    def filterInArea(self,XYWH):
        ''' filter only countour in Area XYWH'''
        temp_contours = []
        (xmin,ymin) = XYWH[:2]
        xmax = xmin + XYWH[2]
        ymax = ymin + XYWH[3]
        for i,cnt in enumerate(self.contours):
            (x,y,w,h) = cv.boundingRect(cnt)
            if( (x>=xmin and x<=xmax) and (y>=ymin and y<=ymax)):
                temp_contours.append(cnt) # [x,y,w,h]
        self.contours = temp_contours.copy()
        self.XYWHs = [ cv.boundingRect(contour) for contour in self.contours]
    def getContours(self):
        return self.contours
    def getXYWHs(self):      ## get ค่า
        return self.XYWHs

distanceThreshold = 60  
mainBlobTracker = BlobTracker(distanceThreshold); # create main tracker ทำแท็กบาร์ ที่สามารถปรับค่าได้สะดวก
cv.namedWindow('Frame',cv.WINDOW_NORMAL)
def changeDistanceThreshold(x):
    global distanceThreshold
    distanceThreshold = cv.getTrackbarPos('DistanceThreshold','Frame')
    mainBlobTracker.setDistanceThreshold(distanceThreshold)
cv.createTrackbar('DistanceThreshold', 'Frame', distanceThreshold, 400, changeDistanceThreshold)

y_startCount = 20 # เริ่มเส้นนับ
y_endCount = 200 # จบเส้นนับ
blobCount = 0 # number of blob which passed line

while True: ## อ่าน VDO 
    ret, frame = capture.read()
    id_frame = capture.get(cv.CAP_PROP_POS_FRAMES)
    if frame is None:
        break
 
    fgMask = backSub.apply(frame,learningRate=0.005) # learningRate
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(id_frame), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    _,fgMask = cv.threshold(fgMask, 200, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9,9))
    fgMask = cv.dilate(fgMask, kernel, iterations=1)

    Blob = BlobExtractor()
    Blob.execute(fgMask)
    Blob.filterMinArea(2000)
    FocusArea = [100,100,950,700] # บริเวณที่แสดง Blob
    Blob.filterInArea(FocusArea)
    

    (H,W) = fgMask.shape    ## ฟังก์ชันวาด contours
    contours_img = np.zeros((H,W,3),dtype=np.uint8)
    if(len(Blob.getContours())>=1): ## กัน contours ที่ลอย
        cv.drawContours(contours_img, Blob.getContours(), -1, (255,0,0)) ## วาด contours สีน้ำเงินแทนวัตถุ

    # plot tracking area
    (xFA,yFA,wFA,hFA) = FocusArea   ## วาดกรอบสีขาวเป็นบริเวณที่ทำการนับวัตถุ
    cv.rectangle(frame, (xFA,yFA), (xFA+wFA,yFA+hFA), (255,255,255)), cv.rectangle(contours_img, (xFA,yFA), (xFA+wFA,yFA+hFA), (255,255,255))
    cv.putText(frame, 'Tracking Zone', (xFA+5, yFA+20), cv.FONT_HERSHEY_SIMPLEX, .7 ,(255,255,255), 1),cv.putText(contours_img, 'Tracking Zone', 
    (xFA+5, yFA+20), cv.FONT_HERSHEY_SIMPLEX, .7 ,(255,255,255), 1)
    
    # plot counting line
    (fHeight,fWidth) = frame.shape[:2]  ## สองเส้น 
    ## วาดเส้นที่หน้าต่างที่มีภาพสี
    cv.putText(frame, 'Counting Zone', (50, y_startCount+30), cv.FONT_HERSHEY_SIMPLEX, 1 ,(0,255,255), 2),cv.putText(contours_img, 'Counting Zone', 
    (50, y_startCount+30), cv.FONT_HERSHEY_SIMPLEX, 1 ,(0,255,255), 2)

    cv.line(frame,(0,y_startCount),(fWidth,y_startCount),(0,255,255),2),    cv.line(frame,(0,y_endCount),(fWidth,y_endCount),(0,255,255),2)
    cv.line(contours_img,(0,y_startCount),(fWidth,y_startCount),(0,255,255),2),    cv.line(contours_img,(0,y_endCount),(fWidth,y_endCount),(0,255,255),2)

    if(id_frame>100): ## แท็กเฟรมที่ 100 เฟรมขึ้นไป
        # blob tracking
        mainBlobTracker.trackXYWHs(Blob.getXYWHs())
        mainBlobTracker.drawTrackedBlobs(frame,fontSize=1,drawPath=True) ## วาดตัวที่ถูกแท็กแล้ว 
        mainBlobTracker.drawTrackedBlobs(contours_img,fontSize=1,drawPath=True) ## วาดตัวที่ถูกแท็กแล้ว 
        # blob counting
        TrackedBlob_Table = mainBlobTracker.getTrackedBlob_Table()
        for TrackedBlobIter in TrackedBlob_Table:
            (_,y,_,_) = TrackedBlobIter.XYWH()
             # if be in y_startCount && y_endCount
            if(y>=y_startCount and y<=y_endCount):
                 # if alive, is not counted and active
                if(TrackedBlobIter.getNLost()==0 and (not TrackedBlobIter.getCounted()) and TrackedBlobIter.alive and TrackedBlobIter.getAllLife()>2):
                    blobCount+=1
                    TrackedBlobIter.setCounted()
        cv.putText(frame, 'Counter : '+str(blobCount), (500, 35), cv.FONT_HERSHEY_SIMPLEX, 1.2 , (128,128,255),2),cv.putText(contours_img, 'Counter : '+str(blobCount), 
        (500, 30), cv.FONT_HERSHEY_SIMPLEX, 1 , (128,128,255))     
    cv.imshow('Frame', frame)   ## แสดงผลรูปต้นฉบับ
    cv.imshow('FG_Mask', fgMask)    ## แสดงผลภาพขาวดำ
    cv.imshow('Contours', contours_img)  ## แสดงผลรถที่นำมานับ
    
    keyboard = cv.waitKey(10)    ## ออกจากลูปแสดงผลของโปรแกรม 
    if keyboard == 'q' or keyboard == 27:
        break
