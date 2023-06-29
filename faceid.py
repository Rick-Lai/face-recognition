import cv2
import psutil
import time
from threading import Thread
from queue import Queue

q=Queue(maxsize = 30)
q2=Queue(maxsize = 30)

def Getframe():
    while True:
        #讀取一張影像
        ret, frame = cam.read()      
        if ret:
            #frame縮小1/4省資源
            #frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            if q.full() is False:
                q.put(frame)                              
        else:
            # 若沒有影像，跳出迴圈
            print("No Signal, press q to quit")
            break
        
def Processframe():
    count = 0
    while True:  
        frame = q.get()          
        faces = face_detector.detectMultiScale(frame, 1.3, 5)
        #人臉辨識 & 繪製方框
        for(x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            count += 1
            print('第 ' + str(count) + ' 次偵測到人, ' + ' 這次偵測到 ' + str(len(faces)) + ' 個人')
        if q2.full() is False:            
            q2.put(frame)   

def WatchCPU():
    while True:
        print('CPU使用率: ' + str(psutil.cpu_percent()) + ' %')
        time.sleep(10)


if __name__ == '__main__': 
    #視訊鏡頭
    cam = cv2.VideoCapture(0)
    #呼叫人臉辨識模型
    face_detector = cv2.CascadeClassifier('D:/faceid/face_detect.xml')
    m1 = Thread(target = Getframe, daemon = True)
    m2 = Thread(target = Processframe, daemon = True)
    m3 = Thread(target = WatchCPU, daemon = True)
    m1.start()
    m2.start()
    m3.start()
    # 建立視窗&調整視窗大小
    cv2.namedWindow('camview', cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('camview', 800, 450)
    #畫面顯示
    while True:
        if q2.empty() is False:
            #還原frame大小
            #frame = cv2.resize(q2.get(), (0, 0), fx=4.0, fy=4.0)
            cv2.imshow('camview',q2.get())
        if cv2.waitKey(10) == ord('q'):
            break  
    # 釋放資源
    cam.release()
    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()
    print('程式退出')