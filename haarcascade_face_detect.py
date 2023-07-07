import cv2
import time

if __name__ == '__main__':
    #視訊鏡頭
    cam = cv2.VideoCapture(0)

    #呼叫人臉辨識模型
    face_detector = cv2.CascadeClassifier('models\haarcascade_frontalface_default.xml')

    # 建立視窗&調整視窗大小
    cv2.namedWindow('camview', cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('camview', 800, 450)

    while True:
        ret, frame = cam.read()

        if ret:
            
            start = time.time()

            faces = face_detector.detectMultiScale(frame, 1.2, 5)

            for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            
            end = time.time()

            fps = 1 / (end - start)
            fps = str(int(fps))
            
            cv2.putText(frame, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('camview',frame)
            
            if cv2.waitKey(10) == ord('q'):
                break 

        else:
            # 若沒有影像，跳出迴圈
            print("No Signal")
            break
        
    # 釋放資源
    cam.release()
    
    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()
    print('程式退出')