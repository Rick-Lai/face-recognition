import cv2 as cv
import time
import numpy as np

if __name__ == '__main__':
    
    #視訊鏡頭
    cam = cv.VideoCapture(0)

    #呼叫人臉辨識模型
    model = cv.dnn.readNetFromCaffe('models\deploy.prototxt', 'models\\res10_300x300_ssd_iter_140000.caffemodel')

    # Set backend and target to CUDA to use GPU
    model.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    # 建立視窗&調整視窗大小
    cv.namedWindow('camview', cv.WINDOW_FREERATIO)
    cv.resizeWindow('camview', 800, 450)

    while True:
        ret, frame = cam.read()

        if ret:
            
            start = time.time()   

            (h, w) = frame.shape[:2]        

            # 建立模型使用的Input資料blob (比例變更為300 x 300)
            blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), True)

            model.setInput(blob)

            output = model.forward()

            for i in range(0, output.shape[2]):

                confidence = output[0, 0, i, 2]

                if confidence > 0.85:

                    box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
                    
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    text = "{:.2f}%".format(confidence * 100)
                                      
                    if startY - 10 > 10:
                        y = startY - 10 

                    else:
                        y = startY + 10
                    
                    cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv.putText(frame, text, (startX, y),cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
           
            end = time.time()
            fps = 1 / (end - start)
            fps = str(int(fps))
          
            cv.putText(frame, fps, (7, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv.LINE_AA)
            cv.imshow('camview',frame)
            
            if cv.waitKey(10) == ord('q'):
                break 
        
    # 釋放資源
    cam.release()
    
    # 關閉所有 OpenCV 視窗
    cv.destroyAllWindows()
    print('程式退出')