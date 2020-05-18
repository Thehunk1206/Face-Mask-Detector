import cv2
import numpy as np

conf_thresh = 0.5

caffe_prototxt = "assets/deploy.prototxt"
caffe_model = "assets/res10_300x300_ssd_iter_140000_fp16.caffemodel"

print("loading model....")
net = cv2.dnn.readNetFromCaffe(caffe_prototxt,caffe_model)

img = cv2.imread("sampleImages/facemask.jpg")
(h,w) = img.shape[:2]

resized_img = cv2.resize(img,(300,300))

imgblob = cv2.dnn.blobFromImage(resized_img,1.0,(300,300),(104.0, 177.0, 123.0))

net.setInput(imgblob)

detection = net.forward()

for i in range(0,detection.shape[2]):
    confidence = detection[0][0][i][2]
    if confidence > conf_thresh:
        
        box = detection[0][0][i][3:7]*np.array([w,h,w,h])
        (start_x,start_y,end_x,end_y) = box.astype("int")
        print(f"face:{i} Confidence: {confidence*100}")
        y = start_y - 10 if start_y - 10 > 10 else start_y + 10

        text = "{:.2f}%".format(confidence*100)
        cv2.rectangle(img,(start_x,start_y),(end_x,end_y),(0,0,255),2)
        cv2.putText(img, text,(start_x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

cv2.imshow("out",img)
cv2.waitKey(0)
