import cv2
import numpy as np
from numpy import argmax


def get_class(outs,height, width):
    class_ids = []                              # all id of classs save here
    confidences = []                            # confidences
    boxes = []                                  # bbox of yolo

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:                        # set up the confidence
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)          #convert it to correct size
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)  # anchor confident set


    return boxes, indexes, class_ids, confidences

weights_path = r"/Users/mohsindyer/PycharmProjects/detect_face_mask/model/yolov3_custom_final.weights"
cfg_path = r"/Users/mohsindyer/PycharmProjects/detect_face_mask/model/yolov3_custom.cfg"
classes_name_path = r"/Users/mohsindyer/PycharmProjects/detect_face_mask/model/obj.names.txt"
video_path = r"/Users/mohsindyer/PycharmProjects/detect_face_mask/data/video_facemask.mp4"

# cap = cv2.VideoCapture(video_path)
# if (cap.isOpened()== False):
#     print("Error opening video stream or file")

net = cv2.dnn.readNet(weights_path, cfg_path)       # load weight and cfg of yolo
classes = []
with open(classes_name_path, "r") as f:         # open file class name to read classes in obj.name
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames() #get layer of yolo network
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
np.random.seed(99)
colors = np.random.uniform(0, 255, size=(len(classes), 3))  # random color to draw bounding box in the webcam
font = cv2.FONT_HERSHEY_PLAIN   #font text to write text in the webcam result

cap = cv2.VideoCapture(0)   #this line to opencv the webcam
while True:     # while true, always read from camera after it is opened
# while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        copy = frame.copy()     # now each frame is like an image, frame=image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width, channels = frame.shape
        # print(height, width)

        blob = cv2.dnn.blobFromImage(frame,scalefactor=1/255, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
        # this line to feed input image, 1/255 rescale it then resize to 416, 416 like the model config
        net.setInput(blob)
        outs = net.forward(output_layers)
        # print(outs)
        boxes, indexes, class_ids, confidences = get_class(outs, height, width)
        # forward image to the model to detect, after that we will have class_name,
        # coordinate of bbox and confidence
        #print("end frame")
        # code below is to draw box, write text on the result like: label, confidence
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence_label = int(confidences[i] * 100)
                try:
                    color = colors[i]
                except:
                    pass
                cv2.rectangle(copy, (x, y), (x + w, y + h), color, 4)
                cv2.putText(copy, f'{label, confidence_label}', (x-95, y - 25), font, 2, color, 4)
                # cv2.putText
            # copy = cv2.resize(copy,(840,840))
            cv2.imshow('img', copy) # show frame of camera
        if cv2.waitKey(1) == 27:
            break

cap.release()
# Closes all the frames
cv2.destroyAllWindows()

