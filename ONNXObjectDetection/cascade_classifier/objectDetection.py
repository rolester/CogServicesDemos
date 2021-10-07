#pip install opencv-python==3.4.3.18
#pip install azureml.core
#pip install onnxruntime
from __future__ import print_function
import cv2 as cv
import cv2 as cv2
import numpy as np
import argparse
from azureml.core.model import Model
import onnxruntime
from object_detection import ObjectDetection
from PIL import Image, ImageDraw
import json

devicename = 0

#LABELS_FILENAME = 'labelsMob.txt'
#model = 'mobilephone.onnx'

LABELS_FILENAME = 'labelswater.txt'
model = 'modelwater.onnx'

#LABELS_FILENAME = 'labelsfruit.txt'
#model = 'modelfruit.onnx'

#model = Model.get_model_path(model_name = 'fruit6.onnx')
model = Model.get_model_path(model_name = model)
onnx_session = onnxruntime.InferenceSession(model, None)

with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]


class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels):
        super(ONNXRuntimeObjectDetection, self).__init__(labels)
        self.session = onnxruntime.InferenceSession(model_filename)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'
        
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))

        if self.is_fp16:
            inputs = inputs.astype(np.float16)

        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1,2,0)).astype(np.float32)

def detectAndDisplay(frame):

    cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    pil_im = Image.fromarray(cv2_im)

    od_model = ONNXRuntimeObjectDetection(model, labels)

    predictions = od_model.predict_image(pil_im)

    i = 0
    sizeofList = len(predictions) 
    while i < sizeofList :
        textlabel = predictions[i].get("tagName")
        textlabel = textlabel + " - " + str(round(predictions[i].get("probability"), 2))
        if (predictions[i].get("probability")) > .4 :
            #if pred over x then draw boc
            pline = (predictions[i].items())
            for item,value, in pline:
                if item == "boundingBox":
                    x = int(value.get("left") *650)
                    y = int(value.get("top") *500)
                    w = int(value.get("width") *650)
                    h = int(value.get("height") *500)
                    position = (x,y)
                    frame = cv2.rectangle(frame, position, (x+w, y+h), (0, 255, 0), 5)
                    cv2.putText(frame, textlabel, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
        i+=1

    cv.imshow('Capture - phone detection', frame)

camera_device = devicename
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    detectAndDisplay(frame)

    if cv.waitKey(10) == 27:
        break
