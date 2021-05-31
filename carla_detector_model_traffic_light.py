import json
import os
import argparse
import cv2
from yolo import YOLO, dummy_loss
from keras.models import load_model
from preprocessing import load_image_predict
from postprocessing import decode_netout
import numpy as np

from postprocessing import draw_boxes

from yolo import YOLO


BASE_DIR = os.path.dirname(__file__)

classes=["go", "stop"]
anchors=[0.24,0.79, 0.80,2.12]
num_classes=2
<<<<<<< Updated upstream
obj_thresh=0.40
=======
obj_thresh=0.4
>>>>>>> Stashed changes
nms_thresh=0.01
max_obj=5

def get_model_from_file():
    path = os.path.join(BASE_DIR, 'model_traffic_light.h5')
    model = load_model(path, custom_objects={'custom_loss': dummy_loss}, compile=False)
    return model

def predict_with_model_from_image(model, image):

    dummy_array = np.zeros((1, 1, 1, 1, max_obj, 4))
    netout = model.predict([image, dummy_array])[0]

    boxes = decode_netout(netout=netout, anchors=anchors,
                          nb_class=num_classes,
                          obj_threshold=obj_thresh,
                          nms_threshold=nms_thresh)
    return boxes

def detect_image(image_RGB, image_BGR, model):

    netout = predict_with_model_from_image(model, image_RGB)
    plt_image = draw_boxes(image_BGR, netout, classes)

    return plt_image,netout
