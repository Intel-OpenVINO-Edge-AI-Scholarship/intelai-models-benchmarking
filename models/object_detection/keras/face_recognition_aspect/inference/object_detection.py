import cv2 as cv
import argparse
import numpy as np
import sys
import time
from threading import Thread
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue
import os
import pickle

from common import *
from tf_text_graph_common import readTextMessage
from tf_text_graph_ssd import createSSDGraph
from tf_text_graph_faster_rcnn import createFasterRCNNGraph

backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV)
targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD)

args = None

class ParamArgs():
    def __init__(self, param_dict):
        update_param_dict(param_dict, self)

def update_param_dict(param_dict, obj):
  for key, value in param_dict.items():
    obj.__setattr__(key, value)

def obtain_args(param_dict):
    return ParamArgs(param_dict)

outNames, config, classes, winName = None, None, None, None

confThreshold = 0.6
nmsThreshold = 0
is_async = False
futureOutputs = []

def main(param_args, frame):
    args = ParamArgs(param_args)
    nmsThreshold = args.nms
    global outNames, config, classes, winName
    # If config specified, try to load it as TensorFlow Object Detection API's pipeline.
    config = readTextMessage(args.config)
    if 'model' in config:
        print('TensorFlow Object Detection API config detected', file=sys.stderr)
        if 'ssd' in config['model'][0]:
            print('Preparing text graph representation for SSD model: ' + args.out_tf_graph, file=sys.stderr)
            createSSDGraph(args.model, args.config, args.out_tf_graph)
            args.config = args.out_tf_graph
        elif 'faster_rcnn' in config['model'][0]:
            print('Preparing text graph representation for Faster-RCNN model: ' + args.out_tf_graph, file=sys.stderr)
            createFasterRCNNGraph(args.model, args.config, args.out_tf_graph)
            args.config = args.out_tf_graph

    # Load names of classes
    classes = None
    if args.classes:
        with open(args.classes, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

    # Load a network
    net = cv.dnn.readNet(args.model, args.config, args.framework)
    net.setPreferableBackend(args.backend)
    net.setPreferableTarget(args.target)
    outNames = net.getUnconnectedOutLayersNames()

    if not frame is None:
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
    
    # Create a 4D blob from a frame.
    inpWidth = frameWidth
    inpHeight = frameHeight
    blob = cv.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=args.rgb, ddepth=cv.CV_8U)

    # Run a model
    net.setInput(blob, scalefactor=args.scale)
    if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
        frame = cv.resize(frame, (inpWidth, inpHeight))
        net.setInput(np.array([[inpHeight, inpWidth, 1.6]], dtype=np.float32), 'im_info')

    if is_async:
        futureOutputs.append(net.forwardAsync())
    else:
        outs = net.forward(outNames)
        boxes, confidences, classIds = postprocess(net, frame, outs)

    return boxes, confidences, classIds

classIds = []
confidences = []
boxes = []

def postprocess(net, frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    global args
    global outNames, config, classes, winName

    global classIds, confidences, boxes

    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)

    if lastLayer.type == 'DetectionOutput':
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        for out in outs:
            for detection in out[0, 0]:
                confidence = detection[2]
                if confidence > confThreshold:
                    left = int(detection[3])
                    top = int(detection[4])
                    right = int(detection[5])
                    bottom = int(detection[6])
                    width = right - left + 1
                    height = bottom - top + 1
                    if width <= 2 or height <= 2:
                        left = int(detection[3] * frameWidth)
                        top = int(detection[4] * frameHeight)
                        right = int(detection[5] * frameWidth)
                        bottom = int(detection[6] * frameHeight)
                        width = right - left + 1
                        height = bottom - top + 1
                    classIds.append(int(detection[1]) - 1)  # Skip background label
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    elif lastLayer.type == 'Region':
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type, file=sys.stderr)
        exit()

    return boxes, confidences, classIds
