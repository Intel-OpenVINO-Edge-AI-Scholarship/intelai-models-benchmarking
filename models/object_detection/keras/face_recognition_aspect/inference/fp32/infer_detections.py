#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

from __future__ import division

import tensorflow as tf
# tf.enable_eager_execution()
import time
from keras.models import Sequential, load_model, Model
from tensorflow.python.data.experimental import parallel_interleave
from tensorflow.python.data.experimental import map_and_batch
import cv2
import cv2 as cv
import PIL
import subprocess
import logging
import pickle
import signal
import subprocess
import random
from pprint import PrettyPrinter

from argparse import ArgumentParser
from inference.face_label_map import category_map
from inference.metrics_face import ArcFace
from inference import object_detection
from augment.augment import rotate
from inference.score import face_recognize_risk, score_model

IMAGE_SIZE = 400
# dataset is VOC2007 with person_test.txt
COCO_NUM_VAL_IMAGES = 2008

import os
os.environ['GLOG_minloglevel'] = '1'
import threading
import shutil

import numpy as np

backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV)
targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD)

def bbox_aggregation(bbox_example):
  return bbox_example

def parse_and_preprocess(serialized_example):
  # Dense features in Example proto.
  feature_map = {
      'image/object/class/text': tf.compat.v1.FixedLenFeature([], 
        dtype=tf.string, default_value=''),
      'image/source_id': tf.compat.v1.FixedLenFeature([], dtype=tf.string, default_value='')
  }
  sparse_int64 = tf.compat.v1.VarLenFeature(dtype=tf.int64)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_int64 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.compat.v1.parse_single_example(serialized_example, feature_map, name='features')

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.cast(tf.concat([ymin, xmin, ymax, xmax], 0), dtype=tf.int64)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  # entity filename
  label = features['image/object/class/text']
  # part = features['image/object/class/label'].values

  image_id = features['image/source_id']

  return bbox[0], label, image_id, features

def exec_evaluator(ground_truth_dicts, detect_dicts, image_id_gt_dict,
total_iter, batch_size):
  import sys
  
  if "numpy" in sys.modules:
    del sys.modules["numpy"]
  if "np" in sys.modules:
    del sys.modules["np"]
  import numpy as np
  
  def setUp():
    np.round = round
  
  from inference.coco_detection_evaluator import CocoDetectionEvaluator

  setUp()
  
  evaluator = CocoDetectionEvaluator()
  for step in range(total_iter):
    # add single ground truth info detected
    # add single image info detected
    if step in list(detect_dicts.keys()):
      try:
        evaluator.add_single_ground_truth_image_info(image_id_gt_dict[step], ground_truth_dicts[step])
        evaluator.add_single_detected_image_info(image_id_gt_dict[step], detect_dicts[step])
      except Exception as e:
        raise e

  if (step + 1) * batch_size >= COCO_NUM_VAL_IMAGES:
    metrics, metrics1 = evaluator.evaluate()
  
  if metrics:
    pp = PrettyPrinter(indent=4)
    pp.pprint(metrics)
    pp.pprint(metrics1)

class model_infer:

  def __init__(self):
    arg_parser = ArgumentParser(description='Parse args')

    arg_parser.add_argument('-b', "--batch-size",
                            help="Specify the batch size. If this " \
                                 "parameter is not specified or is -1, the " \
                                 "largest ideal batch size for the model will " \
                                 "be used.",
                            dest="batch_size", type=int, default=1, required=False)

    arg_parser.add_argument('-e', "--inter-op-parallelism-threads",
                            help='The number of inter-thread.',
                            dest='num_inter_threads', type=int, default=10)

    arg_parser.add_argument('-a', "--intra-op-parallelism-threads",
                            help='The number of intra-thread.',
                            dest='num_intra_threads', type=int, default=10)

    arg_parser.add_argument('-g', "--input-graph",
                            help='Specify the input graph.',
                            dest='input_graph', required=False)

    arg_parser.add_argument('-met', "--method",
                            help='Specify the method.',
                            dest='method', default='pnorm', required=False)
    
    arg_parser.add_argument('--annotations_dir', "--annotations_dir", help="Annotations dir", dest='imagesets_dir', 
                            required=False)

    arg_parser.add_argument('-d', "--data-location",
                            help='Specify the location of the data. '
                                 'If this parameter is not specified, '
                                 'the benchmark will use random/dummy data.',
                            dest="data_location", default=None, required=True)

    arg_parser.add_argument('-rd', "--risk-difference",
                            help='The risk difference measure.',
                            dest='risk_difference', default=0.5, required=True)

    arg_parser.add_argument('-r', "--accuracy-only",
                            help='For accuracy measurement only.',
                            dest='accuracy_only', type=bool, default=False, required=False)

    arg_parser.add_argument('-bo', "--benchmark-only",
                            help='For accuracy measurement only.',
                            dest='benchmark_only', type=bool, default=False, required=False)

    arg_parser.add_argument('-i', "--iter",
                            help='For accuracy measurement only.',
                            dest='total_iter', default=500, type=int)

    arg_parser.add_argument('-w', "--warmup_iter",
                            help='For accuracy measurement only.',
                            dest='warmup_iter', default=0, type=int)

    # parse the arguments
    self.args = arg_parser.parse_args()

    self.param_dict = {
      "framework": "tensorflow",
      "thr": 0.4,
      "model": "/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/nd131-openvino-people-counter-newui/models/opencv_dnn/opencv_face_detector_uint8.pb",
      "backend": 3,
      "async": self.args.batch_size,
      "target": 0,
      "classes": "",
      "nms": 0.65,
      "motion_tracker": False,
      "inputs": "",
      "input": None,
      "scale": 1.0,
      "mean": 1.0,
      "rgb": False,
      "config": "/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/nd131-openvino-people-counter-newui/models/opencv_dnn/opencv_face_detector.pbtxt",
    }

    self.config_dict = dict()
    self.config_dict['ARCFACE_PREBATCHNORM_LAYER_INDEX']=-3
    self.config_dict['ARCFACE_POOLING_LAYER_INDEX']=-4

    self.config = tf.ConfigProto()
    self.config.intra_op_parallelism_threads = self.args.num_intra_threads
    self.config.inter_op_parallelism_threads = self.args.num_inter_threads
    self.config.use_per_session_threads = 1

    self.load_graph()

    if self.args.batch_size == -1:
      self.args.batch_size = 1
    self.args.batch_size = 1

  # pnorm for color images
  def preprocess_bounding_box_images(self, images, bbox, image_source):
    img = np.zeros((len(bbox),160,160,3))
    for ii,box in enumerate(bbox):
      ymin, xmin, ymax, xmax = box
      i = images[ymin:ymax,xmin:xmax]
      i = cv2.resize(i, (160,160))
      img[ii] = i
    return img

  def filter_conventional_box_images(self, bbox):
    boxes = []
    for ii,box in enumerate(bbox):
      left, top, width, height = box
      if width > 70 and height > 70:
        boxes.append(box)
    return boxes

  def preprocess_conventional_box_images(self, images, bbox, image_source):
    img = np.zeros((len(bbox),160,160,3))
    for ii,box in enumerate(bbox):
      left, top, width, height = box
      i = images[top:top+height,left:left+width]
      i = cv2.resize(i, (160,160))
      img[ii] = i
    return img
    
  def build_data_sess(self):
    arcface_model = load_model(self.args.input_graph, custom_objects={'ArcFace': ArcFace})
    if self.args.method == "pnorm":
      self.arcface_model = Model(inputs=arcface_model.input[0], 
      outputs=arcface_model.layers[self.config_dict['ARCFACE_POOLING_LAYER_INDEX']].output)
    elif self.args.method == "lognorm":
      self.arcface_model = Model(inputs=arcface_model.input[0], 
      outputs=arcface_model.layers[self.config_dict['ARCFACE_PREBATCHNORM_LAYER_INDEX']].output)

  def load_graph(self):
    print("graph has been loaded using keras..")

  def run_benchmark(self):
    if self.args.data_location:
      print("Inference with real data.")
    else:
      print("Inference with dummy data.")

    global model, graph
    with tf.Session().as_default() as sess:
      with sess.graph.as_default() as graph:
        if self.args.data_location:
          self.build_data_sess()
        else:
          raise Exception("no data location provided")

        total_iter = 1000
        # warmup_iter = self.args.warmup_iter
        warmup_iter = 0
        ttime = 0.0

        print('total iteration is {0}'.format(str(total_iter)))
        print('warm up iteration is {0}'.format(str(warmup_iter)))

        total_samples = 0
        self.coord = tf.train.Coordinator()
        tfrecord_paths = [self.args.data_location]
        self.filename_queue = tf.train.string_input_producer(
          tfrecord_paths, capacity=self.args.batch_size, name='created_poert')
        # _, serialized_example = self.reader.read(self.filename_queue)
        state = None
        if self.args.data_location:
          image_batches = []
          for step in range(total_iter):
            self.reader = tf.TFRecordReader()
            if state is not None:
              self.reader.restore_state(state)
            _, serialized_example = self.reader.read(self.filename_queue)
            self.threads = tf.train.start_queue_runners(sess=sess, coord = self.coord)
            box, label, image_id, features = parse_and_preprocess(serialized_example)
            features, bbox, label, parts, image_id = \
            tuple(features.items()), box, label, part, image_id
            if features is None:
              break
            bbox = bbox.eval(session=sess)
            label = label.eval(session=sess)
            image_id = image_id.eval(session=sess)

            image_id = image_id if type(image_id) == 'str' else image_id.decode('utf-8')
            # parts = parts[0].split("//") if type(parts[0]) == 'str' else parts[0].decode('utf-8').split("//")
            label = label if type(label) == 'str' else label.decode('utf-8')

            start_time = time.time()

            images = np.asarray(PIL.Image.open(os.path.join(self.args.imagesets_dir, image_id)).convert('RGB'))

            images = self.preprocess_bounding_box_images(images, bbox, image_id)
            total_samples += images.shape[0]

            # image_batches.append(images)

            # if len(image_batches) == self.args.batch_size:
            arcface_features = self.arcface_model.predict(np.vstack(images), verbose=1)
            
            end_time = time.time()

            duration = end_time - start_time
            if (step + 1) % 10 == 0:
              print('steps = {0}, {1} sec'.format(str(step), str(duration)))
            
            if step + 1 > warmup_iter:
              ttime += duration
              
            # if len(image_batches) == self.args.batch_size:
            print ('Batchsize: {0}'.format(str(self.args.batch_size)))
            print ('Time spent per BATCH: {0:10.4f} ms'.format(ttime / total_samples * 1000))
            print ('Total samples/sec: {0:10.4f} samples/s'.format(total_samples * self.args.batch_size / ttime))
            print ('Total labeled samples: {0} person'.format(
              np.where(np.array(label)=="person")[0].shape[0]))

            state = self.reader.serialize_state()

        self.coord.request_stop()
        self.coord.join(self.threads)
  
  def face_detector(self, images, image_ids, batch_size):
    return object_detection.main(self.param_dict, images)

  def augment_images(self, images, greedy=0.69, angle=8, scale=1.0):
    # flip image
    result = np.zeros((len(images)*2,160,160,3))
    for ii in range(0,len(images),1):
      result[2*ii] = images[ii].copy()
      if greedy > random.uniform(0.0, 1.0):
        result[2*ii+1] = rotate(images[ii], angle=angle, scale=scale)
      else:
        result[2*ii+1] = cv2.flip(images[ii], 1)
    return result

  def measure(self, features, risk_difference=0.05, significant=1, to_significant=5):
    measure_scores = []
    for ii in range(0,len(features),2):
      risk_vector1 = score_model(features[ii], significant, to_significant)
      risk_vector2 = score_model(features[ii+1], significant, to_significant)
      measure_scores.append(
        face_recognize_risk(risk_difference, risk_vector1, risk_vector2)
      )
    return measure_scores
  
  def accuracy_check(self):
    print("Inference for accuracy check.")
    total_iter = COCO_NUM_VAL_IMAGES
    fm = category_map
    fm = dict(zip(list(fm.values()),list(fm.keys())))
    print('total iteration is {0}'.format(str(total_iter)))
    global model, graph
    with tf.Session().as_default() as sess:
      if self.args.data_location:
        self.build_data_sess()
      else:
        raise Exception("no data location provided")
      total_samples = 0
      self.coord = tf.train.Coordinator()
      tfrecord_paths = [self.args.data_location]
      ds = tf.data.TFRecordDataset.list_files(tfrecord_paths)
      ds = ds.apply(
        parallel_interleave(
          tf.data.TFRecordDataset, cycle_length=1, block_length=1,
          buffer_output_elements=10000, prefetch_input_elements=10000))
      ds = ds.prefetch(buffer_size=10000)
      ds = ds.apply(
          map_and_batch(
            map_func=parse_and_preprocess,
            batch_size=self.args.batch_size,
            num_parallel_batches=1,
            num_parallel_calls=None))
      ds = ds.prefetch(buffer_size=10000)
      ds_iterator = tf.data.make_one_shot_iterator(ds)
      state = None
      warmup_iter = 0
      
      self.ground_truth_dicts = {}
      self.detect_dicts = {}

      self.total_iter = total_iter
      self.image_id_gt_dict = {}

      if self.args.data_location:
        image_batches = []
        for step in range(total_iter):
          bbox, label, image_id, features = ds_iterator.get_next()
          features, bbox, label, image_id = \
          tuple(features.items()), bbox, label, image_id
          if features is None:
            break
          bbox, label, image_id = sess.run([bbox, label, image_id])

          # ground truth of bounding boxes from pascal voc
          ground_truth = {}
          ground_truth['boxes'] = np.asarray(bbox[0])
          label_gt = [fm[l] if type(l) == 'str' else fm[l.decode('utf-8')] for l in label]
          image_id_gt = [i if type(i) == 'str' else i.decode('utf-8') for i in image_id]
          ground_truth['classes'] = np.array(label_gt*len(ground_truth['boxes']))
          # saving all ground truth dictionaries
          self.ground_truth_dicts[step] = ground_truth
          self.image_id_gt_dict[step] = image_id_gt[0]

          images = np.asarray(PIL.Image.open(os.path.join(self.args.imagesets_dir, image_id_gt[0])).convert('RGB'))
          # face detection
          boxes, confidences, classIds = self.face_detector(images, image_id_gt, self.args.batch_size)

          boxes = self.filter_conventional_box_images(boxes)
          try:
            if len(boxes) > 0:
              # ground truth bounding box
              images_new = self.preprocess_bounding_box_images(images, bbox[0], image_id_gt)
              total_samples += images_new.shape[0]
              images_new = self.augment_images(images_new)

              # detection for bounding boxes from pascal voc
              detect = {}
              label_det = label_gt
              image_id_det = image_id_gt
              # detected conventional bounding box
              images_sync = self.preprocess_conventional_box_images(images, boxes, image_id_det)
              images_sync = self.augment_images(images_sync)

              detect['boxes'] = np.asarray(boxes)
              detect['classes'] = np.asarray(label_det*len(detect['boxes']))
              features = self.arcface_model.predict(images_sync, verbose=1)
              measures = self.measure(features)
              if measures[0][0]:
                detect['scores'] = np.broadcast_to(np.mean(np.asarray(measures[0][1])),len(detect['boxes']))
              elif np.mean(measures[0][1]) > 0:
                detect['scores'] = np.broadcast_to(np.mean(np.asarray(measures[0][1])),len(detect['boxes']))
              else:
                detect['scores'] = np.broadcast_to(np.asarray([0]),len(detect['boxes']))
              
              self.detect_dicts[step] = detect
          except Exception as e:
            print(e.args)

          if (step + 1) % 10 == 0:
            print('steps = {0} step'.format(str(step)))

  def run(self):
    if self.args.accuracy_only:
      self.accuracy_check()
      exec_evaluator(self.ground_truth_dicts, 
      self.detect_dicts, self.image_id_gt_dict, 
      self.total_iter, self.args.batch_size)
    elif self.args.benchmark_only:
      self.run_benchmark()

if __name__ == "__main__":
  infer = model_infer()
  infer.run()

