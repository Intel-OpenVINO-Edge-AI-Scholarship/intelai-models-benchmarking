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

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
from tensorflow.python.data.experimental import parallel_interleave
from tensorflow.python.data.experimental import map_and_batch
import time
from keras.models import Sequential, load_model, Model
import cv2
import PIL
from time import sleep

from argparse import ArgumentParser
from inference.coco_detection_evaluator import CocoDetectionEvaluator
from inference.face_label_map import category_map
from inference.metrics_face import ArcFace

IMAGE_SIZE = 400
# dataset is VOC2007 with person_test.txt
COCO_NUM_VAL_IMAGES = 2008

import os

import numpy as np

def bbox_aggregation(bbox_example):
  return bbox_example

def parse_and_preprocess(serialized_example):
  # Dense features in Example proto.
  feature_map = {
      'image/object/class/text': tf.FixedLenFeature([], dtype=tf.string, 
                                          default_value=''),
      'image/source_id': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
  }
  sparse_int64 = tf.FixedLenFeature([], dtype=tf.string, default_value='')
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_int64 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(serialized_example, feature_map, name='features')

  xmin = tf.expand_dims(features['image/object/bbox/xmin'], 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'], 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'], 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'], 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.cast(tf.concat([ymin, xmin, ymax, xmax], 0), dtype=tf.int64)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  # bbox = tf.expand_dims(bbox, 0)
  # bbox = tf.transpose(bbox, [0, 2, 1])

  # entity filename
  label = features['image/object/class/text']
  # part = features['image/object/class/label'].values

  image_id = features['image/source_id']

  return bbox, label, image_id, features

def iter_tensor(iterator):
  return iterator

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
    img = np.zeros((1,len(bbox),160,160,3))
    for ii in range(len(bbox)):
      ymin, xmin, ymax, xmax = bbox[ii][0]
      i = images[ii][ymin:ymax,xmin:xmax]
      i = cv2.resize(i, (160,160))
      img[0,ii] = i
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
        warmup_iter = 0
        ttime = 0.0

        print('total iteration is {0}'.format(str(total_iter)))
        print('warm up iteration is {0}'.format(str(warmup_iter)))

        total_samples = 0
        if self.args.data_location:
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
          for step in range(total_iter):
            box, label, image_source, features = ds_iterator.get_next()
            if features is None:
              break
            box = box.eval(session=sess)
            label = label.eval(session=sess).flatten()
            image_source = image_source.eval(session=sess).flatten()

            image_source = [image_id if type(image_id) == 'str' else image_id.decode('utf-8') for image_id in image_source]
            label = [x if type(x) == 'str' else x.decode('utf-8') for x in label]

            start_time = time.time()

            images = [
              np.asarray(PIL.Image.open(os.path.join(self.args.imagesets_dir, image_id)).convert('RGB')) for image_id in image_source
            ]

            images = self.preprocess_bounding_box_images(images, bbox, image_source)
            total_samples += images.shape[0]

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

          # if len(image_batches) == self.args.batch_size:
          # image_batches = []

        # self.coord.join(self.threads, stop_grace_period_secs=1)

        # if len(image_batches) < self.args.batch_size:
          # arcface_features = self.arcface_model.predict(np.vstack(image_batches), verbose=1)
          # print ('Batchsize: {0}'.format(str(self.args.batch_size)))
          # print ('Time spent per BATCH: {0:10.4f} ms'.format(ttime / total_samples * 1000))
          # print ('Total samples/sec: {0:10.4f} samples/s'.format(total_samples * self.args.batch_size / ttime))
          # print ('Total labeled samples: {0} person, {1} head'.format(
          #   np.where(np.array(label)=="person")[0].shape[0], 
          #   np.where(np.array(parts)=="head")[0].shape[0]))

        # self.coord.request_stop()
        # self.coord.join(self.threads)

  def get_input(self, sess):
    box, label, part, image_id, features = parse_and_preprocess(serialized_example)
    # self.threads = tf.train.start_queue_runners(sess=sess, coord = self.coord)

    return tuple(features.items()), box, label, part, image_id

  def accuracy_check(self):
    print("Inference for accuracy check.")
    self.build_data_sess()
    evaluator = CocoDetectionEvaluator()
    iter = 0
    while True:
      print('Run {0} iter'.format(iter))
      iter += 1
      features, bbox, label, image_id, sess = self.get_input()
      if features is None:
          break
      bbox = bbox.eval(session=sess)
      label = label.eval(session=sess)
      image_id = image_id.eval(session=sess)
      ground_truth = {}
      ground_truth['boxes'] = np.asarray(bbox)
      label = [x if type(x) == 'str' else x.decode('utf-8') for x in label]
      ground_truth['classes'] = np.asarray([x for x in label])
      image_id = image_id[0] if type(image_id[0]) == 'str' else image_id[0].decode('utf-8')
      evaluator.add_single_ground_truth_image_info(image_id, ground_truth)
      num, boxes, scores, labels = sess.run(self.output_tensors, {self.input_tensor: input_images})
      eval_image_augment_scores()
      detection = {}
      num = int(num[0])
      detection['boxes'] = np.asarray(boxes[0])[0:num]
      detection['scores'] = np.asarray(scores[0])[0:num]
      detection['classes'] = np.asarray(labels[0])[0:num]
      evaluator.add_single_detected_image_info(image_id, detection)
      if iter * self.args.batch_size >= COCO_NUM_VAL_IMAGES:
        evaluator.evaluate()
        break

  def run(self):
    if self.args.benchmark_only:
      self.run_benchmark()
    elif self.args.accuracy_only:
      self.accuracy_check()



if __name__ == "__main__":
  infer = model_infer()
  infer.run()

