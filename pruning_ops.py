# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using multiple GPU's with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import pdb
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
# import cifar10

def prune(scale):
	pruning_ops = []
	for x in tf.trainable_variables():
		if ('full_connected_layer' in x.op.name):
			zeros = tf.zeros_like(x)
			ones = tf.ones_like(x)
			sigma = tf.reduce_max(tf.abs(x))*scale
			#Add a mask to constrain the weights being pruned to be zeros.
			try:
				with tf.variable_scope('mask', reuse=True) as scope:
					mask = tf.get_variable(x.op.name,x.shape)
			except:
				with tf.variable_scope('mask', reuse=None) as scope:
					mask = tf.get_variable(x.op.name,x.shape)
			# Prune the weights with probabilistic or deterministic method.
			prun_op = x.assign(tf.where(tf.greater(tf.abs(x), sigma), x, zeros))
			constrain_op = mask.assign(tf.where(tf.greater(tf.abs(x), sigma), ones, zeros))
			
			pruning_ops.append(prun_op)
			pruning_ops.append(constrain_op)
	return tf.group(*pruning_ops)

