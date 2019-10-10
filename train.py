#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @ 2019 Liming Liu     HuNan University
#


"""Train the model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import logging
import os
import os.path as osp
import random
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorflow.python import pywrap_tensorflow

import configuration
from utils.misc_utils import auto_select_gpu, mkdir_p, save_cfgs

from models.bisenet import BiseNet

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--dataset", type=str, required=True,
                    help="Path to the parent directory of the datset.")
parser.add_argument("--random-scale", action="store_true", default=False)
parser.add_argument("--random-mirror", action="store_true", default=False)
parser.add_argument("--width", type=int, default=800)
parser.add_argument("--height", type=int, default=600)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--epoch-size", type=int, default=2000)
args = parser.parse_args()

class_dict_file_path = osp.join(args.dataset, "class_dict.csv")
with open(class_dict_file_path) as f:
  class_dict_lines = f.read().split("\n")
  class_dict_lines = [l.strip() for l in class_dict_lines if l.strip() != ""]

configuration.TRAIN_CONFIG["DataSet"] = args.dataset
configuration.TRAIN_CONFIG["class_dict"] = osp.join(args.dataset, "class_dict.csv")
configuration.TRAIN_CONFIG["train_data_config"]["input_dir"] = osp.join(args.dataset, "train")
configuration.TRAIN_CONFIG["train_data_config"]["output_dir"] = osp.join(args.dataset, "train_labels")
configuration.TRAIN_CONFIG["train_data_config"]["random_scale"] = args.random_scale
configuration.TRAIN_CONFIG["train_data_config"]["random_mirror"] = args.random_mirror
configuration.TRAIN_CONFIG["train_data_config"]["num_examples_per_epoch"] = args.epoch_size
configuration.TRAIN_CONFIG["train_data_config"]["epoch"] = args.num_epochs
configuration.TRAIN_CONFIG["train_data_config"]["batch_size"] = args.batch_size


ex = Experiment(configuration.RUN_NAME)
ex.observers.append(FileStorageObserver.create(osp.join(configuration.LOG_DIR, 'sacred')))

num_classes = len(class_dict_lines) - 1


def _configure_learning_rate(train_config, global_step):
  lr_config = train_config['lr_config']

  num_batches_per_epoch = \
    int(train_config['train_data_config']['num_examples_per_epoch'] / train_config['train_data_config']['batch_size'])

  lr_policy = lr_config['policy']
  if lr_policy == 'piecewise_constant':
    lr_boundaries = [int(e * num_batches_per_epoch) for e in lr_config['lr_boundaries']]
    return tf.train.piecewise_constant(global_step,
                                       lr_boundaries,
                                       lr_config['lr_values'])
  elif lr_policy == 'exponential':
    decay_steps = int(num_batches_per_epoch) * lr_config['num_epochs_per_decay']
    return tf.train.exponential_decay(lr_config['initial_lr'],
                                      global_step,
                                      decay_steps=decay_steps,
                                      decay_rate=lr_config['lr_decay_factor'],
                                      staircase=lr_config['staircase'])
  elif lr_policy == 'polynomial':
    T_total = (int(num_batches_per_epoch)+1) * train_config['train_data_config']['epoch']
    return lr_config['initial_lr'] * (1 - tf.to_float(global_step)/T_total)**lr_config['power']
  elif lr_policy == 'cosine':
    T_total = train_config['train_data_config']['epoch'] * num_batches_per_epoch
    return 0.5 * lr_config['initial_lr'] * (1 + tf.cos(np.pi * tf.to_float(global_step) / T_total))
  else:
    raise ValueError('Learning rate policy [%s] was not recognized', lr_policy)


def _configure_optimizer(train_config, learning_rate):
  optimizer_config = train_config['optimizer_config']
  optimizer_name = optimizer_config['optimizer'].upper()
  if optimizer_name == 'MOMENTUM':
    optimizer = tf.train.MomentumOptimizer(
      learning_rate,
      momentum=optimizer_config['momentum'],
      use_nesterov=optimizer_config['use_nesterov'],
      name='Momentum')
  elif optimizer_name == 'SGD':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer_name == 'RMSProp':
    optimizer = tf.train.RMSPropOptimizer(learning_rate, optimizer_config['decay'], optimizer_config['momentum'])
  else:
    raise ValueError('Optimizer [%s] was not recognized', optimizer_config['optimizer'])
  return optimizer


def get_tensors_in_checkpoint_file(file_name, all_tensors=True, tensor_name=None):
    varlist = []
    var_value = []
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            varlist.append(key)
            var_value.append(reader.get_tensor(key))
    else:
        varlist.append(tensor_name)
        var_value.append(reader.get_tensor(tensor_name))
    return (varlist, var_value)


def match_loaded_and_memory_tensors(loaded_tensors):
    full_var_list = list()
    # Loop all loaded tensors
    for i, (tensor_name, tensor_loaded) in enumerate(zip(loaded_tensors[0], loaded_tensors[1])):
        try:
            # Extract tensor
            tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name + ":0")
            if not np.array_equal(tensor_aux.shape, tensor_loaded.shape) \
                    and not np.array_equal(tensor_aux.shape, tensor_loaded.shape[::-1]):
                print('Weight mismatch for tensor {}: RAM model: {}, Loaded model: {}'.format(tensor_name, tensor_aux.shape,
                                                                                              tensor_loaded.shape))
            else:
                full_var_list.append(tensor_aux)
        except:
            print('Loaded a tensor from weights file which has not been found in model: ' + tensor_name)
    return full_var_list


def main(model_config, train_config):
  os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()

  # Create training directory which will be used to save: configurations, model files, TensorBoard logs
  train_dir = train_config['train_dir']
  if not osp.isdir(train_dir):
    logging.info('Creating training directory: %s', train_dir)
    mkdir_p(train_dir)

  g = tf.Graph()
  with g.as_default():
    # Set fixed seed for reproducible experiments
    random.seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    tf.set_random_seed(train_config['seed'])

    # Build the training and validation model
    model = BiseNet(model_config, train_config, num_classes, mode="train")
    model.build()
    model_va = BiseNet(model_config, train_config, num_classes, mode="validation")
    model_va.build(reuse=True)

    # Save configurations for future reference
    save_cfgs(train_dir, model_config, train_config)

    learning_rate = _configure_learning_rate(train_config, model.global_step)
    optimizer = _configure_optimizer(train_config, learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    # Set up the training ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.contrib.layers.optimize_loss(loss=model.total_loss,
                                                   global_step=model.global_step,
                                                   learning_rate=learning_rate,
                                                   optimizer=optimizer,
                                                   clip_gradients=train_config['clip_gradients'],
                                                   learning_rate_decay_fn=None,
                                                   summaries=['learning_rate'])


    summary_writer = tf.summary.FileWriter(train_dir, g)
    summary_op = tf.summary.merge_all()

    global_variables_init_op = tf.global_variables_initializer()
    local_variables_init_op = tf.local_variables_initializer()

    # Dynamically allocate GPU memory
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)

    sess = tf.Session(config=sess_config)
    model_path = tf.train.latest_checkpoint(train_config['train_dir'])

    if not model_path:
      sess.run(global_variables_init_op)
      sess.run(local_variables_init_op)
      start_step = 0

      if model_config['frontend_config']['pretrained_dir'] and model.init_fn:
        model.init_fn(sess)
    else:
      logging.info('Restore from last checkpoint: {}'.format(model_path))
      vars_in_checkpoint = get_tensors_in_checkpoint_file(file_name=model_path)
      loadable_tensors = match_loaded_and_memory_tensors(vars_in_checkpoint)
      # print("\n".join([t.name for t in loadable_tensors]))
      loader = tf.train.Saver(var_list=loadable_tensors)
      sess.run(global_variables_init_op)
      sess.run(local_variables_init_op)
      loader.restore(sess, model_path)
      start_step = tf.train.global_step(sess, model.global_step.name) + 1

    saver = tf.train.Saver(tf.global_variables(),
                           max_to_keep=train_config['max_checkpoints_to_keep'])

    g.finalize()  # Finalize graph to avoid adding ops by mistake

    # Training loop
    data_config = train_config['train_data_config']
    total_steps = int(data_config['epoch'] *
                      data_config['num_examples_per_epoch'] /
                      data_config['batch_size'])
    logging.info('Step: {}/{}'.format(start_step, total_steps))
    for step in range(start_step, total_steps):
      start_time = time.time()
      _, predict_loss, loss = sess.run([train_op, model.loss, model.total_loss])
      duration = time.time() - start_time

      if step % 10 == 0:
        examples_per_sec = data_config['batch_size'] / float(duration)
        time_remain = data_config['batch_size'] * (total_steps - step) / examples_per_sec
        m, s = divmod(time_remain, 60)
        h, m = divmod(m, 60)
        format_str = ('%s: step %d, total loss = %.2f, predict loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch; %dh:%02dm:%02ds remains)')
        logging.info(format_str % (datetime.now(), step, loss, predict_loss,
                                   examples_per_sec, duration, h, m, s))

      if step % 10 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      if step % train_config['save_model_every_n_step'] == 0 or (step + 1) == total_steps:
        checkpoint_path = osp.join(train_config['train_dir'], 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
