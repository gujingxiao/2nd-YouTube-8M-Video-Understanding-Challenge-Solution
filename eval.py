# -×- encoding: utf-8 -*-
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary for evaluating Tensorflow models on the YouTube-8M dataset."""

import glob
import json
import os
import time

import eval_util
import losses
import frame_level_models
import video_level_models
import readers
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import utils
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/home/gujingxiao/projects/yt8m/models/video_level",
                      "存放训练模型的路径，Tensorboard metrics文件也存在这里 ")
  flags.DEFINE_string( "eval_data_pattern", "/home/gujingxiao/projects/yt8m/video_level/validate/validate*.tfrecord",
                      "存放验证集的路径.")
  # Other flags.
  flags.DEFINE_integer("batch_size", 2048, "每个batch计算多少个examples.")
  flags.DEFINE_integer("num_readers", 4, "用多少个线程来读取数据.")
  flags.DEFINE_boolean("run_once", True, "是否只验证一次.")
  flags.DEFINE_integer("top_k", 20, "每个视频做出前多少个预测.官方成绩使用前20")
  flags.DEFINE_float("gpu_ratio", "0.26", "GPU使用率")


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def get_input_evaluation_tensors(reader, data_pattern, batch_size=1024, num_readers=1):
  """Creates the section of the graph which reads the evaluation data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns: tuple including features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video.
  """
  logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
  with tf.name_scope("eval_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)
    eval_data = [reader.prepare_reader(filename_queue) for _ in range(num_readers)]

    return tf.train.batch_join(eval_data,batch_size=batch_size,capacity=5 * batch_size,
                        allow_smaller_final_batch=True,enqueue_many=True)


def build_graph(reader, model, eval_data_pattern, label_loss_fn, batch_size=1024, num_readers=1):
  """Creates the Tensorflow graph for evaluation.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit from BaseModel.
    eval_data_pattern: glob path to the evaluation data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit from BaseLoss.
    batch_size: How many examples to process at a time.
    num_readers: How many threads to use for I/O operations.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")
  video_id_batch, model_input_raw, labels_batch, num_frames = get_input_evaluation_tensors(  # pylint: disable=g-line-too-long
      reader,
      eval_data_pattern,
      batch_size=batch_size,
      num_readers=num_readers)
  tf.summary.histogram("model_input_raw", model_input_raw)

  feature_dim = len(model_input_raw.get_shape()) - 1

  # Normalize input features.
  model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

  with tf.variable_scope("tower"):
    result = model.create_model(model_input,
                                num_frames=num_frames,
                                vocab_size=reader.num_classes,
                                labels=labels_batch,
                                is_training=False)
    predictions = result["predictions"]
    #support_predictions = result["support_predictions"]
    tf.summary.histogram("model_activations", predictions)
    print(result.keys())
    if "loss" in result.keys():
      label_loss = result["loss"]
    else:
        if "support_predictions2" in result.keys():
            print('Use MultiTaskChainCrossEntropyLoss Function!!!')
            support_predictions1 = result["support_predictions"]
            support_predictions2 = result["predictions2"]
            support_predictions3 = result["support_predictions2"]
            label_loss = label_loss_fn.calculate_loss(predictions, support_predictions1, support_predictions2, support_predictions3, labels_batch)
            tf.add_to_collection("support_predictions", support_predictions1)
            tf.add_to_collection("predictions2", support_predictions2)
            tf.add_to_collection("support_predictions2", support_predictions3)

        elif "support_predictions" in result.keys():
            print('Use MultiTaskCrossEntropyLoss Function!!!')
            support_predictions = result["support_predictions"]
            label_loss = label_loss_fn.calculate_loss(predictions, support_predictions, labels_batch)
            tf.add_to_collection("support_predictions", support_predictions1)

        else:
            print("Use CrossEntropyLoss Function!!!")
            label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("predictions", predictions)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("input_batch_raw", model_input_raw)
  tf.add_to_collection("video_id_batch", video_id_batch)
  tf.add_to_collection("num_frames", num_frames)
  tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
  tf.add_to_collection("summary_op", tf.summary.merge_all())


def get_latest_checkpoint():
  index_files = glob.glob(os.path.join(FLAGS.train_dir, 'model.ckpt-*.index'))

  # No files
  if not index_files:
    return None


  # Index file path with the maximum step size.
  latest_index_file = sorted(
      [(int(os.path.basename(f).split("-")[-1].split(".")[0]), f)
       for f in index_files])[-1][1]

  # Chop off .index suffix and return
  return latest_index_file[:-6]


def evaluation_loop(video_id_batch, prediction_batch, label_batch, loss,
                    summary_op, saver, summary_writer, evl_metrics,
                    last_global_step_val):
  """Run the evaluation loop once.

  Args:
    video_id_batch: a tensor of video ids mini-batch.
    prediction_batch: a tensor of predictions mini-batch.
    label_batch: a tensor of label_batch mini-batch.
    loss: a tensor of loss for the examples in the mini-batch.
    summary_op: a tensor which runs the tensorboard summary operations.
    saver: a tensorflow saver to restore the model.
    summary_writer: a tensorflow summary_writer
    evl_metrics: an EvaluationMetrics object.
    last_global_step_val: the global step used in the previous evaluation.

  Returns:
    The global_step used in the latest model.
  """

  global_step_val = -1
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_ratio)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    latest_checkpoint = get_latest_checkpoint()
    if latest_checkpoint:
      logging.info("Loading checkpoint for eval: " + latest_checkpoint)
      # Restores from checkpoint
      saver.restore(sess, latest_checkpoint)
      # Assuming model_checkpoint_path looks something like:
      # /my-favorite-path/yt8m_train/model.ckpt-0, extract global_step from it.
      global_step_val = os.path.basename(latest_checkpoint).split("-")[-1]

      # Save model
      saver.save(sess, os.path.join(FLAGS.train_dir, "inference_model"))
    else:
      logging.info("No checkpoint file found.")
      return global_step_val

    if global_step_val == last_global_step_val:
      logging.info("skip this checkpoint global_step_val=%s "
                   "(same as the previous one).", global_step_val)
      return global_step_val

    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    fetches = [video_id_batch, prediction_batch, label_batch, loss, summary_op]
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))
      logging.info("enter eval_once loop global_step_val = %s. ",
                   global_step_val)

      evl_metrics.clear()

      examples_processed = 0
      while not coord.should_stop():
        batch_start_time = time.time()
        _, predictions_val, labels_val, loss_val, summary_val = sess.run(
            fetches)
        seconds_per_batch = time.time() - batch_start_time
        example_per_second = labels_val.shape[0] / seconds_per_batch
        examples_processed += labels_val.shape[0]

        iteration_info_dict = evl_metrics.accumulate(predictions_val,
                                                     labels_val, loss_val)
        iteration_info_dict["examples_per_second"] = example_per_second

        iterinfo = utils.AddGlobalStepSummary(
            summary_writer,
            global_step_val,
            iteration_info_dict,
            summary_scope="Eval")
        logging.info("examples_processed: %d | %s", examples_processed,
                     iterinfo)

    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")
      # calculate the metrics for the entire epoch
      epoch_info_dict = evl_metrics.get()
      epoch_info_dict["epoch_id"] = global_step_val

      summary_writer.add_summary(summary_val, global_step_val)
      epochinfo = utils.AddEpochSummary(
          summary_writer,
          global_step_val,
          epoch_info_dict,
          summary_scope="Eval")
      logging.info(epochinfo)
      evl_metrics.clear()
    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: " + str(e))
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    return global_step_val


def evaluate():
  tf.set_random_seed(0)  # for reproducibility

  # Write json of flags
  model_flags_path = os.path.join(FLAGS.train_dir, "model_flags.json")
  if not os.path.exists(model_flags_path):
    raise IOError(("Cannot find file %s. Did you run train.py on the same "
                   "--train_dir?") % model_flags_path)
  flags_dict = json.loads(open(model_flags_path).read())

  with tf.Graph().as_default():
    # convert feature_names and feature_sizes to lists of values
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        flags_dict["feature_names"], flags_dict["feature_sizes"])

    if flags_dict["frame_features"]:
      reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                              feature_sizes=feature_sizes)
    else:
      reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                   feature_sizes=feature_sizes)

    model = find_class_by_name(flags_dict["model"],
        [frame_level_models, video_level_models])()
    label_loss_fn = find_class_by_name(flags_dict["label_loss"], [losses])()

    if FLAGS.eval_data_pattern is "":
      raise IOError("'eval_data_pattern' was not specified. " +
                     "Nothing to evaluate.")

    build_graph(
        reader=reader,
        model=model,
        eval_data_pattern=FLAGS.eval_data_pattern,
        label_loss_fn=label_loss_fn,
        num_readers=FLAGS.num_readers,
        batch_size=FLAGS.batch_size)
    logging.info("built evaluation graph")
    video_id_batch = tf.get_collection("video_id_batch")[0]
    if len(tf.get_collection('support_predictions2')) > 0:
      prediction_batch = tf.get_collection("predictions")[0] * FLAGS.support_loss_1 + tf.get_collection("support_predictions")[0] * FLAGS.support_loss_2 + tf.get_collection("predictions2")[0] * FLAGS.support_loss_3 + tf.get_collection("support_predictions2")[0] * FLAGS.support_loss_4
    elif len(tf.get_collection('support_predictions')) > 0:
      prediction_batch = tf.get_collection("predictions")[0] * (1.0 - FLAGS.support_loss_percent) + tf.get_collection("support_predictions")[0] * FLAGS.support_loss_percent
    elif len(tf.get_collection('predictions')) > 0:  
      prediction_batch = tf.get_collection("predictions")[0]
    else:
      raise IOError("'GLOBAL_FLAGS is invalid. Please check if the loss functions is correct.")

    label_batch = tf.get_collection("labels")[0]
    loss = tf.get_collection("loss")[0]
    summary_op = tf.get_collection("summary_op")[0]

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir, graph=tf.get_default_graph())

    evl_metrics = eval_util.EvaluationMetrics(reader.num_classes, FLAGS.top_k)

    last_global_step_val = -1
    while True:
      last_global_step_val = evaluation_loop(video_id_batch, prediction_batch,
                                             label_batch, loss, summary_op,
                                             saver, summary_writer, evl_metrics,
                                             last_global_step_val)
      if FLAGS.run_once:
        break


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  print("tensorflow version: %s" % tf.__version__)
  evaluate()


if __name__ == "__main__":
  app.run()

