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

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
# MoeModel
flags.DEFINE_integer("moe_num_mixtures", 3,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

# DeepChainModel
flags.DEFINE_integer("deep_chain_layers", 3,
    "The number of layers used for DeepChainModel，建议使用3,4")
flags.DEFINE_integer("deep_chain_relu_cells", 1024,
    "The number of relu cells used for DeepChainModel，建议使用256,512,1024")

# MultiTaskDeepChainModel
flags.DEFINE_integer("chain_layers_1", 3,
    "The number of layers used for DeepChainModel，建议使用3, 4")
flags.DEFINE_integer("chain_elu_cells", 896,
    "The number of relu cells used for DeepChainModel，建议大于256")
flags.DEFINE_integer("chain_layers_2", 2,
    "The number of layers used for DeepChainModel，建议使用2, 3")
flags.DEFINE_integer("chain_leaky_relu_cells", 896,
    "The number of relu cells used for DeepChainModel，建议大于256")


class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=3,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = FLAGS.moe_num_mixtures # 当等于25时为小于1G的最大模型，GAP=0.840193

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")

    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:,:num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

class DeepCombineChainModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None,
                   dropout=False, keep_prob=None, **unused_params):

    num_layers = FLAGS.deep_chain_layers
    relu_cells = FLAGS.deep_chain_relu_cells

    next_input = model_input
    support_lists = []
    for layer in range(num_layers):
      sub_prediction = self.sub_model(next_input, 1536, num_mixtures=2, sub_scope=sub_scope+"prediction-%d"%layer,
                                      dropout=dropout, keep_prob=keep_prob)
      sub_activation = slim.fully_connected(sub_prediction, relu_cells, activation_fn=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty), scope=sub_scope+"relu-%d"%layer)

      sub_relu = tf.nn.elu(sub_activation)
      relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
      next_input = tf.concat([next_input, relu_norm], axis=1)
      support_lists.append(sub_prediction)

    main_predictions = self.sub_model(next_input, vocab_size, num_mixtures=4, sub_scope=sub_scope+"-main")

    support_lists = tf.stack(support_lists, axis=1)
    support_activations = tf.reduce_mean(support_lists, axis=1)
    support_predictions = self.sub_model(support_activations, vocab_size, num_mixtures=4, sub_scope=sub_scope+"-support")

    return {"predictions": main_predictions, "support_predictions": support_predictions}

  def sub_model(self, model_input, vocab_size, num_mixtures=None,
                l2_penalty=1e-8, sub_scope="",
                dropout=False, keep_prob=None, **unused_params):
    num_mixtures = num_mixtures

    if dropout:
      model_input = tf.nn.dropout(model_input, keep_prob=keep_prob)

    gate_activations = slim.fully_connected(model_input, vocab_size * (num_mixtures + 1), activation_fn=None,
        biases_initializer=None, weights_regularizer=slim.l2_regularizer(l2_penalty), scope="gates-"+sub_scope)
    expert_activations = slim.fully_connected(model_input, vocab_size * num_mixtures, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope="experts-"+sub_scope)

    gating_distribution = tf.nn.softmax(tf.reshape(gate_activations, [-1, num_mixtures + 1]))
    expert_distribution = tf.nn.sigmoid(tf.reshape(expert_activations, [-1, num_mixtures]))

    final_probabilities_by_class_and_batch = tf.reduce_sum(gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch, [-1, vocab_size])
    return final_probabilities


class MultiTaskCombineChainModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None,
                   dropout=False, keep_prob=None, **unused_params):

    num_layers_1 = FLAGS.chain_layers_1
    elu_cells = FLAGS.chain_elu_cells
    num_layers_2 = FLAGS.chain_layers_2
    leaky_relu_cells = FLAGS.chain_leaky_relu_cells

    next_input_1 = model_input
    next_input_2 = model_input
    support_lists_1 = []
    support_lists_2 = []
    for layer in range(num_layers_1):
      sub_prediction_1 = self.sub_model(next_input_1, 896, num_mixtures=2, sub_scope=sub_scope+"prediction1-%d"%layer,
                                      dropout=dropout, keep_prob=keep_prob)
      sub_activation_1 = slim.fully_connected(sub_prediction_1, elu_cells, activation_fn=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty), scope=sub_scope+"elu-%d"%layer)

      sub_elu = tf.nn.elu(sub_activation_1)
      elu_norm = tf.nn.l2_normalize(sub_elu, dim=1)
      next_input_1 = tf.concat([next_input_1, elu_norm], axis=1)
      support_lists_1.append(sub_prediction_1)

    main_predictions_1 = self.sub_model(next_input_1, vocab_size, num_mixtures=3, sub_scope=sub_scope+"-main1")
    support_lists_1 = tf.stack(support_lists_1, axis=1)
    support_activations_1 = tf.reduce_mean(support_lists_1, axis=1)
    support_predictions_1 = self.sub_model(support_activations_1, vocab_size, num_mixtures=2, sub_scope=sub_scope+"-support1")

    for layer in range(num_layers_2):
      sub_prediction_2 = self.sub_model(next_input_2, 896, num_mixtures=2, sub_scope=sub_scope+"prediction2-%d"%layer,
                                      dropout=dropout, keep_prob=keep_prob)
      sub_activation_2 = slim.fully_connected(sub_prediction_2, leaky_relu_cells, activation_fn=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty), scope=sub_scope+"leakyrelu-%d"%layer)

      sub_leakyrelu = tf.nn.leaky_relu(sub_activation_2)
      leakyrelu_norm = tf.nn.l2_normalize(sub_leakyrelu, dim=1)
      next_input_2 = tf.concat([next_input_2, leakyrelu_norm], axis=1)
      support_lists_2.append(sub_prediction_2)

    main_predictions_2 = self.sub_model(next_input_2, vocab_size, num_mixtures=3, sub_scope=sub_scope+"-main2")
    support_lists_2 = tf.stack(support_lists_2, axis=1)
    support_activations_2 = tf.reduce_mean(support_lists_2, axis=1)
    support_predictions_2 = self.sub_model(support_activations_2, vocab_size, num_mixtures=2, sub_scope=sub_scope+"-support2")


    return {"predictions": main_predictions_1, "support_predictions": support_predictions_1, "predictions2": main_predictions_2, "support_predictions2": support_predictions_2}

  def sub_model(self, model_input, vocab_size, num_mixtures=None,
                l2_penalty=1e-8, sub_scope="",
                dropout=False, keep_prob=None, **unused_params):
    num_mixtures = num_mixtures

    if dropout:
      model_input = tf.nn.dropout(model_input, keep_prob=keep_prob)

    gate_activations = slim.fully_connected(model_input, vocab_size * (num_mixtures + 1), activation_fn=None,
        biases_initializer=None, weights_regularizer=slim.l2_regularizer(l2_penalty), scope="gates-"+sub_scope)
    expert_activations = slim.fully_connected(model_input, vocab_size * num_mixtures, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope="experts-"+sub_scope)

    gating_distribution = tf.nn.softmax(tf.reshape(gate_activations, [-1, num_mixtures + 1]))
    expert_distribution = tf.nn.sigmoid(tf.reshape(expert_activations, [-1, num_mixtures]))

    final_probabilities_by_class_and_batch = tf.reduce_sum(gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch, [-1, vocab_size])
    return final_probabilities


