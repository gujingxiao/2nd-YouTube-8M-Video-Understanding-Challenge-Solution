# -*- coding:UTF-8 -*-
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

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

FLAGS = flags.FLAGS
# FLAGS for General
flags.DEFINE_integer("iterations", 150,
                     "Number of frames per batch for DBoF.")

# ========== For MultiCombinedFeatureFrameModelLF ========================
# # FLAGS for NetVLAGDModelLF
# flags.DEFINE_integer("netvlad_cluster_size", 56,
#                      "Number of units in the NetVLAD cluster layer.")
# flags.DEFINE_integer("netvlad_hidden_size", 768,
#                      "Number of units in the NetVLAD hidden layer.")
#
# # FLAGS for NetFVModelLF
# flags.DEFINE_integer("fv_cluster_size", 56,
#                     "Number of units in the NetVLAD cluster layer.")
# flags.DEFINE_integer("fv_hidden_size", 768,
#                     "Number of units in the NetVLAD hidden layer.")
# flags.DEFINE_float("fv_coupling_factor", 0.01,
#                     "Coupling factor.")
#
# # FLAGS for Gated-Soft-DbofModelLF
# flags.DEFINE_integer("dbof_cluster_size", 2048,
#                     "Number of units in the DBoF cluster layer.")
# flags.DEFINE_integer("dbof_hidden_size", 512,
#                     "Number of units in the DBoF hidden layer.")
# flags.DEFINE_bool("softdbof_maxpool", False,
#                   'add max pool to soft dbof')
# ======================= End Here ========================================


# =============== For  GatedDbofWithNetFVModelLF 0.87089 ==================
# # FLAGS for NetFVModelLF
# flags.DEFINE_integer("fv_cluster_size", 52,
#                     "Number of units in the NetVLAD cluster layer.")
# flags.DEFINE_integer("fv_hidden_size", 1024,
#                     "Number of units in the NetVLAD hidden layer.")
# flags.DEFINE_float("fv_coupling_factor", 0.01,
#                     "Coupling factor.")
#
# # FLAGS for Gated-Soft-DbofModelLF
# flags.DEFINE_integer("dbof_cluster_size", 2560,
#                     "Number of units in the DBoF cluster layer.")
# flags.DEFINE_integer("dbof_hidden_size", 1024,
#                     "Number of units in the DBoF hidden layer.")
# flags.DEFINE_bool("softdbof_maxpool", False,
#                   'add max pool to soft dbof')
# ======================= End Here ========================================

# ========== For MultiEnsembleFrameModelLF ========================
# FLAGS for NetVLAGDModelLF
flags.DEFINE_integer("netvlad_cluster_size", 40,
                     "Number of units in the NetVLAD cluster layer.")
flags.DEFINE_integer("netvlad_hidden_size", 736,
                     "Number of units in the NetVLAD hidden layer.")

# FLAGS for NetFVModelLF
flags.DEFINE_integer("fv_cluster_size", 40,
                    "Number of units in the NetVLAD cluster layer.")
flags.DEFINE_integer("fv_hidden_size", 736,
                    "Number of units in the NetVLAD hidden layer.")
flags.DEFINE_float("fv_coupling_factor", 0.01,
                    "Coupling factor.")

# FLAGS for Gated-Soft-DbofModelLF
flags.DEFINE_integer("dbof_cluster_size", 2048,
                    "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 736,
                    "Number of units in the DBoF hidden layer.")
flags.DEFINE_bool("softdbof_maxpool", False,
                  'add max pool to soft dbof')
# ======================= End Here ========================================


# FLAGS for general purpose
flags.DEFINE_string("video_level_classifier_model", "MultiEnsembleChainModel",
                    "Some Frame-Level models can be decomposed into a generalized"
                    "pooling operation followed by a classifier layer")

# ====================================================== #
#                        NetVLAGD                        #
# ====================================================== #
class NetVLAGD():
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        cluster_weights = tf.get_variable("cluster_weights_netvlad", [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        activation = tf.matmul(reshaped_input, cluster_weights)
        activation = slim.batch_norm(activation, center=True, scale=True, is_training=self.is_training, scope="cluster_bn_netvlad")

        activation = tf.nn.softmax(activation)
        activation = tf.reshape(activation, [-1, int(self.max_frames), int(self.cluster_size)])

        gate_weights = tf.get_variable("gate_weights_netvlad", [1, int(self.cluster_size), int(self.feature_size)],
                                       initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        gate_weights = tf.sigmoid(gate_weights)
        activation = tf.transpose(activation, perm=[0, 2, 1])
        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])

        vlagd = tf.matmul(activation, reshaped_input)
        vlagd = tf.multiply(vlagd, gate_weights)
        vlagd = tf.transpose(vlagd, perm=[0, 2, 1])
        vlagd = tf.nn.l2_normalize(vlagd, 1)
        vlagd = tf.reshape(vlagd, [-1, int(self.cluster_size * self.feature_size)])
        vlagd = tf.nn.l2_normalize(vlagd, 1)

        return vlagd

# ==================================================== #
#                         NetFV                        #
# ==================================================== #
class NetFV():
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        cluster_weights = tf.get_variable("cluster_weights_netfv", [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        covar_weights = tf.get_variable("covar_weights_netfv", [self.feature_size, self.cluster_size],
                                        initializer=tf.random_normal_initializer(mean=1.0, stddev=1 / math.sqrt(self.feature_size)))
        covar_weights = tf.square(covar_weights)
        eps = tf.constant([1e-6])
        covar_weights = tf.add(covar_weights, eps)

        activation = tf.matmul(reshaped_input, cluster_weights)
        activation = slim.batch_norm(activation, center=True, scale=True, is_training=self.is_training, scope="cluster_bn_netfv")
        activation = tf.nn.softmax(activation)
        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)
        cluster_weights2 = tf.scalar_mul(FLAGS.fv_coupling_factor, cluster_weights)
        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])
        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
        fv1 = tf.matmul(activation, reshaped_input)
        fv1 = tf.transpose(fv1, perm=[0, 2, 1])

        # computing second order FV
        a2 = tf.multiply(a_sum, tf.square(cluster_weights2))
        b2 = tf.multiply(fv1, cluster_weights2)
        fv2 = tf.matmul(activation, tf.square(reshaped_input))

        fv2 = tf.transpose(fv2, perm=[0, 2, 1])
        fv2 = tf.add_n([a2, fv2, tf.scalar_mul(-2, b2)])

        fv2 = tf.divide(fv2, tf.square(covar_weights))
        fv2 = tf.subtract(fv2, a_sum)
        fv2 = tf.reshape(fv2, [-1, self.cluster_size * self.feature_size])

        fv2 = tf.nn.l2_normalize(fv2, 1)
        fv2 = tf.reshape(fv2, [-1, self.cluster_size * self.feature_size])
        fv2 = tf.nn.l2_normalize(fv2, 1)

        fv1 = tf.subtract(fv1, a)
        fv1 = tf.divide(fv1, covar_weights)

        fv1 = tf.nn.l2_normalize(fv1, 1)
        fv1 = tf.reshape(fv1, [-1, self.cluster_size * self.feature_size])
        fv1 = tf.nn.l2_normalize(fv1, 1)

        return tf.concat([fv1, fv2], 1)

# ================================================== #
#                Gated-Soft-DbofModel                #
# ================================================== #
class GatedDBoF():
    def __init__(self, feature_size, max_frames, cluster_size, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        feature_size = self.feature_size
        cluster_size = self.cluster_size
        max_frames = self.max_frames
        is_training = self.is_training

        cluster_weights = tf.get_variable("cluster_weights_gatedbof", [feature_size, cluster_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))

        activation = tf.matmul(reshaped_input, cluster_weights)
        activation = slim.batch_norm(activation, center=True, scale=True, is_training=is_training, scope="cluster_bn_gatedbof")

        activation = tf.nn.softmax(activation)
        activation = tf.reshape(activation, [-1, max_frames, cluster_size])
        activation_sum = tf.reduce_sum(activation, 1)

        activation_max = tf.reduce_max(activation, 1)
        activation_max = tf.nn.l2_normalize(activation_max, 1)

        dim_red = tf.get_variable("dim_red_gatedbof", [cluster_size, feature_size],
                                  initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))

        cluster_weights_2 = tf.get_variable("cluster_weights_2_gatedbof", [feature_size, cluster_size],
                                            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))

        activation = tf.matmul(activation_max, dim_red)
        activation = tf.matmul(activation, cluster_weights_2)
        activation = slim.batch_norm(activation, center=True, scale=True, is_training=is_training, scope="cluster_bn_2_gatedbof")

        activation = tf.sigmoid(activation)
        activation = tf.multiply(activation, activation_sum)
        activation = tf.nn.l2_normalize(activation, 1)

        return activation


class SoftDBoF():
    def __init__(self, feature_size, max_frames, cluster_size, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        feature_size = self.feature_size
        cluster_size = self.cluster_size
        max_frames = self.max_frames
        is_training = self.is_training

        cluster_weights = tf.get_variable("cluster_weights_softdbof", [feature_size, cluster_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))

        activation = tf.matmul(reshaped_input, cluster_weights)
        activation = slim.batch_norm(activation, center=True, scale=True, is_training=is_training, scope="cluster_bn_softdbof")

        activation = tf.nn.softmax(activation)
        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        activation_sum = tf.reduce_sum(activation, 1)
        activation_sum = tf.nn.l2_normalize(activation_sum, 1)

        activation_max = tf.reduce_max(activation, 1)
        activation_max = tf.nn.l2_normalize(activation_max, 1)
        activation = tf.concat([activation_sum, activation_max], 1)

        return activation



class GatedDbofWithNetFVModelLF(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, iterations=None,
                     add_batch_norm=None, sample_random_frames=None, cluster_size=None,
                     hidden_size=None, is_training=True, **unused_params):
        iterations = FLAGS.iterations
        dbof_cluster_size = FLAGS.dbof_cluster_size
        dbof_hidden_size = FLAGS.dbof_hidden_size
        fv_cluster_size = FLAGS.fv_cluster_size
        fv_hidden_size = FLAGS.fv_hidden_size

        # Process Input
        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])
        reshaped_input = slim.batch_norm(reshaped_input, center=True, scale=True, is_training=is_training, scope="input_bn")

        #====== Gated Dbof =======
        video_Dbof = GatedDBoF(1152, max_frames, dbof_cluster_size, is_training)

        with tf.variable_scope("feature_DBOF"):
            dbof = video_Dbof.forward(reshaped_input)

        dbof_dim = dbof.get_shape().as_list()[1]
        hidden1_weights_dbof = tf.get_variable("hidden1_weights_dbof", [dbof_dim, dbof_hidden_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(dbof_cluster_size)))

        activation_dbof = tf.matmul(dbof, hidden1_weights_dbof)
        activation_dbof = slim.batch_norm(activation_dbof, center=True, scale=True, is_training=is_training, scope="hidden1_bn_dbof")
        activation_dbof = tf.nn.elu(activation_dbof)

        #====== NetFVModel ======
        video_NetFV = NetFV(1152, max_frames, fv_cluster_size, add_batch_norm, is_training)

        with tf.variable_scope("video_FV"):
            fv = video_NetFV.forward(reshaped_input)

        fv_dim = fv.get_shape().as_list()[1]
        hidden1_weights_fv = tf.get_variable("hidden1_weights_netfv", [fv_dim, fv_hidden_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fv_cluster_size)))

        activation_fv = tf.matmul(fv, hidden1_weights_fv)
        activation_fv = slim.batch_norm(activation_fv, center=True, scale=True, is_training=is_training, scope="hidden1_bn_netfv")
        activation_fv = tf.nn.leaky_relu(activation_fv)

        # Gating
        gating_weights_fv = tf.get_variable("gating_weights_2", [fv_hidden_size, fv_hidden_size],
                                         initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fv_hidden_size)))
        gates_fv = tf.matmul(activation_fv, gating_weights_fv)
        gates_fv = slim.batch_norm(gates_fv, center=True, scale=True, is_training=is_training, scope="gating_bn_netfv")

        gates_fv = tf.sigmoid(gates_fv)
        activation_fv = tf.multiply(activation_fv, gates_fv)

        # Final activation
        final_activation = tf.concat([activation_fv, activation_dbof], 1)
        print('shape:', final_activation.get_shape().as_list())
        aggregated_model = getattr(video_level_models, FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=final_activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)

class MultiCombinedFeatureFrameModelLF(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, iterations=None,
                     add_batch_norm=None, sample_random_frames=None, cluster_size=None,
                     hidden_size=None, is_training=True, **unused_params):
        iterations = FLAGS.iterations
        dbof_cluster_size = FLAGS.dbof_cluster_size
        dbof_hidden_size = FLAGS.dbof_hidden_size
        fv_cluster_size = FLAGS.fv_cluster_size
        fv_hidden_size = FLAGS.fv_hidden_size
        netvlad_cluster_size = FLAGS.netvlad_cluster_size
        netvlad_hidden_size = FLAGS.netvlad_hidden_size

        #======= Process Input ========
        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])
        reshaped_input = slim.batch_norm(reshaped_input, center=True, scale=True, is_training=is_training, scope="input_bn")

        #====== Soft Dbof =======
        feature_Dbof = SoftDBoF(1152, max_frames, dbof_cluster_size, is_training)

        with tf.variable_scope("feature_DBOF"):
            dbof = feature_Dbof.forward(reshaped_input)

        dbof_dim = dbof.get_shape().as_list()[1]
        hidden1_weights_dbof = tf.get_variable("hidden1_weights_dbof", [dbof_dim, dbof_hidden_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(dbof_cluster_size)))

        activation_dbof = tf.matmul(dbof, hidden1_weights_dbof)
        activation_dbof = slim.batch_norm(activation_dbof, center=True, scale=True, is_training=is_training, scope="hidden1_bn_dbof")
        activation_dbof = tf.nn.elu(activation_dbof)

        #====== NetFVModel ======
        feature_NetFV = NetFV(1152, max_frames, fv_cluster_size, add_batch_norm, is_training)

        with tf.variable_scope("feature_FV"):
            fv = feature_NetFV.forward(reshaped_input)

        fv_dim = fv.get_shape().as_list()[1]
        hidden1_weights_fv = tf.get_variable("hidden1_weights_netfv", [fv_dim, fv_hidden_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fv_cluster_size)))

        activation_fv = tf.matmul(fv, hidden1_weights_fv)
        activation_fv = slim.batch_norm(activation_fv, center=True, scale=True, is_training=is_training, scope="hidden1_bn_netfv")
        activation_fv = tf.nn.leaky_relu(activation_fv)

        # Gating
        gating_weights_fv = tf.get_variable("gating_weights_netfv", [fv_hidden_size, fv_hidden_size],
                                         initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fv_hidden_size)))
        gates_fv = tf.matmul(activation_fv, gating_weights_fv)
        gates_fv = slim.batch_norm(gates_fv, center=True, scale=True, is_training=is_training, scope="gating_bn_netfv")

        gates_fv = tf.sigmoid(gates_fv)
        activation_fv = tf.multiply(activation_fv, gates_fv)

        # ====== NetVladModel ======
        feature_NetVLAD = NetVLAGD(1152, max_frames, netvlad_cluster_size, True, is_training)

        with tf.variable_scope("feature_VLAD"):
            vlad = feature_NetVLAD.forward(reshaped_input)

        vlad_dim = vlad.get_shape().as_list()[1]
        hidden1_weights = tf.get_variable("hidden1_weights_netvlad", [vlad_dim, netvlad_hidden_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(netvlad_cluster_size)))

        # Batch norm and Relu
        activation_vlad = tf.matmul(vlad, hidden1_weights)
        activation_vlad = slim.batch_norm(activation_vlad, center=True, scale=True, is_training=is_training, scope="hidden1_bn_netvlad")
        activation_vlad = tf.nn.leaky_relu(activation_vlad)

        # Gating
        gating_weights_vlad = tf.get_variable("gating_weights_netvlad", [netvlad_hidden_size, netvlad_hidden_size],
                                         initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(netvlad_hidden_size)))
        gates_vlad = tf.matmul(activation_vlad, gating_weights_vlad)

        # Batch norm
        gates_vlad = slim.batch_norm(gates_vlad, center=True, scale=True, is_training=is_training, scope="gating_bn_netvlad")

        # Activations
        gates_vlad = tf.sigmoid(gates_vlad)
        activation_vlad = tf.multiply(activation_vlad, gates_vlad)

        # Final activation
        final_activation = tf.concat([activation_fv, activation_dbof, activation_vlad], 1)
        print('shape:', final_activation.get_shape().as_list())
        aggregated_model = getattr(video_level_models, FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=final_activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


class MultiEnsembleFrameModelLF(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, iterations=None,
                     add_batch_norm=None, sample_random_frames=None, cluster_size=None,
                     hidden_size=None, is_training=True, **unused_params):
        iterations = FLAGS.iterations
        dbof_cluster_size = FLAGS.dbof_cluster_size
        dbof_hidden_size = FLAGS.dbof_hidden_size
        fv_cluster_size = FLAGS.fv_cluster_size
        fv_hidden_size = FLAGS.fv_hidden_size
        netvlad_cluster_size = FLAGS.netvlad_cluster_size
        netvlad_hidden_size = FLAGS.netvlad_hidden_size

        #======= Process Input ========
        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])
        reshaped_input = slim.batch_norm(reshaped_input, center=True, scale=True, is_training=is_training, scope="input_bn")

        #====== Soft Dbof =======
        feature_SoftDbof = SoftDBoF(1152, max_frames, dbof_cluster_size, is_training)

        with tf.variable_scope("feature_SoftDBOF"):
            softdbof = feature_SoftDbof.forward(reshaped_input)

        softdbof_dim = softdbof.get_shape().as_list()[1]
        hidden1_weights_softdbof = tf.get_variable("hidden1_weights_softdbof", [softdbof_dim, dbof_hidden_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(dbof_cluster_size)))

        activation_softdbof = tf.matmul(softdbof, hidden1_weights_softdbof)
        activation_softdbof = slim.batch_norm(activation_softdbof, center=True, scale=True, is_training=is_training, scope="hidden1_bn_softdbof")
        activation_softdbof = tf.nn.leaky_relu(activation_softdbof)

        #====== Gated Dbof =======
        feature_GatedDbof = GatedDBoF(1152, max_frames, dbof_cluster_size, is_training)

        with tf.variable_scope("feature_GatedDBOF"):
            gatedbof = feature_GatedDbof.forward(reshaped_input)

        gatedbof_dim = gatedbof.get_shape().as_list()[1]
        hidden1_weights_gatedbof = tf.get_variable("hidden1_weights_gateddbof", [gatedbof_dim, dbof_hidden_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(dbof_cluster_size)))

        activation_gatedbof = tf.matmul(gatedbof, hidden1_weights_gatedbof)
        activation_gatedbof = slim.batch_norm(activation_gatedbof, center=True, scale=True, is_training=is_training, scope="hidden1_bn_gatedbof")
        activation_gatedbof = tf.nn.elu(activation_gatedbof)

        #====== NetFVModel ======
        feature_NetFV = NetFV(1152, max_frames, fv_cluster_size, add_batch_norm, is_training)

        with tf.variable_scope("feature_FV"):
            fv = feature_NetFV.forward(reshaped_input)

        fv_dim = fv.get_shape().as_list()[1]
        hidden1_weights_fv = tf.get_variable("hidden1_weights_netfv", [fv_dim, fv_hidden_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fv_cluster_size)))

        activation_fv = tf.matmul(fv, hidden1_weights_fv)
        activation_fv = slim.batch_norm(activation_fv, center=True, scale=True, is_training=is_training, scope="hidden1_bn_netfv")
        activation_fv = tf.nn.leaky_relu(activation_fv)

        # Gating
        gating_weights_fv = tf.get_variable("gating_weights_netfv", [fv_hidden_size, fv_hidden_size],
                                         initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fv_hidden_size)))
        gates_fv = tf.matmul(activation_fv, gating_weights_fv)
        gates_fv = slim.batch_norm(gates_fv, center=True, scale=True, is_training=is_training, scope="gating_bn_netfv")

        gates_fv = tf.sigmoid(gates_fv)
        activation_fv = tf.multiply(activation_fv, gates_fv)

        # ====== NetVladModel ======
        feature_NetVLAD = NetVLAGD(1152, max_frames, netvlad_cluster_size, True, is_training)

        with tf.variable_scope("feature_VLAD"):
            vlad = feature_NetVLAD.forward(reshaped_input)

        vlad_dim = vlad.get_shape().as_list()[1]
        hidden1_weights = tf.get_variable("hidden1_weights_netvlad", [vlad_dim, netvlad_hidden_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(netvlad_cluster_size)))

        # Batch norm and Relu
        activation_vlad = tf.matmul(vlad, hidden1_weights)
        activation_vlad = slim.batch_norm(activation_vlad, center=True, scale=True, is_training=is_training, scope="hidden1_bn_netvlad")
        activation_vlad = tf.nn.elu(activation_vlad)

        # Gating
        gating_weights_vlad = tf.get_variable("gating_weights_netvlad", [netvlad_hidden_size, netvlad_hidden_size],
                                         initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(netvlad_hidden_size)))
        gates_vlad = tf.matmul(activation_vlad, gating_weights_vlad)

        # Batch norm
        gates_vlad = slim.batch_norm(gates_vlad, center=True, scale=True, is_training=is_training, scope="gating_bn_netvlad")

        # Activations
        gates_vlad = tf.sigmoid(gates_vlad)
        activation_vlad = tf.multiply(activation_vlad, gates_vlad)

        # Final activation
        softdbof_netvlad_activation = tf.concat([activation_softdbof, activation_vlad], 1)
        gatedbof_netfv_activation = tf.concat([activation_gatedbof, activation_fv], 1)
        aggregated_model = getattr(video_level_models, FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input1=softdbof_netvlad_activation,
            model_input2=gatedbof_netfv_activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)
