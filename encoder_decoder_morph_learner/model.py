import sys
import os
import re
import math
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers.utils import conv_output_length

from .kwargs import ENCODER_DECODER_MORPH_LEARNER_INITIALIZATION_KWARGS
from .util import sn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

if hasattr(rnn_cell_impl, 'LayerRNNCell'):
    LayerRNNCell = rnn_cell_impl.LayerRNNCell
else:
    LayerRNNCell = rnn_cell_impl._LayerRNNCell

is_embedding_dimension = re.compile('d([0-9]+)')


def get_session(session):
    if session is None:
        sess = tf.get_default_session()
    else:
        sess = session

    return sess


def get_activation(activation, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if activation:
                if activation == 'hard_sigmoid':
                    out = tf.keras.backend.hard_sigmoid
                elif activation == 'bsn_round':
                    out = lambda x: round_straight_through(tf.sigmoid(x), session=session)
                elif activation == 'bsn_bernoulli':
                    out = lambda x: bernoulli_straight_through(tf.sigmoid(x), session=session)
                elif isinstance(activation, str):
                    out = getattr(tf.nn, activation)
                else:
                    out = activation
            else:
                out = lambda x: x

    return out


def get_initializer(initializer, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if isinstance(initializer, str):
                out = getattr(tf, initializer)
            else:
                out = initializer

            if 'glorot' in initializer:
                out = out()

            return out


def get_regularizer(init, scale=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if scale is None:
                scale = 0.001

            if init is None:
                out = None
            elif isinstance(init, str):
                out = getattr(tf.contrib.layers, init)(scale=scale)
            elif isinstance(init, float):
                out = tf.contrib.layers.l2_regularizer(scale=init)

            return out


def round_straight_through(x, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            with ops.name_scope("BinaryRound") as name:
                with session.graph.gradient_override_map({"Round": "Identity"}):
                    return tf.round(x, name=name)


def bernoulli_straight_through(x, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            with ops.name_scope("BernoulliSample") as name:
                with session.graph.gradient_override_map({"Ceil": "Identity", "Sub": "BernoulliSample_ST"}):
                    return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)


class MultiLSTMCell(LayerRNNCell):
    def __init__(
            self,
            num_units,
            num_layers,
            forget_bias=1.0,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            kernel_initializer='glorot_uniform_initializer',
            bias_initializer='zeros_initializer',
            refeed_outputs=False,
            reuse=None,
            name=None,
            dtype=None,
            session=None
    ):
        self.session = get_session(session)

        with self.session.as_default():
            with self.session.graph.as_default():
                super(MultiLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

                if not isinstance(num_units, list):
                    self._num_units = [num_units] * num_layers
                else:
                    self._num_units = num_units

                assert len(self._num_units) == num_layers, 'num_units must either be an integer or a list of integers of length num_layers'

                self._num_layers = num_layers
                self._forget_bias = forget_bias

                self._activation = get_activation(activation, session=self.session)
                self._inner_activation = get_activation(inner_activation, session=self.session)
                self._recurrent_activation = get_activation(recurrent_activation, session=self.session)

                self._kernel_initializer = get_initializer(kernel_initializer, session=self.session)
                self._bias_initializer = get_initializer(bias_initializer, session=self.session)

                self._refeed_outputs = refeed_outputs

    def _regularize(self, var, regularizer):
        if regularizer is not None:
            with self.session.as_default():
                with self.session.graph.as_default():
                    reg = tf.contrib.layers.apply_regularization(regularizer, [var])
                    self.regularizer_losses.append(reg)

    @property
    def state_size(self):
        out = []
        for l in range(self._num_layers):
            size = (self._num_units[l], self._num_units[l])
            out.append(size)

        out = tuple(out)

        return out

    @property
    def output_size(self):
        out = self._num_units[-1]

        return out

    def build(self, inputs_shape):
        with self.session.as_default():
            with self.session.graph.as_default():
                if inputs_shape[1].value is None:
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                     % inputs_shape)

                self._kernel = []
                self._bias = []

                for l in range(self._num_layers):
                    if l == 0:
                        bottom_up_dim = inputs_shape[1].value
                    else:
                        bottom_up_dim = self._num_units[l-1]

                    recurrent_dim = self._num_units[l]
                    output_dim = 4 * self._num_units[l]
                    if self._refeed_outputs and l == 0:
                        refeed_dim = self._num_units[-1]
                    else:
                        refeed_dim = 0

                    kernel = self.add_variable(
                        'kernel_%d' %l,
                        shape=[bottom_up_dim + recurrent_dim + refeed_dim, output_dim],
                        initializer=self._kernel_initializer
                    )
                    self._kernel.append(kernel)

                    bias = self.add_variable(
                        'bias_%d' %l,
                        shape=[1, output_dim],
                        initializer=self._bias_initializer
                    )
                    self._bias.append(bias)

        self.built = True

    def call(self, inputs, state):
        with self.session.as_default():
            new_state = []

            h_below = inputs
            for l, layer in enumerate(state):
                c_behind, h_behind = layer

                # Gather inputs
                layer_inputs = [h_below, h_behind]

                if self._refeed_outputs and l == 0 and self._num_layers > 1:
                    layer_inputs.append(state[-1][1])

                # Compute gate pre-activations
                s = tf.matmul(
                    tf.concat(layer_inputs, axis=1),
                    self._kernel[l]
                )

                # Add bias
                s = s + self._bias[l]

                # Alias useful variables
                if l < self._num_layers - 1:
                    # Use inner activation if non-final layer
                    activation = self._inner_activation
                else:
                    # Use outer activation if final layer
                    activation = self._activation
                units = self._num_units[l]

                # Forget gate
                f = self._recurrent_activation(s[:, :units] + self._forget_bias)
                # Input gate
                i = self._recurrent_activation(s[:, units:units * 2])
                # Output gate
                o = self._recurrent_activation(s[:, units * 2:units * 3])
                # Cell proposal
                g = activation(s[:, units * 3:units * 4])

                # Compute new cell state
                c = f * c_behind + i * g

                # Compute the gated output
                h = o * activation(c)

                new_state.append((c, h))

                h_below = h

            new_state = tuple(new_state)
            new_output = new_state[-1][1]

            return new_output, new_state


class DenseLayer(object):

    def __init__(
            self,
            training,
            units=None,
            use_bias=True,
            activation=None,
            batch_normalization_decay=0.9,
            normalize_weights=False,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.use_bias = use_bias
        self.activation = get_activation(activation, session=self.session)
        self.batch_normalization_decay = batch_normalization_decay
        self.normalize_weights = normalize_weights

        self.dense_layer = None
        self.projection = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            if self.units is None:
                out_dim = inputs.shape[-1]
            else:
                out_dim = self.units

            with self.session.as_default():
                with self.session.graph.as_default():
                    self.dense_layer = tf.keras.layers.Dense(
                        out_dim,
                        input_shape=[inputs.shape[1]],
                        use_bias=self.use_bias
                    )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                H = self.dense_layer(inputs)

                if self.normalize_weights:
                    self.w = self.dense_layer.kernel
                    self.g = tf.Variable(tf.ones(self.w.shape[1]), dtype=tf.float32)
                    self.v = tf.norm(self.w, axis=0)
                    self.dense_layer.kernel = self.v

                if self.batch_normalization_decay:
                    # H = tf.layers.batch_normalization(H, training=self.training)
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None
                    )
                if self.activation is not None:
                    H = self.activation(H)

                return H


class DenseResidualLayer(object):

    def __init__(
            self,
            training,
            units=None,
            use_bias=True,
            layers_inner=3,
            activation_inner=None,
            activation=None,
            batch_normalization_decay=0.9,
            project_inputs=False,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.use_bias = use_bias
        self.layers_inner = layers_inner
        self.activation_inner = get_activation(activation_inner, session=self.session)
        self.activation = get_activation(activation, session=self.session)
        self.batch_normalization_decay = batch_normalization_decay
        self.project_inputs = project_inputs

        self.dense_layers = None
        self.projection = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    if self.units is None:
                        out_dim = inputs.shape[-1]
                    else:
                        out_dim = self.units

                    self.dense_layers = []

                    for i in range(self.layers_inner):
                        if i == 0:
                            in_dim = inputs.shape[1]
                        else:
                            in_dim = out_dim
                        l = tf.keras.layers.Dense(
                            out_dim,
                            input_shape=[in_dim],
                            use_bias=self.use_bias
                        )
                        self.dense_layers.append(l)

                    if self.project_inputs:
                        self.projection = tf.keras.layers.Dense(
                            out_dim,
                            input_shape=[inputs.shape[1]]
                        )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                F = inputs
                for i in range(self.layers_inner - 1):
                    F = self.dense_layers[i](F)
                    if self.batch_normalization_decay:
                        # F = tf.layers.batch_normalization(F, training=self.training)
                        F = tf.contrib.layers.batch_norm(
                            F,
                            decay=self.batch_normalization_decay,
                            center=True,
                            scale=True,
                            zero_debias_moving_mean=True,
                            is_training=self.training,
                            updates_collections=None
                        )
                    if self.activation_inner is not None:
                        F = self.activation_inner(F)

                F = self.dense_layers[-1](F)
                if self.batch_normalization_decay:
                    # F = tf.layers.batch_normalization(F, training=self.training)
                    F = tf.contrib.layers.batch_norm(
                        F,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None
                    )

                if self.project_inputs:
                    x = self.projection(inputs)
                else:
                    x = inputs

                H = F + x

                if self.activation is not None:
                    H = self.activation(H)

                return H


class Conv1DLayer(object):

    def __init__(
            self,
            training,
            kernel_size,
            n_filters=None,
            stride=1,
            padding='valid',
            use_bias=True,
            activation=None,
            batch_normalization_decay=0.9,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.activation = get_activation(activation, session=self.session)
        self.batch_normalization_decay = batch_normalization_decay

        self.conv_1d_layer = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    if self.n_filters is None:
                        out_dim = inputs.shape[-1]
                    else:
                        out_dim = self.n_filters

                    self.conv_1d_layer = tf.keras.layers.Conv1D(
                        out_dim,
                        self.kernel_size,
                        padding=self.padding,
                        strides=self.stride,
                        use_bias=self.use_bias
                    )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                H = inputs

                H = self.conv_1d_layer(H)

                if self.batch_normalization_decay:
                    # H = tf.layers.batch_normalization(H, training=self.training)
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None
                    )

                if self.activation is not None:
                    H = self.activation(H)

                return H


class Conv1DResidualLayer(object):

    def __init__(
            self,
            training,
            kernel_size,
            n_filters=None,
            stride=1,
            padding='valid',
            use_bias=True,
            layers_inner=3,
            activation=None,
            activation_inner=None,
            batch_normalization_decay=0.9,
            project_inputs=False,
            n_timesteps=None,
            n_input_features=None,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.layers_inner = layers_inner
        self.activation = get_activation(activation, session=self.session)
        self.activation_inner = get_activation(activation_inner, session=self.session)
        self.batch_normalization_decay = batch_normalization_decay
        self.project_inputs = project_inputs
        self.n_timesteps = n_timesteps
        self.n_input_features = n_input_features

        self.conv_1d_layers = None
        self.projection = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            if self.n_filters is None:
                out_dim = inputs.shape[-1]
            else:
                out_dim = self.n_filters

            self.built = True

            self.conv_1d_layers = []

            with self.session.as_default():
                with self.session.graph.as_default():

                    conv_output_shapes = [[int(inputs.shape[1]), int(inputs.shape[2])]]

                    for i in range(self.layers_inner):
                        if isinstance(self.stride, list):
                            cur_strides = self.stride[i]
                        else:
                            cur_strides = self.stride

                        l = tf.keras.layers.Conv1D(
                            out_dim,
                            self.kernel_size,
                            padding=self.padding,
                            strides=cur_strides,
                            use_bias=self.use_bias
                        )

                        if self.padding in ['causal', 'same'] and self.stride == 1:
                            output_shape = conv_output_shapes[-1]
                        else:
                            output_shape = [
                                conv_output_length(
                                    x,
                                    self.kernel_size,
                                    self.padding,
                                    self.stride
                                ) for x in conv_output_shapes[-1]
                            ]

                        conv_output_shapes.append(output_shape)

                        self.conv_1d_layers.append(l)

                    self.conv_output_shapes = conv_output_shapes

                    if self.project_inputs:
                        self.projection = tf.keras.layers.Dense(
                            self.conv_output_shapes[-1][0] * out_dim,
                            input_shape=[self.conv_output_shapes[0][0] * self.conv_output_shapes[0][1]]
                        )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                F = inputs

                for i in range(self.layers_inner - 1):
                    F = self.conv_1d_layers[i](F)

                    if self.batch_normalization_decay:
                        # F = tf.layers.batch_normalization(F, training=self.training)
                        F = tf.contrib.layers.batch_norm(
                            F,
                            decay=self.batch_normalization_decay,
                            center=True,
                            scale=True,
                            zero_debias_moving_mean=True,
                            is_training=self.training,
                            updates_collections=None
                        )
                    if self.activation_inner is not None:
                        F = self.activation_inner(F)

                F = self.conv_1d_layers[-1](F)

                if self.batch_normalization_decay:
                    # F = tf.layers.batch_normalization(F, training=self.training)
                    F = tf.contrib.layers.batch_norm(
                        F,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None
                    )

                if self.project_inputs:
                    x = tf.layers.Flatten()(inputs)
                    x = self.projection(x)
                    x = tf.reshape(x, tf.shape(F))
                else:
                    x = inputs

                H = F + x

                if self.activation is not None:
                    H = self.activation(H)

                return H


class RNNLayer(object):

    def __init__(
            self,
            units=None,
            activation=None,
            recurrent_activation='sigmoid',
            kernel_initializer='glorot_uniform_initializer',
            bias_initializer='zeros_initializer',
            refeed_outputs = False,
            return_sequences=True,
            name=None,
            session=None
    ):
        self.session = get_session(session)

        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.refeed_outputs = refeed_outputs
        self.return_sequences = return_sequences
        self.name = name

        self.rnn_layer = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    RNN = tf.keras.layers.LSTM

                    if self.units:
                        output_dim = self.units
                    else:
                        output_dim = inputs.shape[-1]

                    self.rnn_layer = RNN(
                        output_dim,
                        return_sequences=self.return_sequences,
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation
                    )

            self.built = True

    def __call__(self, inputs, mask=None):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                H = self.rnn_layer(inputs, mask=mask)

                return H


class MultiRNNLayer(object):

    def __init__(
            self,
            units=None,
            layers=1,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            kernel_initializer='glorot_uniform_initializer',
            bias_initializer='zeros_initializer',
            refeed_outputs = False,
            return_sequences=True,
            name=None,
            session=None
    ):
        self.session = get_session(session)

        self.units = units
        self.layers = layers
        self.activation = activation
        self.inner_activation = inner_activation
        self.recurrent_activation = recurrent_activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.refeed_outputs = refeed_outputs
        self.return_sequences = return_sequences
        self.name = name

        self.rnn_layer = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    # RNN = getattr(tf.keras.layers, self.rnn_type)

                    if self.units is None:
                        units = [inputs.shape[-1]] * self.layers
                    else:
                        units = self.units

                    # self.rnn_layer = RNN(
                    #     out_dim,
                    #     return_sequences=self.return_sequences,
                    #     activation=self.activation,
                    #     recurrent_activation=self.recurrent_activation
                    # )
                    # self.rnn_layer = tf.contrib.rnn.BasicLSTMCell(
                    #     out_dim,
                    #     activation=self.activation,
                    #     name=self.name
                    # )

                    self.rnn_layer = MultiLSTMCell(
                        units,
                        self.layers,
                        activation=self.activation,
                        inner_activation=self.inner_activation,
                        recurrent_activation=self.recurrent_activation,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        refeed_outputs=self.refeed_outputs,
                        name=self.name,
                        session=self.session
                    )

            self.built = True

    def __call__(self, inputs, mask=None):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                # H = self.rnn_layer(inputs, mask=mask)
                if mask is None:
                    sequence_length = None
                else:
                    sequence_length = tf.reduce_sum(mask, axis=1)

                H, _ = tf.nn.dynamic_rnn(
                    self.rnn_layer,
                    inputs,
                    sequence_length=sequence_length,
                    dtype=tf.float32
                )

                if not self.return_sequences:
                    H = H[:,-1]

                return H


class EncoderDecoderMorphLearner(object):

    ############################################################
    # Initialization methods
    ############################################################

    _INITIALIZATION_KWARGS = ENCODER_DECODER_MORPH_LEARNER_INITIALIZATION_KWARGS

    _doc_header = """
        Encoder-decoder morphology learner.
    """
    _doc_args = """
        :param morph_feature_map: ``dict``; dictionary of morphological feature keys/values
        :param vocab: ``list``; list of all unique character symbols
    \n"""
    _doc_kwargs = '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join(
        [x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (
                                 x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value)
                             for
                             x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_args + _doc_kwargs

    def __init__(self, morph_feature_map, vocab, **kwargs):

        self.morph_feature_map = morph_feature_map
        self.vocab = sorted(list(set(vocab)))
        for kwarg in EncoderDecoderMorphLearner._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        self.plot_ix = None

        self._initialize_session()
        self._initialize_metadata()

    def _initialize_session(self):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

    def _initialize_metadata(self):
        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)
        self.UINT_TF = getattr(np, 'u' + self.int_type)
        self.UINT_NP = getattr(tf, 'u' + self.int_type)
        self.regularizer_losses = []

        self.ix_to_char = [None] + self.vocab
        self.char_to_ix = {}
        for i, c in enumerate(self.ix_to_char):
            self.char_to_ix[i] = c
        self.vocab_size = len(self.char_to_ix)

        if isinstance(self.n_units_encoder, str):
            self.units_encoder = [int(x) for x in self.n_units_encoder.split()]
            if len(self.units_encoder) == 1:
                self.units_encoder = [self.units_encoder[0]] * self.n_layers_encoder
        elif isinstance(self.n_units_encoder, int):
            self.units_encoder = [self.n_units_encoder] * self.n_layers_encoder
        else:
            self.units_encoder = self.n_units_encoder
        assert len(self.units_encoder) == self.n_layers_encoder, 'Misalignment in number of layers between n_layers_encoder and n_units_encoder.'

        if isinstance(self.n_units_decoder, str):
            self.units_decoder = [int(x) for x in self.n_units_decoder.split()]
            if len(self.units_decoder) == 1:
                self.units_decoder = [self.units_decoder[0]] * (self.n_layers_decoder - 1)
        elif isinstance(self.n_units_decoder, int):
            self.units_decoder = [self.n_units_decoder] * (self.n_layers_decoder - 1)
        else:
            self.units_decoder = self.n_units_decoder
        assert len(self.units_decoder) == (self.n_layers_decoder - 1), 'Misalignment in number of layers between n_layers_decoder and n_units_decoder.'

        if self.pad_seqs:
            if self.mask_padding and ('hmlstm' in self.encoder_type.lower() or 'rnn' in self.encoder_type.lower()):
                self.input_padding = 'post'
            else:
                self.input_padding = 'pre'
            self.target_padding = 'post'
        else:
            self.input_padding = None
            self.target_padding = None

        self.n_timestamps = self.max_len

    def _pack_metadata(self):
        md = {
            'morph_feature_map': self.morph_feature_map,
            'vocab': self.vocab,
        }
        for kwarg in EncoderDecoderMorphLearner._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        self.morph_feature_map = md.pop('morph_feature_map')
        self.vocab = md.pop('vocab')
        for kwarg in EncoderDecoderMorphLearner._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

    def __getstate__(self):
        return self._pack_metadata()

    def __setstate__(self, state):
        self._unpack_metadata(state)
        self._initialize_session()
        self._initialize_metadata()






    ############################################################
    # Private model construction methods
    ############################################################

    def build(self, n_train, outdir=None, restore=True, verbose=True):
        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './encoder_decoder_morph_model/'
        else:
            self.outdir = outdir

        self._initialize_inputs()
        with tf.variable_scope('encoder'):
            self.encoder = self._initialize_encoder(self.string_feats)
        self._initialize_morph_filters()
        self.morph_label_logits, self.morph_label_probs, self.morph_label = self._initialize_classifier(self.encoder)
        with tf.variable_scope('decoder'):
            self.decoder = self._initialize_decoder(self.morph_label_probs, self.n_timesteps_output)
        self._initialize_objective(n_train)
        self._initialize_saver()
        self._initialize_logging()

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.report_uninitialized = tf.report_uninitialized_variables(
                    var_list=None
                )
        self.load(restore=restore)

    def _initialize_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.string_feats = tf.placeholder(dtype=self.FLOAT_TF, shape=[None, self.n_timestamps, self.vocab_size])
                self.string_feats_mask = tf.placeholder(dtype=self.FLOAT_TF, shape=[None, self.n_timestamps])

                morph_feats = {}
                for f in self.morph_feature_map:
                    n_vals = len(self.morph_feature_map[f])
                    zero = tf.zeros([tf.shape(self.string_feats)[0], n_vals], dtype=self.FLOAT_TF)
                    morph_feats[f] = tf.placeholder_with_default(zero, shape=[None, n_vals], name=sn(f))
                self.morph_feats = morph_feats

                self.global_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_step'
                )
                self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
                self.global_batch_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_batch_step'
                )
                self.incr_global_batch_step = tf.assign(self.global_batch_step, self.global_batch_step + 1)

                self.training_batch_norm = tf.placeholder(tf.bool, name='training_batch_norm')
                self.training_dropout = tf.placeholder(tf.bool, name='training_dropout')

    def _initialize_morph_filters(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                morph_dimension_filter = {}
                morph_value_filter = {}

                for f in self.morph_feature_map:
                    dim_filter = tf.Variable(tf.zeros([]), dtype=self.FLOAT_TF)
                    dim_filter = tf.sigmoid(dim_filter)
                    morph_dimension_filter[f] = dim_filter

                    n_vals = self.morph_feats[f].shape[-1]
                    value_filter = tf.Variable(tf.zeros([1, n_vals]), dtype=self.FLOAT_TF)
                    value_filter = tf.sigmoid(value_filter)
                    morph_value_filter[f] = value_filter

                self.morph_dimension_filter = morph_dimension_filter
                self.morph_value_filter = morph_value_filter

    def _initialize_encoder(self, encoder_in):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.mask_padding:
                    mask = self.string_feats_mask
                else:
                    mask = None

                encoder = encoder_in
                if self.input_dropout_rate is not None:
                    encoder = tf.layers.dropout(
                        encoder,
                        self.input_dropout_rate,
                        noise_shape=[tf.shape(encoder)[0], tf.shape(encoder)[1], 1],
                        training=self.training_dropout
                    )

                if self.encoder_type.lower() in ['rnn', 'cnn_rnn']:
                    if self.encoder_type == 'cnn_rnn':
                        encoder = Conv1DLayer(
                            self.training_batch_norm,
                            self.conv_kernel_size,
                            n_filters=self.n_coef * (self.order + 1),
                            padding='same',
                            activation=tf.nn.elu,
                            batch_normalization_decay=self.encoder_batch_normalization_decay,
                            session=self.sess
                        )(encoder)

                    encoder = MultiRNNLayer(
                        units=self.units_encoder,
                        layers=self.n_layers_encoder,
                        activation=self.encoder_activation,
                        inner_activation=self.encoder_inner_activation,
                        recurrent_activation=self.encoder_recurrent_activation,
                        refeed_outputs=False,
                        return_sequences=False,
                        name='RNNEncoder',
                        session=self.sess
                    )(encoder, mask=mask)


                elif self.encoder_type.lower() == 'cnn':
                    encoder = Conv1DLayer(
                        self.training_batch_norm,
                        self.conv_kernel_size,
                        n_filters=self.n_coef * (self.order + 1),
                        padding='same',
                        activation=tf.nn.elu,
                        batch_normalization_decay=self.encoder_batch_normalization_decay,
                        session=self.sess
                    )(encoder)

                    for i in range(self.n_layers_encoder - 1):
                        if i > 0 and self.encoder_resnet_n_layers_inner:
                            encoder = Conv1DResidualLayer(
                                self.training_batch_norm,
                                self.conv_kernel_size,
                                n_filters=self.units_encoder[i],
                                padding='causal',
                                layers_inner=self.encoder_resnet_n_layers_inner,
                                activation=self.encoder_inner_activation,
                                activation_inner=self.encoder_inner_activation,
                                batch_normalization_decay=self.batch_normalization_decay,
                                session=self.sess
                            )(encoder)
                        else:
                            encoder = Conv1DLayer(
                                self.training_batch_norm,
                                self.conv_kernel_size,
                                n_filters=self.units_encoder[i],
                                padding='causal',
                                activation=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess
                            )(encoder)

                    encoder = DenseLayer(
                        self.training_batch_norm,
                        units=self.units_encoder[-1],
                        activation=self.encoder_activation,
                        batch_normalization_decay=None,
                        session=self.sess
                    )(tf.layers.Flatten()(encoder))

                elif self.encoder_type.lower() == 'dense':
                    encoder = tf.layers.Flatten()(encoder)

                    for i in range(self.n_layers_encoder - 1):
                        if i > 0 and self.encoder_resnet_n_layers_inner:
                            encoder = DenseResidualLayer(
                                self.training_batch_norm,
                                units=self.n_timesteps_input * self.units_encoder[i],
                                layers_inner=self.encoder_resnet_n_layers_inner,
                                activation=self.encoder_inner_activation,
                                activation_inner=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess
                            )(encoder)
                        else:
                            encoder = DenseLayer(
                                self.training_batch_norm,
                                units=self.n_timesteps_input * self.units_encoder[i],
                                activation=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess
                            )(encoder)

                    encoder = DenseLayer(
                        self.training_batch_norm,
                        units=self.units_encoder[-1],
                        activation=self.encoder_activation,
                        batch_normalization_decay=None,
                        session=self.sess
                    )(encoder)

                else:
                    raise ValueError('Encoder type "%s" is not currently supported' %self.encoder_type)

                return encoder

    def _initialize_classifier(self, classifier_in):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                morph_label_logits = {}
                morph_label_probs = {}
                morph_labels = {}
                i = 0
                for f in sorted(list(self.morph_feature_map.keys())):
                    n_val = len(self.morph_feature_map[f])
                    logits = classifier_in[:,i:i+n_val]
                    morph_label_logits[f] = logits
                    label_probs = tf.nn.softmax(logits)
                    morph_label_probs[f] = label_probs
                    label = tf.argmax(logits)
                    morph_labels[f] = label
                    i += n_val

                return morph_label_logits, morph_label_probs, morph_labels

    def _initialize_decoder(self, decoder_in, n_timesteps):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                decoder = {}
                for f in decoder_in:
                    decoder_in_new = decoder_in[f]
                    decoder_in_new *= self.morph_dimension_filter[f] * self.morph_value_filter[f]
                    decoder[f] = decoder_in_new
                decoder = tf.concat([decoder[f] for f in decoder], axis=1)

                if self.mask_padding:
                    mask = self.string_feats_mask
                else:
                    mask = None

                if self.decoder_type.lower() in ['rnn', 'cnn_rnn']:
                    tile_dims = [1] * (len(decoder.shape) + 1)
                    tile_dims[-2] = n_timesteps

                    decoder = tf.tile(
                        decoder[..., None, :],
                        tile_dims
                    )

                    decoder = MultiRNNLayer(
                        units=self.units_decoder + [self.vocab_size],
                        layers=self.n_layers_decoder,
                        activation=self.decoder_inner_activation,
                        inner_activation=self.decoder_inner_activation,
                        recurrent_activation=self.decoder_recurrent_activation,
                        refeed_outputs=self.n_layers_decoder > 1,
                        return_sequences=True,
                        name='RNNDecoder',
                        session=self.sess
                    )(decoder, mask=mask)

                    decoder = DenseLayer(
                        self.training_batch_norm,
                        units=self.vocab_size,
                        activation=self.decoder_activation,
                        batch_normalization_decay=self.decoder_batch_normalization_decay,
                        session=self.sess
                    )(decoder)

                elif self.decoder_type.lower() == 'cnn':
                    assert n_timesteps is not None, 'n_timesteps must be defined when decoder_type == "cnn"'

                    decoder = DenseLayer(
                        self.training_batch_norm,
                        n_timesteps * self.units_decoder[0],
                        activation=tf.nn.elu,
                        batch_normalization_decay=self.decoder_batch_normalization_decay,
                        session=self.sess
                    )(decoder)

                    decoder_shape = tf.concat([tf.shape(decoder)[:-2], [n_timesteps, self.units_decoder[0]]], axis=0)
                    decoder = tf.reshape(decoder, decoder_shape)

                    for i in range(self.n_layers_decoder - 1):
                        if i > 0 and self.decoder_resnet_n_layers_inner:
                            decoder = Conv1DResidualLayer(
                                self.training_batch_norm,
                                self.conv_kernel_size,
                                n_filters=self.units_decoder[i],
                                padding='same',
                                layers_inner=self.decoder_resnet_n_layers_inner,
                                activation=self.decoder_inner_activation,
                                activation_inner=self.decoder_inner_activation,
                                batch_normalization_decay=self.decoder_batch_normalization_decay,
                                session=self.sess
                            )(decoder)
                        else:
                            decoder = Conv1DLayer(
                                self.training_batch_norm,
                                self.conv_kernel_size,
                                n_filters=self.units_decoder[i],
                                padding='same',
                                activation=self.decoder_inner_activation,
                                batch_normalization_decay=self.decoder_batch_normalization_decay,
                                session=self.sess
                            )(decoder)

                        self._regularize_correspondences(self.n_layers_encoder - i - 2, decoder)

                    decoder = DenseLayer(
                            self.training_batch_norm,
                            units=n_timesteps * self.vocab_size,
                            activation=self.decoder_inner_activation,
                            batch_normalization_decay=False,
                            session=self.sess
                    )(tf.layers.Flatten()(decoder))

                    decoder_shape = tf.concat([tf.shape(decoder)[:-2], [n_timesteps, self.vocab_size]], axis=0)
                    decoder = tf.reshape(decoder, decoder_shape)

                elif self.decoder_type.lower() == 'dense':
                    assert n_timesteps is not None, 'n_timesteps must be defined when decoder_type == "dense"'

                    for i in range(self.n_layers_decoder - 1):

                        in_shape_flattened, out_shape_unflattened = self._get_decoder_shapes(decoder, n_timesteps, self.units_decoder[i], expand_sequence=i==0)
                        decoder = tf.reshape(decoder, in_shape_flattened)

                        if i > 0 and self.decoder_resnet_n_layers_inner:
                            if self.units_decoder[i] != self.units_decoder[i-1]:
                                project_inputs = True
                            else:
                                project_inputs = False

                            decoder = DenseResidualLayer(
                                self.training_batch_norm,
                                units=n_timesteps * self.units_decoder[i],
                                layers_inner=self.decoder_resnet_n_layers_inner,
                                activation=self.decoder_inner_activation,
                                activation_inner=self.decoder_inner_activation,
                                project_inputs=project_inputs,
                                batch_normalization_decay=self.decoder_batch_normalization_decay,
                                session=self.sess
                            )(decoder)
                        else:
                            decoder = DenseLayer(
                                self.training_batch_norm,
                                units=n_timesteps * self.units_decoder[i],
                                activation=self.decoder_inner_activation,
                                batch_normalization_decay=self.decoder_batch_normalization_decay,
                                session=self.sess
                            )(decoder)

                        decoder = tf.reshape(decoder, out_shape_unflattened)

                        self._regularize_correspondences(self.n_layers_encoder - i - 2, decoder)

                    in_shape_flattened, out_shape_unflattened = self._get_decoder_shapes(decoder, n_timesteps, self.vocab_size)
                    decoder = tf.reshape(decoder, in_shape_flattened)

                    decoder = DenseLayer(
                        self.training_batch_norm,
                        units=n_timesteps * self.vocab_size,
                        activation=self.decoder_activation,
                        batch_normalization_decay=None,
                        session=self.sess
                    )(decoder)

                    decoder = tf.reshape(decoder, out_shape_unflattened)

                else:
                    raise ValueError('Decoder type "%s" is not currently supported' %self.decoder_type)

                return decoder

    def _initialize_objective(self, n_train):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                encoder_loss = 0.
                for f in self.morph_feature_map:
                    encoder_logits = self.morph_label_logits[f]
                    encoder_targets = self.morph_feats[f]
                    filter_weights = self.morph_value_filter[f] * self.morph_dimension_filter[f]
                    loss_new = tf.losses.sigmoid_cross_entropy(encoder_targets, encoder_logits, weights=filter_weights)
                    encoder_loss += loss_new

                decoder_logits = self.decoder
                decoder_loss = tf.losses.softmax_cross_entropy(self.string_feats, decoder_logits, weights=self.string_feats_mask)

                loss = encoder_loss + decoder_loss

                if len(self.regularizer_losses) > 0:
                    self.regularizer_loss_total = tf.add_n(self.regularizer_losses)
                    loss += self.regularizer_loss_total
                else:
                    self.regularizer_loss_total = tf.constant(0., dtype=self.FLOAT_TF)

                self.loss = loss
                self.optim = self._initialize_optimizer(self.optim_name)
                self.train_op = self.optim.minimize(self.loss, global_step=self.global_batch_step)

    def _initialize_optimizer(self, name):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                lr = tf.constant(self.learning_rate, dtype=self.FLOAT_TF)
                if name is None:
                    self.lr = lr
                    return None
                if self.lr_decay_family is not None:
                    lr_decay_steps = tf.constant(self.lr_decay_steps, dtype=self.INT_TF)
                    lr_decay_rate = tf.constant(self.lr_decay_rate, dtype=self.FLOAT_TF)
                    lr_decay_staircase = self.lr_decay_staircase
                    self.lr = getattr(tf.train, self.lr_decay_family)(
                        lr,
                        self.global_step,
                        lr_decay_steps,
                        lr_decay_rate,
                        staircase=lr_decay_staircase,
                        name='learning_rate'
                    )
                    if np.isfinite(self.learning_rate_min):
                        lr_min = tf.constant(self.learning_rate_min, dtype=self.FLOAT_TF)
                        INF_TF = tf.constant(np.inf, dtype=self.FLOAT_TF)
                        self.lr = tf.clip_by_value(self.lr, lr_min, INF_TF)
                else:
                    self.lr = lr

                clip = self.max_global_gradient_norm

                return {
                    'SGD': lambda x: self._clipped_optimizer_class(tf.train.GradientDescentOptimizer)(x, max_global_norm=clip) if clip else tf.train.GradientDescentOptimizer(x),
                    'Momentum': lambda x: self._clipped_optimizer_class(tf.train.MomentumOptimizer)(x, 0.9, max_global_norm=clip) if clip else tf.train.MomentumOptimizer(x, 0.9),
                    'AdaGrad': lambda x: self._clipped_optimizer_class(tf.train.AdagradOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdagradOptimizer(x),
                    'AdaDelta': lambda x: self._clipped_optimizer_class(tf.train.AdadeltaOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdadeltaOptimizer(x),
                    'Adam': lambda x: self._clipped_optimizer_class(tf.train.AdamOptimizer)(x, max_global_norm=clip) if clip else tf.train.AdamOptimizer(x),
                    'FTRL': lambda x: self._clipped_optimizer_class(tf.train.FtrlOptimizer)(x, max_global_norm=clip) if clip else tf.train.FtrlOptimizer(x),
                    'RMSProp': lambda x: self._clipped_optimizer_class(tf.train.RMSPropOptimizer)(x, max_global_norm=clip) if clip else tf.train.RMSPropOptimizer(x),
                    'Nadam': lambda x: self._clipped_optimizer_class(tf.contrib.opt.NadamOptimizer)(x, max_global_norm=clip) if clip else tf.contrib.opt.NadamOptimizer(x)
                }[name](self.lr)

    def _initialize_logging(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                pass

    def _initialize_saver(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.saver = tf.train.Saver()

                self.check_numerics_ops = [tf.check_numerics(v, 'Numerics check failed') for v in tf.trainable_variables()]



    ############################################################
    # Private utility methods
    ############################################################

    ## Thanks to Keisuke Fujii (https://github.com/blei-lab/edward/issues/708) for this idea
    def _clipped_optimizer_class(self, base_optimizer):
        class ClippedOptimizer(base_optimizer):
            def __init__(self, *args, max_global_norm=None, **kwargs):
                super(ClippedOptimizer, self).__init__( *args, **kwargs)
                self.max_global_norm = max_global_norm

            def compute_gradients(self, *args, **kwargs):
                grads_and_vars = super(ClippedOptimizer, self).compute_gradients(*args, **kwargs)
                if self.max_global_norm is None:
                    return grads_and_vars
                grads = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)[0]
                vars = [v for _, v in grads_and_vars]
                grads_and_vars = []
                for grad, var in zip(grads, vars):
                    grads_and_vars.append((grad, var))
                return grads_and_vars

            def apply_gradients(self, grads_and_vars, **kwargs):
                if self.max_global_norm is None:
                    return grads_and_vars
                grads = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)[0]
                vars = [v for _, v in grads_and_vars]
                grads_and_vars = []
                for grad, var in zip(grads, vars):
                    grads_and_vars.append((grad, var))

                return super(ClippedOptimizer, self).apply_gradients(grads_and_vars, **kwargs)

        return ClippedOptimizer

    def _regularize(self, var, regularizer):
        if regularizer is not None:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    reg = tf.contrib.layers.apply_regularization(regularizer, [var])
                    self.regularizer_losses.append(reg)

    def _get_decoder_shapes(self, decoder, n_timesteps, units, expand_sequence=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                decoder_in_shape = tf.shape(decoder)
                if expand_sequence:
                    decoder_in_shape_flattened = decoder_in_shape
                else:
                    feat = int(decoder.shape[-1])
                    decoder_in_shape_flattened = tf.concat([decoder_in_shape[:-2], [n_timesteps * feat]], axis=0)
                decoder_out_shape = tf.concat([decoder_in_shape_flattened[:-1], [n_timesteps, units]], axis=0)

                return decoder_in_shape_flattened, decoder_out_shape

    # Thanks to Ralph Mao (https://github.com/RalphMao) for this workaround
    def _restore_inner(self, path, predict=False, allow_missing=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                try:
                    if predict:
                        self.ema_saver.restore(self.sess, path)
                    else:
                        self.saver.restore(self.sess, path)
                except tf.errors.DataLossError:
                    sys.stderr.write('Read failure during load. Trying from backup...\n')
                    if predict:
                        self.ema_saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                    else:
                        self.saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                except tf.errors.NotFoundError as err:  # Model contains variables that are missing in checkpoint, special handling needed
                    if allow_missing:
                        reader = tf.train.NewCheckpointReader(path)
                        saved_shapes = reader.get_variable_to_shape_map()
                        model_var_names = sorted(
                            [(var.name, var.name.split(':')[0]) for var in tf.global_variables()])
                        ckpt_var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                                                 if var.name.split(':')[0] in saved_shapes])

                        model_var_names_set = set([x[1] for x in model_var_names])
                        ckpt_var_names_set = set([x[1] for x in ckpt_var_names])

                        missing_in_ckpt = model_var_names_set - ckpt_var_names_set
                        if len(missing_in_ckpt) > 0:
                            sys.stderr.write(
                                'Checkpoint file lacked the variables below. They will be left at their initializations.\n%s.\n\n' % (
                                    sorted(list(missing_in_ckpt))))
                        missing_in_model = ckpt_var_names_set - model_var_names_set
                        if len(missing_in_model) > 0:
                            sys.stderr.write(
                                'Checkpoint file contained the variables below which do not exist in the current model. They will be ignored.\n%s.\n\n' % (
                                    sorted(list(missing_in_ckpt))))

                        restore_vars = []
                        name2var = dict(
                            zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

                        with tf.variable_scope('', reuse=True):
                            for var_name, saved_var_name in ckpt_var_names:
                                curr_var = name2var[saved_var_name]
                                var_shape = curr_var.get_shape().as_list()
                                if var_shape == saved_shapes[saved_var_name]:
                                    restore_vars.append(curr_var)

                        if predict:
                            self.ema_map = {}
                            for v in restore_vars:
                                self.ema_map[self.ema.average_name(v)] = v
                            saver_tmp = tf.train.Saver(self.ema_map)
                        else:
                            saver_tmp = tf.train.Saver(restore_vars)

                        saver_tmp.restore(self.sess, path)
                    else:
                        raise err








    ############################################################
    # Public methods
    ############################################################

    def n_minibatch(self, n):
        return math.ceil(float(n) / self.minibatch_size)

    def minibatch_scale(self, n):
        return float(n) / self.minibatch_size

    def check_numerics(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                for op in self.check_numerics_ops:
                    self.sess.run(op)

    def run_train_step(self, feed_dict, return_losses=True, return_reconstructions=False, return_labels=False):
        return NotImplementedError

    def fit(
            self,
            train_data,
            cv_data=None,
            n_iter=None,
            n_plot=10,
            verbose=True
    ):
        if self.global_step.eval(session=self.sess) == 0:
            self.save()

        # TODO: Complete this

    def initialized(self):
        """
        Check whether model has been initialized.

        :return: ``bool``; whether the model has been initialized.
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                uninitialized = self.sess.run(self.report_uninitialized)
                if len(uninitialized) == 0:
                    return True
                else:
                    return False

    def save(self, dir=None):
        if dir is None:
            dir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                failed = True
                i = 0

                # Try/except to handle race conditions in Windows
                while failed and i < 10:
                    try:
                        self.saver.save(self.sess, dir + '/model.ckpt')
                        with open(dir + '/m.obj', 'wb') as f:
                            pickle.dump(self, f)
                        failed = False
                    except:
                        sys.stderr.write('Write failure during save. Retrying...\n')
                        time.sleep(1)
                        i += 1
                if i >= 10:
                    sys.stderr.write('Could not save model to checkpoint file. Saving to backup...\n')
                    self.saver.save(self.sess, dir + '/model_backup.ckpt')
                    with open(dir + '/m.obj', 'wb') as f:
                        pickle.dump(self, f)

    def load(self, outdir=None, predict=False, restore=True, allow_missing=True):
        """
        Load weights from a DNN-Seg checkpoint and/or initialize the DNN-Seg model.
        Missing weights in the checkpoint will be kept at their initializations, and unneeded weights in the checkpoint will be ignored.

        :param outdir: ``str``; directory in which to search for weights. If ``None``, use model defaults.
        :param predict: ``bool``; load EMA weights because the model is being used for prediction. If ``False`` load training weights.
        :param restore: ``bool``; restore weights from a checkpoint file if available, otherwise initialize the model. If ``False``, no weights will be loaded even if a checkpoint is found.
        :param allow_missing: ``bool``; load all weights found in the checkpoint file, allowing those that are missing to remain at their initializations. If ``False``, weights in checkpoint must exactly match those in the model graph, or else an error will be raised. Leaving set to ``True`` is helpful for backward compatibility, setting to ``False`` can be helpful for debugging.
        :return:
        """
        if outdir is None:
            outdir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not self.initialized():
                    self.sess.run(tf.global_variables_initializer())
                if restore and os.path.exists(outdir + '/checkpoint'):
                    self._restore_inner(outdir + '/model.ckpt', predict=predict, allow_missing=allow_missing)
                else:
                    if predict:
                        sys.stderr.write('No EMA checkpoint available. Leaving internal variables unchanged.\n')

    def set_predict_mode(self, mode):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not mode or self.ema_decay:
                    self.load(predict=mode)

    def report_settings(self, indent=0):
        out = ' ' * indent + 'MODEL SETTINGS:\n'
        out += ' ' * (indent + 2) + 'k: %s\n' %self.k
        for kwarg in ENCODER_DECODER_MORPH_LEARNER_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * (indent + 2) + '%s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        return out



