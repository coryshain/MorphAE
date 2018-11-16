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
from Levenshtein.StringMatcher import distance as lev_dist

from .kwargs import ENCODER_DECODER_MORPH_LEARNER_INITIALIZATION_KWARGS
from .util import sn, get_random_permutation

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


def initialize_embeddings(categories, dim, default=0., session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            categories = sorted(list(set(categories)))
            n_categories = len(categories)
            index_table = tf.contrib.lookup.index_table_from_tensor(
                tf.constant(categories),
                num_oov_buckets=1
            )
            embedding_matrix = tf.Variable(tf.fill([n_categories+1, dim], default))

            return index_table, embedding_matrix


@ops.RegisterGradient("BernoulliSample_ST")
def bernoulliSample_ST(op, grad):
    return [grad, tf.zeros(tf.shape(op.inputs[1]))]


def get_data_generator(data, char_to_ix, morph_to_ix, lex_to_ix, max_seq_len=25, randomize=True):
    n_data = len(data)
    i = 0
    if randomize:
        ix, ix_inv = get_random_permutation(n_data)
    else:
        ix = np.arange(n_data)

    preprocessing_function = get_data_preprocessing_function(
        char_to_ix,
        morph_to_ix,
        lex_to_ix,
        max_seq_len=max_seq_len
    )

    while True:
        if i >= n_data:
            i = 0
            if randomize:
                ix, ix_inv = get_random_permutation(n_data)

        lexeme, form, form_mask, morph_feat = preprocessing_function(data[ix[i]])
        i += 1

        yield lexeme, form, form_mask, morph_feat

def get_data_preprocessing_function(char_to_ix, morph_to_ix, lex_to_ix, max_seq_len=25):
    n_char = len(char_to_ix)
    n_morph = len(morph_to_ix)
    n_lex = len(lex_to_ix)

    def preprocessing_function(data_point):
        lexeme = data_point[0]
        lexeme_out = np.zeros((n_lex,))
        if lexeme in lex_to_ix:
            lex_ix = lex_to_ix[lexeme]
        else:
            lex_ix = -1
        lexeme_out[lex_ix] = 1

        form_str = data_point[1]
        offset = max_seq_len - len(form_str)
        form_out = np.zeros((max_seq_len, n_char))
        form_out[:,-1] = 1
        form_mask_out = np.zeros((max_seq_len,))
        for k, c in enumerate(form_str):
            form_out[k, -1] = 1
            form_out[k, char_to_ix[c]] = 1
        if offset > 0:
            form_mask_out[:-offset] = 1
        else:
            form_mask_out[:] = 1

        morph_feat_str = data_point[2]
        morph_feat_out = np.zeros((n_morph,))
        for m in morph_feat_str:
            morph_feat_out[morph_to_ix[m]] = 1

        return lexeme_out, form_out, form_mask_out, morph_feat_out

    return preprocessing_function


def reconstruct_characters(char_probs, char_set):
    out = []
    indices = np.argmax(char_probs, axis=-1)
    for w in indices:
        cur = ''
        for i in w:
            if char_set[i] is not None:
                cur += char_set[i]
        out.append(cur)

    return out


def reconstruct_morph_feats(morph_feat_probs, morph_set):
    out = []
    morph_feats_discrete = morph_feat_probs > 0.5
    for w in morph_feats_discrete:
        m_feats = []
        for j, p in enumerate(w):
            if p:
                m_feats.append(morph_set[j])
        out.append(';'.join(m_feats))

    return out


def stringify_data(form_gold, form_pred, morph_gold, morph_pred, char_set=None, morph_set=None):
    if not isinstance(form_gold, list):
        assert char_set, 'If gold forms are given as one-hot, char_set must be provided.'
        form_gold =  reconstruct_characters(form_gold, char_set)

    if not isinstance(form_pred, list):
        assert char_set, 'If predicted forms are given as one-hot, char_set must be provided.'
        form_pred = reconstruct_characters(form_pred, char_set)

    if not isinstance(morph_gold, list):
        assert char_set, 'If gold morphs are given as multi-hot, morph_set must be provided.'
        morph_gold = reconstruct_morph_feats(morph_gold, morph_set)

    if not isinstance(morph_pred, list):
        assert char_set, 'If predicted morphs are given as multi-hot, morph_set must be provided.'
        morph_pred = reconstruct_morph_feats(morph_pred, morph_set)

    max_len_reconst = 0
    max_len_morph = 0
    for x in form_gold:
        max_len_reconst = max(len(x), max_len_reconst)
    for x in form_pred:
        max_len_reconst = max(len(x), max_len_reconst)
    for x in morph_gold:
        max_len_morph = max(len(x), max_len_morph)
    for x in morph_pred:
        max_len_morph = max(len(x), max_len_morph)

    out_str = ''
    for i in range(len(form_gold)):
        out_str += '    GOLD: %s | %s\n' %(morph_gold[i] + ' ' * (max_len_morph - len(morph_gold[i])), form_gold[i] + ' ' * (max_len_reconst - len(form_gold[i])))
        out_str += '    PRED: %s | %s\n\n' %(morph_pred[i] + ' ' * (max_len_morph - len(morph_pred[i])), form_pred[i] + ' ' * (max_len_reconst - len(form_pred[i])))

    return out_str


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
            refeed_discretized_outputs=False,
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

                self._refeed_discretized_outputs = refeed_discretized_outputs

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
                    if self._refeed_discretized_outputs and l == 0:
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

                # if self._refeed_outputs and l == 0 and self._num_layers > 1:
                if self._refeed_discretized_outputs:
                    prev_state_in = tf.argmax(state[-1][1], axis=-1)
                    prev_state_in = tf.one_hot(prev_state_in, state[-1][1].shape[-1])
                    layer_inputs.append(prev_state_in)

                layer_inputs = tf.concat(layer_inputs, axis=1)

                # Compute gate pre-activations
                s = tf.matmul(
                    layer_inputs,
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
            refeed_discretized_outputs = False,
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
        self.refeed_discretized_outputs = refeed_discretized_outputs
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
                        refeed_discretized_outputs=self.refeed_discretized_outputs,
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

    def __init__(self, morph_set, lex_set, char_set, **kwargs):

        self.morph_set = sorted(list(set(morph_set)))
        self.lex_set = sorted(list(set(lex_set)))
        self.char_set = sorted(list(set(char_set)))
        for kwarg in EncoderDecoderMorphLearner._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

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

        self.ix_to_morph = self.morph_set
        self.morph_to_ix = {}
        for i, m in enumerate(self.ix_to_morph):
            self.morph_to_ix[m] = i
        self.morph_set_size = len(self.ix_to_morph)

        self.ix_to_lex = self.lex_set + [None]
        self.lex_to_ix = {}
        for i, l in enumerate(self.ix_to_lex):
            self.lex_to_ix[l] = i
        self.lex_set_size = len(self.ix_to_lex)

        self.ix_to_char = self.char_set + [None]
        self.char_to_ix = {}
        for i, c in enumerate(self.ix_to_char):
            self.char_to_ix[c] = i
        self.char_set_size = len(self.ix_to_char)

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

    def _pack_metadata(self):
        md = {
            'morph_set': self.morph_set,
            'char_set': self.char_set,
            'lex_set': self.lex_set,
        }
        for kwarg in EncoderDecoderMorphLearner._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        self.morph_set = md.pop('morph_set')
        self.char_set = md.pop('char_set')
        self.lex_set = md.pop('lex_set')
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

    def build(self, outdir=None, restore=True, verbose=True):
        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './encoder_decoder_morph_model/'
        else:
            self.outdir = outdir

        self._initialize_inputs()
        with tf.variable_scope('encoder'):
            self.encoder = self._initialize_encoder(self.forms)
        self.lex_classifier, self.morph_classifier_logits, self.morph_classifier_probs, self.morph_classifier = self._initialize_classifier(self.encoder)
        with self.sess.as_default():
            with self.sess.graph.as_default():
                decoder_in_lex = tf.cond(self.use_gold_lex, lambda: self.lex_embeddings, lambda: self.lex_classifier)
                decoder_in_morph = tf.cond(self.use_gold_morph, lambda: self.morph_feats, lambda: self.morph_classifier)
        with tf.variable_scope('decoder'):
            if self.decoder_type in ['rnn', 'cnn_rnn'] and False:
                n_timesteps_output = self.n_timesteps_output
            else:
                n_timesteps_output = self.n_timesteps
            self.morph_classifier_filtered = decoder_in_morph * self.morph_filter
            self.decoder_logits = self._initialize_decoder(decoder_in_lex, self.morph_classifier_filtered, n_timesteps_output)
        self._initialize_objective()
        self._initialize_saver()
        self._initialize_logging()

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.report_uninitialized = tf.report_uninitialized_variables(
                    var_list=None
                )
                self.decoder = tf.nn.softmax(self.decoder_logits)

                # Reverse lexeme lookup by encoding similarity
                lex_encodings = self.lex_classifier
                lex_encodings /= (tf.norm(lex_encodings, axis=-1, keepdims=True) + self.epsilon)

                lex_embeddings = self.lex_embedding_matrix
                lex_embeddings /= (tf.norm(lex_embeddings, axis=-1, keepdims=True) + self.epsilon)

                cos_sim = tf.tensordot(lex_encodings, tf.transpose(lex_embeddings, perm=[1, 0]), axes=1)
                self.lexeme_reverse_lookup = tf.argmax(cos_sim, axis=-1)

        self.load(restore=restore)

    def _initialize_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():

                # Counters
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

                # Boolean settings
                self.training_batch_norm = tf.placeholder_with_default(tf.constant(True, dtype=tf.bool), shape=[], name='training_batch_norm')
                self.training_dropout = tf.placeholder_with_default(tf.constant(True, dtype=tf.bool), shape=[], name='training_dropout')
                self.use_gold_lex = tf.placeholder_with_default(tf.constant(True, dtype=tf.bool), shape=[], name='use_gold_lex')
                self.use_gold_morph = tf.placeholder_with_default(tf.constant(True, dtype=tf.bool), shape=[], name='use_gold_morph')
                self.sample_discrete = tf.placeholder_with_default(tf.constant(True, dtype=tf.bool), shape=[], name='sample_discrete')

                # Inputs
                self.forms = tf.placeholder(dtype=self.FLOAT_TF, shape=[None, self.n_timesteps, self.char_set_size], name='forms')
                one = tf.ones([tf.shape(self.forms)[0], self.n_timesteps], dtype=self.FLOAT_TF)
                self.forms_mask = tf.placeholder_with_default(one, shape=[None, self.n_timesteps], name='forms')
                self.n_timesteps_output = tf.shape(self.forms)[1]

                zero = tf.zeros([tf.shape(self.forms)[0], self.morph_set_size], dtype=self.FLOAT_TF)
                self.morph_feats = tf.placeholder_with_default(zero, shape=[None, self.morph_set_size], name='morph_feats')

                # Filter
                morph_filter_logits = tf.Variable(tf.zeros([1, self.morph_set_size]), dtype=self.FLOAT_TF, name='morph_filter')
                if self.discretize_filter:
                    if self.slope_annealing_rate:
                        rate = self.slope_annealing_rate
                        slope_coef = tf.minimum(10., 1 + rate * tf.cast(self.global_step, dtype=tf.float32))
                        morph_filter_logits *= slope_coef
                morph_filter_probs = tf.sigmoid(morph_filter_logits)
                morph_filter = morph_filter_probs
                if self.discretize_filter:
                    sample_fn = lambda: bernoulli_straight_through(morph_filter, session=self.sess)
                    round_fn = lambda: round_straight_through(morph_filter, session=self.sess)
                    morph_filter = tf.cond(self.sample_discrete, sample_fn, round_fn)
                self.morph_filter_logits = morph_filter_logits
                self.morph_filter_probs = morph_filter_probs
                self.morph_filter = morph_filter

                # Lexical embeddings
                self.lex_embedding_matrix = tf.Variable(tf.fill([self.lex_set_size, self.lex_emb_dim], tf.constant(0., dtype=self.FLOAT_TF)), name='lex_embedding_matrix')
                zero = tf.zeros([tf.shape(self.forms)[0], self.lex_set_size], dtype=self.FLOAT_TF)
                self.lex_feats = tf.placeholder_with_default(zero, shape=[None, self.lex_set_size], name='lex_feats')

                self.lex_embeddings = tf.matmul(
                    self.lex_feats,
                    self.lex_embedding_matrix
                )

                # Logging placeholders
                self.loss_summary = tf.placeholder(tf.float32, name='loss_summary')
                self.accuracy_summary = tf.placeholder(tf.float32, name='accuracy_summary')
                self.levenshtein_summary = tf.placeholder(tf.float32, name='levenshtein_summary')

    def _initialize_encoder(self, encoder_in):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.mask_padding:
                    mask = self.forms_mask
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

                    for l in range(self.n_layers_encoder):
                        encoder = RNNLayer(
                            units=self.units_encoder[l],
                            activation=self.encoder_inner_activation,
                            recurrent_activation = self.encoder_recurrent_activation,
                            return_sequences=False,
                            name='RNNEncoder%s' %l,
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
                        batch_normalization_decay=self.encoder_batch_normalization_decay,
                        session=self.sess
                    )(tf.layers.Flatten()(encoder))

                elif self.encoder_type.lower() == 'dense':
                    encoder = tf.layers.Flatten()(encoder)

                    for i in range(self.n_layers_encoder - 1):
                        if i > 0 and self.encoder_resnet_n_layers_inner:
                            encoder = DenseResidualLayer(
                                self.training_batch_norm,
                                units=self.n_timesteps * self.units_encoder[i],
                                layers_inner=self.encoder_resnet_n_layers_inner,
                                activation=self.encoder_inner_activation,
                                activation_inner=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess
                            )(encoder)
                        else:
                            encoder = DenseLayer(
                                self.training_batch_norm,
                                units=self.n_timesteps * self.units_encoder[i],
                                activation=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess
                            )(encoder)

                    encoder = DenseLayer(
                        self.training_batch_norm,
                        units=self.units_encoder[-1],
                        activation=self.encoder_activation,
                        batch_normalization_decay=self.encoder_batch_normalization_decay,
                        session=self.sess
                    )(encoder)

                else:
                    raise ValueError('Encoder type "%s" is not currently supported' %self.encoder_type)

                return encoder

    def _initialize_classifier(self, classifier_in):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                lex_classifier = DenseLayer(
                    self.training_batch_norm,
                    units=self.lex_emb_dim,
                    activation=self.encoder_activation,
                    batch_normalization_decay=None,
                    session=self.sess
                )(classifier_in)

                morph_classifier_logits = DenseLayer(
                    self.training_batch_norm,
                    units=self.morph_set_size,
                    activation=self.encoder_activation,
                    batch_normalization_decay=None,
                    session=self.sess
                )(classifier_in)

                if self.discretize_morph_encoder:
                    if self.slope_annealing_rate:
                        rate = self.slope_annealing_rate
                        slope_coef = tf.minimum(10., 1 + rate * tf.cast(self.global_step, dtype=tf.float32))
                        morph_classifier_logits *= slope_coef
                # morph_classifier_probs = tf.keras.backend.hard_sigmoid(morph_classifier_logits)
                morph_classifier_probs = tf.sigmoid(morph_classifier_logits)
                if self.discretize_morph_encoder:
                    sample_fn = lambda: bernoulli_straight_through(morph_classifier_probs, session=self.sess)
                    round_fn = lambda: round_straight_through(morph_classifier_probs, session=self.sess)
                    morph_classifier = tf.cond(self.sample_discrete, sample_fn, round_fn)
                else:
                    morph_classifier = morph_classifier_probs

                return lex_classifier, morph_classifier_logits, morph_classifier_probs, morph_classifier

    def _initialize_decoder(self, lex_classifier, morph_classifier, n_timesteps):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                decoder = tf.concat([lex_classifier, morph_classifier], axis=1)

                # if self.mask_padding:
                #     mask = self.forms_mask
                # else:
                #     mask = None
                mask = None

                units_decoder = self.units_decoder + [self.char_set_size]

                if self.decoder_type.lower() in ['rnn', 'cnn_rnn']:
                    tile_dims = [1] * (len(decoder.shape) + 1)
                    tile_dims[-2] = n_timesteps

                    decoder = tf.tile(
                        decoder[..., None, :],
                        tile_dims
                    )
                    time_ix = tf.range(n_timesteps)[None,:]
                    time_feat = tf.one_hot(time_ix, n_timesteps)
                    time_feat = tf.tile(
                        time_feat,
                        [tf.shape(decoder)[0], 1, 1]
                    )

                    decoder = tf.concat(
                        [decoder, time_feat],
                        axis=2
                    )

                    # in_shape_flattened, out_shape_unflattened = self._get_decoder_shapes(
                    #     decoder,
                    #     n_timesteps,
                    #     units_decoder[0],
                    #     expand_sequence=True
                    # )
                    # decoder = DenseLayer(
                    #     self.training_batch_norm,
                    #     units=n_timesteps * units_decoder[0],
                    #     activation=tf.nn.elu,
                    #     batch_normalization_decay=self.decoder_batch_normalization_decay,
                    #     session=self.sess
                    # )(decoder)
                    # decoder = tf.reshape(decoder, out_shape_unflattened)
                    # print(decoder.shape)


                    # for l in range(self.n_layers_decoder):
                    #     decoder = RNNLayer(
                    #         units=units_decoder[l],
                    #         activation=self.decoder_inner_activation,
                    #         recurrent_activation = self.decoder_recurrent_activation,
                    #         return_sequences=True,
                    #         name='RNNDecoder%s' %l,
                    #         session=self.sess
                    #     )(decoder, mask=mask)

                    decoder = MultiRNNLayer(
                        units=units_decoder,
                        layers=self.n_layers_decoder,
                        activation=self.decoder_activation,
                        inner_activation=self.decoder_inner_activation,
                        recurrent_activation=self.decoder_recurrent_activation,
                        refeed_discretized_outputs=True,
                        return_sequences=True,
                        name='RNNDecoder',
                        session=self.sess
                    )(decoder, mask=mask)

                    # decoder = RNNLayer(
                    #     units=self.char_set_size,
                    #     activation=self.decoder_activation,
                    #     recurrent_activation=self.decoder_recurrent_activation,
                    #     return_sequences=True,
                    #     name='RNNDecoderFinal',
                    #     session=self.sess
                    # )(decoder, mask=mask)

                    # decoder = DenseLayer(
                    #     self.training_batch_norm,
                    #     units=self.char_set_size,
                    #     activation=self.decoder_activation,
                    #     batch_normalization_decay=self.decoder_batch_normalization_decay,
                    #     session=self.sess
                    # )(decoder)

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

                    decoder = DenseLayer(
                            self.training_batch_norm,
                            units=n_timesteps * self.char_set_size,
                            activation=self.decoder_inner_activation,
                            batch_normalization_decay=False,
                            session=self.sess
                    )(tf.layers.Flatten()(decoder))

                    decoder_shape = tf.concat([tf.shape(decoder)[:-2], [n_timesteps, self.char_set_size]], axis=0)
                    decoder = tf.reshape(decoder, decoder_shape)

                elif self.decoder_type.lower() == 'dense':
                    assert n_timesteps is not None, 'n_timesteps must be defined when decoder_type == "dense"'

                    for i in range(self.n_layers_decoder - 1):

                        in_shape_flattened, out_shape_unflattened = self._get_decoder_shapes(decoder, n_timesteps, self.units_decoder[i], expand_sequence=i==0)
                        decoder = tf.reshape(decoder, in_shape_flattened)

                        if i > 0 and self.decoder_resnet_n_layers_inner:
                            if units_decoder[i] != units_decoder[i-1]:
                                project_inputs = True
                            else:
                                project_inputs = False

                            decoder = DenseResidualLayer(
                                self.training_batch_norm,
                                units=n_timesteps * units_decoder[i],
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

                    in_shape_flattened, out_shape_unflattened = self._get_decoder_shapes(decoder, n_timesteps, self.char_set_size)
                    decoder = tf.reshape(decoder, in_shape_flattened)

                    decoder = DenseLayer(
                        self.training_batch_norm,
                        units=n_timesteps * units_decoder[-1],
                        activation=self.decoder_activation,
                        batch_normalization_decay=None,
                        session=self.sess
                    )(decoder)

                    decoder = tf.reshape(decoder, out_shape_unflattened)

                else:
                    raise ValueError('Decoder type "%s" is not currently supported' %self.decoder_type)

                return decoder

    def _initialize_objective(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                morph_encoder_loss = tf.losses.sigmoid_cross_entropy(
                    self.morph_feats,
                    self.morph_classifier_logits,
                    weights=self.morph_filter
                )
                morph_encoder_loss *= self.morph_encoder_loss_scale
                # morph_encoder_loss = 0.

                lex_encoder_loss = tf.losses.mean_squared_error(
                    self.lex_embeddings,
                    self.lex_classifier
                )
                # lex_encoder_loss = tf.losses.softmax_cross_entropy(
                #     self.lex_feats,
                #     self.lex_classifier_logits
                # )
                # lex_encoder_loss = 0.

                encoder_loss = morph_encoder_loss + lex_encoder_loss

                decoder_targets = self.forms
                decoder_mask = self.forms_mask
                # decoder_targets = tf.Print(decoder_targets, [decoder_targets, tf.argmax(self.decoder_logits, axis=-1), decoder_mask], summarize=25)

                decoder_loss = tf.losses.softmax_cross_entropy(
                    decoder_targets,
                    self.decoder_logits,
                    # weights=decoder_mask
                )

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
                for i, m in enumerate(self.ix_to_morph):
                    name = sn(m) + '_filter_prob'
                    tf.summary.scalar('filter/' + name, self.morph_filter_probs[0,i], collections=['params'])
                self.summary_params = tf.summary.merge_all(key='params')

                tf.summary.scalar('training_loss', self.loss_summary, collections=['metrics'])
                tf.summary.scalar('accuracy', self.accuracy_summary, collections=['metrics'])
                tf.summary.scalar('levenshtein', self.levenshtein_summary, collections=['metrics'])


                if self.log_graph:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/edml', self.sess.graph)
                else:
                    self.writer = tf.summary.FileWriter(self.outdir + '/tensorboard/edml')
                self.summary_metrics = tf.summary.merge_all(key='metrics')

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

    def evaluate_reconstructions(self, cv_data, n_print=10):
        cv_data_generator = get_data_generator(
            cv_data,
            self.char_to_ix,
            self.morph_to_ix,
            self.lex_to_ix,
            randomize=False
        )

        n_eval = len(cv_data)
        if self.pad_seqs:
            if not np.isfinite(self.eval_minibatch_size):
                minibatch_size = n_eval
            else:
                minibatch_size = self.eval_minibatch_size
            n_minibatch = math.ceil(float(n_eval) / minibatch_size)
        else:
            minibatch_size = 1
            n_minibatch = n_eval

        perm, perm_inv = get_random_permutation(n_eval)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                fd_filter = {
                    self.training_dropout: False,
                    self.training_batch_norm: False,
                    self.sample_discrete: False
                }
                morph_filter = self.sess.run(self.morph_filter, feed_dict=fd_filter)[0]
                morph_filter_ix = np.rint(morph_filter)
                morph_filter_ix = np.where(morph_filter_ix)[0]
                morph_filter_str = []
                for ix in morph_filter_ix:
                    morph_filter_str.append(self.ix_to_morph[ix])

                for setting in ['encoder_in', 'gold_in']:
                    if setting == 'encoder_in':
                        eval_type = 'encoded'
                    else:
                        eval_type = 'gold'

                    sys.stderr.write('Extracting reconstruction evaluation data using %s decoder inputs...\n\n' % eval_type)

                    char_probs = []
                    morph_probs = []
                    forms = []
                    morph_feats = []

                    for i in range(0, n_eval, minibatch_size):
                        sys.stderr.write('\rMinibatch %d/%d' %((i/minibatch_size)+1, n_minibatch))
                        lexemes = []
                        forms_cur = []
                        forms_mask = []
                        morph_feats_cur = []
                        for j in range(min(minibatch_size, n_eval - i)):
                            lexeme_cur, form_cur, form_mask_cur, morph_feat_cur = next(cv_data_generator)
                            lexemes.append(lexeme_cur)
                            forms_cur.append(form_cur)
                            forms_mask.append(form_mask_cur)
                            morph_feats_cur.append(morph_feat_cur)
                        lexemes = np.stack(lexemes, axis=0)
                        forms_cur = np.stack(forms_cur, axis=0)
                        forms_mask = np.stack(forms_mask, axis=0)
                        morph_feats_cur = np.stack(morph_feats_cur, axis=0)

                        if setting == 'encoder_in':
                            use_gold = False
                        else:
                            use_gold = True

                        fd_minibatch = {
                            self.forms: forms_cur,
                            self.forms_mask: forms_mask,
                            self.lex_feats: lexemes,
                            self.morph_feats: morph_feats_cur,
                            self.training_dropout: False,
                            self.training_batch_norm: False,
                            self.use_gold_lex: use_gold,
                            self.use_gold_morph: use_gold,
                            self.sample_discrete: False
                        }

                        char_probs_cur, morph_probs_cur = self.sess.run(
                            [self.decoder, self.morph_classifier_filtered],
                            feed_dict=fd_minibatch
                        )

                        forms.append(forms_cur)
                        morph_feats.append(morph_feats_cur)
                        char_probs.append(char_probs_cur)
                        morph_probs.append(morph_probs_cur)

                    sys.stderr.write('\n\n')

                    char_probs = np.concatenate(char_probs, axis=0)
                    forms = np.concatenate(forms, axis=0)
                    morph_feats = np.concatenate(morph_feats, axis=0)
                    reconstructions_gold = reconstruct_characters(forms, self.ix_to_char)
                    reconstructions_pred = reconstruct_characters(char_probs, self.ix_to_char)

                    morph_probs = np.concatenate(morph_probs, axis=0)
                    morph_preds = morph_probs > 0.5

                    acc = 0
                    dist = 0
                    failure_ix = []

                    for i in range(len(reconstructions_gold)):
                        match = reconstructions_gold[i] == reconstructions_pred[i]
                        acc += match
                        dist += lev_dist(reconstructions_gold[i], reconstructions_pred[i])
                        if not match:
                            failure_ix.append(i)

                    failure_forms_gold = forms[failure_ix]
                    failure_forms_pred = char_probs[failure_ix]
                    failure_morphs_gold = morph_feats[failure_ix]
                    failure_morphs_pred = morph_preds[failure_ix]

                    failure_str = stringify_data(
                        failure_forms_gold,
                        failure_forms_pred,
                        failure_morphs_gold,
                        failure_morphs_pred,
                        char_set=self.ix_to_char,
                        morph_set=self.ix_to_morph
                    )

                    acc = float(acc) / n_eval
                    dist /= n_eval

                    sys.stderr.write('Reconstruction ealuation using %s lex/morph features:\n' %eval_type)
                    sys.stderr.write('  Exact match accuracy: %s\n  Mean Levenshtein distance: %s\n  Reconstruction examples:\n' %(acc, dist))

                    reconst_to_print_gold = forms[perm[:n_print]]
                    reconst_to_print_pred = char_probs[perm[:n_print]]
                    morph_to_print_gold = morph_feats[perm[:n_print]]
                    morph_to_print_pred = morph_preds[perm[:n_print]]

                    out_str = stringify_data(
                        reconst_to_print_gold,
                        reconst_to_print_pred,
                        morph_to_print_gold,
                        morph_to_print_pred,
                        char_set=self.ix_to_char,
                        morph_set=self.ix_to_morph
                    )

                    sys.stderr.write(out_str)

                sys.stderr.write('Active morphological features:\n')
                sys.stderr.write('  %s\n\n' %(';'.join(morph_filter_str)))

                return acc, dist, failure_str

    def create_reinflection_data(self, data):
        reinflection_map = {}
        for x in data:
            lexeme = x[0]
            form = x[1]
            morph_str = x[2]

            if lexeme not in reinflection_map:
                reinflection_map[lexeme] = {}

            morph_ix = []
            for m in morph_str:
                morph_ix.append(self.morph_to_ix[m])
            morph_feat_set = tuple(sorted(morph_ix))

            if morph_feat_set not in reinflection_map[lexeme]:
                reinflection_map[lexeme][morph_feat_set] = form

        input_data = []
        reinflection_targets = []

        for lexeme in sorted(list(reinflection_map.keys())):
            morph_tuples = sorted(list(reinflection_map[lexeme].keys()))
            for morph_1 in morph_tuples:
                input_form = reinflection_map[lexeme][morph_1]
                for morph_2 in morph_tuples:
                    target_form = reinflection_map[lexeme][morph_2]
                    morph_2_str = [self.ix_to_morph[m] for m in morph_2]
                    input_data.append((lexeme, input_form, morph_2_str))
                    reinflection_targets.append(target_form)

        return input_data, reinflection_targets

    def evaluate_reinflections(self, cv_data, n_print=10, n_eval=None):
        input_data, reinflection_targets = self.create_reinflection_data(cv_data)

        if not n_eval:
            n_eval = len(input_data)

        if self.pad_seqs:
            if not np.isfinite(self.eval_minibatch_size):
                minibatch_size = n_eval
            else:
                minibatch_size = self.eval_minibatch_size
            n_minibatch = math.ceil(float(n_eval) / minibatch_size)
        else:
            minibatch_size = 1
            n_minibatch = n_eval

        perm, perm_inv = get_random_permutation(n_eval)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                for setting in ['encoder_in', 'gold_in']:
                    input_data_generator = get_data_generator(
                        input_data,
                        self.char_to_ix,
                        self.morph_to_ix,
                        self.lex_to_ix,
                        randomize=False
                    )

                    if setting == 'encoder_in':
                        eval_type = 'encoded'
                    else:
                        eval_type = 'gold'

                    sys.stderr.write('Extracting reinflection evaluation data using %s decoder inputs...\n\n' % eval_type)

                    char_probs = []
                    morph_feats = []

                    for i in range(0, n_eval, minibatch_size):
                        sys.stderr.write('\rMinibatch %d/%d' %((i/minibatch_size)+1, n_minibatch))
                        sys.stderr.flush()
                        lexemes = []
                        forms_cur = []
                        forms_mask = []
                        morph_feats_cur = []
                        for j in range(min(minibatch_size, n_eval - i)):
                            lexeme_cur, form_cur, form_mask_cur, morph_feat_cur = next(input_data_generator)
                            lexemes.append(lexeme_cur)
                            forms_cur.append(form_cur)
                            forms_mask.append(form_mask_cur)
                            morph_feats_cur.append(morph_feat_cur)
                        lexemes = np.stack(lexemes, axis=0)
                        forms_cur = np.stack(forms_cur, axis=0)
                        forms_mask = np.stack(forms_mask, axis=0)
                        morph_feats_cur = np.stack(morph_feats_cur, axis=0)

                        if setting == 'encoder_in':
                            use_gold = False
                        else:
                            use_gold = True

                        fd_minibatch = {
                            self.forms: forms_cur,
                            self.forms_mask: forms_mask,
                            self.lex_feats: lexemes,
                            self.morph_feats: morph_feats_cur,
                            self.training_dropout: False,
                            self.training_batch_norm: False,
                            self.use_gold_lex: use_gold,
                            self.use_gold_morph: True,
                            self.sample_discrete: False
                        }

                        char_probs_cur = self.sess.run(
                            self.decoder,
                            feed_dict=fd_minibatch
                        )

                        char_probs.append(char_probs_cur)
                        morph_feats.append(morph_feats_cur)

                    sys.stderr.write('\n\n')

                    char_probs = np.concatenate(char_probs, axis=0)
                    morph_feats = np.concatenate(morph_feats, axis=0)

                    reinflections_gold = reinflection_targets[:n_eval]
                    reinflections_pred = reconstruct_characters(char_probs, self.ix_to_char)

                    acc = 0
                    dist = 0

                    for i in range(n_eval):
                        match = reinflections_gold[i] == reinflections_pred[i]
                        acc += match
                        dist += lev_dist(reinflections_gold[i], reinflections_pred[i])

                    acc = float(acc) / n_eval
                    dist /= n_eval


                    sys.stderr.write('Reinflection evaluation using %s lex/morph features:\n' %eval_type)
                    sys.stderr.write('  Exact match accuracy: %s\n  Mean Levenshtein distance: %s\n  Reconstruction examples:\n' %(acc, dist))

                    reinfl_to_print_gold = []
                    reinfl_to_print_pred = []
                    for i in range(n_print):
                        ix = perm[i]
                        reinfl_to_print_gold.append(reinflections_gold[ix])
                        reinfl_to_print_pred.append(reinflections_pred[ix])
                    morph_to_print_gold = morph_feats[perm[:n_print]]
                    morph_to_print_pred = [''] * n_print

                    out_str = stringify_data(
                        reinfl_to_print_gold,
                        reinfl_to_print_pred,
                        morph_to_print_gold,
                        morph_to_print_pred,
                        char_set=self.ix_to_char,
                        morph_set=self.ix_to_morph
                    )

                    sys.stderr.write(out_str)

    def fit(
            self,
            train_data,
            cv_data=None,
            n_iter=None,
            n_print=10,
            verbose=True
    ):
        if verbose:
            usingGPU = tf.test.is_gpu_available()
            sys.stderr.write('Using GPU: %s\n' % usingGPU)

        with self.sess.as_default():
            with self.sess.graph.as_default():
                if verbose:
                    self.report_settings()

                n_train = len(train_data)
                if self.pad_seqs:
                    if not np.isfinite(self.minibatch_size):
                        minibatch_size = n_train
                    else:
                        minibatch_size = self.minibatch_size
                    n_minibatch = math.ceil(float(n_train) / minibatch_size)
                else:
                    minibatch_size = 1
                    n_minibatch = n_train

                if self.global_step.eval(session=self.sess) == 0:
                    self.save()

                train_data_generator = get_data_generator(
                    train_data,
                    self.char_to_ix,
                    self.morph_to_ix,
                    self.lex_to_ix
                )

                while self.global_step.eval(session=self.sess) < n_iter:
                    if verbose:
                        t0_iter = time.time()
                        sys.stderr.write('-' * 50 + '\n')
                        sys.stderr.write('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                        sys.stderr.write('\n')
                        if self.optim_name is not None and self.lr_decay_family is not None:
                            sys.stderr.write('Learning rate: %s\n' % self.lr.eval(session=self.sess))

                    if verbose:
                        sys.stderr.write('Updating...\n')
                        pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                    loss_total = 0.

                    for i in range(0, n_train, self.minibatch_size):
                        lexemes = []
                        forms = []
                        forms_mask = []
                        morph_feats = []
                        for j in range(min(self.minibatch_size, n_train - i)):
                            lexeme_cur, form_cur, form_mask_cur, morph_feat_cur = next(train_data_generator)
                            lexemes.append(lexeme_cur)
                            forms.append(form_cur)
                            forms_mask.append(form_mask_cur)
                            morph_feats.append(morph_feat_cur)
                        lexemes = np.stack(lexemes, axis=0)
                        forms = np.stack(forms, axis=0)
                        forms_mask = np.stack(forms_mask, axis=0)
                        morph_feats = np.stack(morph_feats, axis=0)

                        # print(forms)
                        # print(forms.argmax(axis=-1))
                        # print(forms.shape)
                        # input()

                        fd_minibatch = {
                            self.forms: forms,
                            self.forms_mask: forms_mask,
                            self.lex_feats: lexemes,
                            self.morph_feats: morph_feats
                        }

                        _, loss_cur, reg_cur = self.sess.run([self.train_op, self.loss, self.regularizer_loss_total], feed_dict=fd_minibatch)

                        if self.ema_decay:
                            self.sess.run(self.ema_op)
                        if not np.isfinite(loss_cur):
                            loss_cur = 0
                        loss_total += loss_cur

                        self.sess.run(self.incr_global_batch_step)
                        if verbose:
                            pb.update((i / minibatch_size) + 1, values=[('loss', loss_cur), ('reg', reg_cur)])

                        self.check_numerics()

                    loss_total /= n_minibatch

                    self.sess.run(self.incr_global_step)

                    if self.save_freq > 0 and self.global_step.eval(session=self.sess) % self.save_freq == 0:
                        try:
                            self.check_numerics()
                            numerics_passed = True
                        except:
                            numerics_passed = False

                        if numerics_passed:
                            if verbose:
                                sys.stderr.write('Saving model...\n')

                            self.save()

                            acc, dist, failures = self.evaluate_reconstructions(cv_data, n_print=n_print)

                            fd_summary = {
                                self.loss_summary: loss_total,
                                self.accuracy_summary: acc,
                                self.levenshtein_summary: dist
                            }

                            summary_metrics = self.sess.run(self.summary_metrics, feed_dict=fd_summary)
                            self.writer.add_summary(summary_metrics, self.global_step.eval(session=self.sess))

                            summary_params = self.sess.run(self.summary_params)
                            self.writer.add_summary(summary_params, self.global_step.eval(session=self.sess))

                        else:
                            if verbose:
                                sys.stderr.write('Numerics check failed. Aborting save and reloading from previous checkpoint...\n')

                            self.load()

                    if verbose:
                        t1_iter = time.time()
                        sys.stderr.write('Iteration time: %.2fs\n' % (t1_iter - t0_iter))

                self.evaluate_reconstructions(cv_data, n_print=n_print)
                self.evaluate_reinflections(cv_data, n_print=n_print)

        self.finalize()


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
                    tf.tables_initializer().run()
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
        for kwarg in ENCODER_DECODER_MORPH_LEARNER_INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * (indent + 2) + '%s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        return out

    def process_string(self, input, reinflections=None):
        if reinflections is None:
            reinflections = []
        if [] not in reinflections:
            reinflections.insert(0, [])

        with self.sess.as_default():
            with self.sess.graph.as_default():
                n_char = len(self.char_to_ix)

                max_seq_len = self.n_timesteps
                form_str = input
                offset = max_seq_len - len(form_str)
                form = np.zeros((max_seq_len, n_char))
                form[:, -1] = 1
                form_mask = np.zeros((max_seq_len,))
                for k, c in enumerate(form_str):
                    form[k, -1] = 1
                    form[k, self.char_to_ix[c]] = 1
                if offset > 0:
                    form_mask[:-offset] = 1
                else:
                    form_mask[:] = 1

                form = form[None, ...]
                form_mask = form_mask[None, ...]

                fd = {
                    self.forms: form,
                    self.forms_mask: form_mask,
                    self.training_dropout: False,
                    self.training_batch_norm: False,
                    self.use_gold_lex: False,
                    self.use_gold_morph: False,
                    self.sample_discrete: False
                }

                char_probs, morph_feat_probs, lexeme_ix = self.sess.run(
                    [self.decoder, self.morph_classifier_filtered, self.lexeme_reverse_lookup],
                    feed_dict=fd
                )

                morph_feats_reinfl = []
                for r in reinflections:
                    try:
                        morph_feat_str = r
                        morph_feat_cur = np.zeros((self.morph_set_size,))
                        for m in morph_feat_str:
                            morph_feat_cur[self.morph_to_ix[m]] = 1
                        morph_feats_reinfl.append(morph_feat_cur)
                    except KeyError as e:
                        sys.stderr.write('Morph feature "%s" not found. Skipping reinflection...\n' %str(e))
                morph_feats_reinfl = np.stack(morph_feats_reinfl, axis=0)
                form = np.tile(form, [len(morph_feats_reinfl), 1, 1])
                form_mask = np.tile(form_mask, [len(morph_feats_reinfl), 1])

                fd = {
                    self.forms: form,
                    self.forms_mask: form_mask,
                    self.morph_feats: morph_feats_reinfl,
                    self.training_dropout: False,
                    self.training_batch_norm: False,
                    self.use_gold_lex: False,
                    self.use_gold_morph: True,
                    self.sample_discrete: False
                }

                char_probs_reinfl = self.sess.run(
                    self.decoder,
                    feed_dict=fd
                )

                char_probs = np.concatenate([char_probs, char_probs_reinfl])
                morph_feat_probs = np.concatenate([morph_feat_probs, morph_feats_reinfl], axis=0)

                reconst = reconstruct_characters(char_probs, self.ix_to_char)
                morph_feats = reconstruct_morph_feats(morph_feat_probs, self.ix_to_morph)
                lexeme = self.ix_to_lex[lexeme_ix[0]]

                return reconst, morph_feats, lexeme

    def finalize(self):
        """
        Close the EDML instance to prevent memory leaks.

        :return: ``None``
        """
        self.sess.close()







