from functools import cmp_to_key

class Kwarg(object):
    """
    Data structure for storing keyword arguments and their docstrings.

    :param key: ``str``; Key
    :param default_value: Any; Default value
    :param dtypes: ``list`` or ``class``; List of classes or single class. Members can also be specific required values, either ``None`` or values of type ``str``.
    :param descr: ``str``; Description of kwarg
    """

    def __init__(self, key, default_value, dtypes, descr, aliases=None):
        if aliases is None:
            aliases = []
        self.key = key
        self.default_value = default_value
        if not isinstance(dtypes, list):
            self.dtypes = [dtypes]
        else:
            self.dtypes = dtypes
        self.dtypes = sorted(self.dtypes, key=cmp_to_key(Kwarg.type_comparator))
        self.descr = descr
        self.aliases = aliases

    def dtypes_str(self):
        if len(self.dtypes) == 1:
            out = '``%s``' %self.get_type_name(self.dtypes[0])
        elif len(self.dtypes) == 2:
            out = '``%s`` or ``%s``' %(self.get_type_name(self.dtypes[0]), self.get_type_name(self.dtypes[1]))
        else:
            out = ', '.join(['``%s``' %self.get_type_name(x) for x in self.dtypes[:-1]]) + ' or ``%s``' %self.get_type_name(self.dtypes[-1])

        return out

    def get_type_name(self, x):
        if isinstance(x, type):
            return x.__name__
        if isinstance(x, str):
            return '"%s"' %x
        return str(x)

    def in_settings(self, settings):
        out = False
        if self.key in settings:
            out = True

        if not out:
            for alias in self.aliases:
                if alias in settings:
                    out = True
                    break

        return out

    def kwarg_from_config(self, settings):
        if len(self.dtypes) == 1:
            val = {
                str: settings.get,
                int: settings.getint,
                float: settings.getfloat,
                bool: settings.getboolean
            }[self.dtypes[0]](self.key, None)

            if val is None:
                for alias in self.aliases:
                    val = {
                        str: settings.get,
                        int: settings.getint,
                        float: settings.getfloat,
                        bool: settings.getboolean
                    }[self.dtypes[0]](alias, self.default_value)
                    if val is not None:
                        break

            if val is None:
                val = self.default_value

        else:
            from_settings = settings.get(self.key, None)
            if from_settings is None:
                for alias in self.aliases:
                    from_settings = settings.get(alias, None)
                    if from_settings is not None:
                        break

            if from_settings is None:
                val = self.default_value
            else:
                parsed = False
                for x in reversed(self.dtypes):
                    if x == None:
                        if from_settings == 'None':
                            val = None
                            parsed = True
                            break
                    elif isinstance(x, str):
                        if from_settings == x:
                            val = from_settings
                            parsed = True
                            break
                    else:
                        try:
                            val = x(from_settings)
                            parsed = True
                            break
                        except TypeError:
                            pass

                assert parsed, 'Invalid value "%s" received for %s' %(from_settings, self.key)

        return val



    @staticmethod
    def type_comparator(a, b):
        '''
        Types precede strings, which precede ``None``
        :param a: First element
        :param b: Second element
        :return: ``-1``, ``0``, or ``1``, depending on outcome of comparison
        '''
        if isinstance(a, type) and not isinstance(b, type):
            return -1
        elif not isinstance(a, type) and isinstance(b, type):
            return 1
        elif isinstance(a, str) and not isinstance(b, str):
            return -1
        elif isinstance(b, str) and not isinstance(a, str):
            return 1
        else:
            return 0





ENCODER_DECODER_MORPH_LEARNER_INITIALIZATION_KWARGS = [
    Kwarg(
        'outdir',
        './encoder_decoder_morph_learner_model/',
        str,
        "Path to output directory, where logs and model parameters are saved."
    ),
    Kwarg(
        'lex_emb_dim',
        256,
        int,
        "Dimensionality of lexical embedding space."
    ),
    Kwarg(
        'morph_encoder_loss_scale',
        1.,
        float,
        "Scale of morphological encoder loss."
    ),
    Kwarg(
        'decoder_input_attention_type',
        None,
        [str, None],
        "Attention type to use for selecting inputs (encoded vs. potentially-noisy gold) to the decoder. One of ['hard', 'soft', or 'None']."
    ),
    Kwarg(
        'discretize_filter',
        True,
        bool,
        "Whether to implement the morphological feature filter using binary stochastic neurons. If False, filter weights are continuous on [0,1]."
    ),
    Kwarg(
        'discretize_morph_encoder',
        True,
        bool,
        "Whether to implement the morphological encoder using binary stochastic neurons. If False, morphological encodings are continuous on [0,1]."
    ),
    Kwarg(
        'discretize_lex_encoder',
        False,
        bool,
        "Whether to implement the lexical encoder using binary stochastic neurons. If False, lexical encodings are unconstrained."
    ),
    Kwarg(
        'encoder_type',
        'rnn',
        str,
        "Encoder network to use. One of ``dense``, ``cnn``, or ``rnn``."
    ),
    Kwarg(
        'n_layers_encoder',
        1,
        int,
        "Number of layers to use for encoder. Ignored if **encoder_type** is not ``dense``."
    ),
    Kwarg(
        'n_units_encoder',
        128,
        [int, str],
        "Number of units to use in encoder layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_encoder** - 1 space-delimited integers, one for each layer in order from bottom to top."
    ),
    Kwarg(
        'encoder_activation',
        'elu',
        [str, None],
        "Name of activation to use at the output of the encoder",
    ),
    Kwarg(
        'encoder_inner_activation',
        'tanh',
        [str, None],
        "Name of activation to use for any internal layers of the encoder",
        aliases=['inner_activation']
    ),
    Kwarg(
        'encoder_recurrent_activation',
        'sigmoid',
        [str, None],
        "Name of activation to use for recurrent activation in recurrent layers of the encoder. Ignored if encoder is not recurrent.",
        aliases=['recurrent_activation']
    ),
    Kwarg(
        'encoder_weight_regularization',
        None,
        [float, None],
        "Scale of L2 regularization to apply to all encoder weights and biases. If ``None``, no weight regularization."
    ),
    Kwarg(
        'encoder_weight_normalization',
        False,
        bool,
        "Apply weight normalization to encoder. Ignored unless encoder is recurrent."
    ),
    Kwarg(
        'encoder_layer_normalization',
        False,
        bool,
        "Apply layer normalization to encoder. Ignored unless encoder is recurrent."
    ),
    Kwarg(
        'decoder_type',
        'rnn',
        str,
        "Decoder network to use. One of ``dense``, ``cnn``, or ``rnn``."
    ),
    Kwarg(
        'n_units_decoder',
        128,
        [int, str],
        "Number of units to use in non-final decoder layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_decoder** - 1 space-delimited integers, one for each layer in order from top to bottom."
    ),
    Kwarg(
        'n_layers_decoder',
        1,
        int,
        "Number of layers to use for decoder. Ignored if **decoder_type** is not ``dense``."
    ),
    Kwarg(
        'decoder_activation',
        'tanh',
        [str, None],
        "Name of activation to use at the output of the decoder"
    ),
    Kwarg(
        'decoder_inner_activation',
        'tanh',
        [str, None],
        "Name of activation to use for any internal layers of the decoder",
        aliases=['inner_activation']
    ),
    Kwarg(
        'decoder_recurrent_activation',
        'sigmoid',
        [str, None],
        "Name of activation to use for recurrent activation in recurrent layers of the decoder. Ignored if decoder is not recurrent.",
        aliases=['recurrent_activation']
    ),
    Kwarg(
        'n_timesteps',
        None,
        int,
        "Number of timesteps in the input. If ``None``, dynamic number of timesteps."
    ),
    Kwarg(
        'conv_kernel_size',
        3,
        int,
        "Size of kernel to use in convolutional layers. Ignored if no residual layers in the model."
    ),
    Kwarg(
        'encoder_resnet_n_layers_inner',
        None,
        [int, None],
        "Implement internal encoder layers as residual layers with **resnet_n_layers_inner** internal layers each. If ``None``, do not use residual layers.",
        aliases=['resnet_n_layers_inner']
    ),
    Kwarg(
        'decoder_resnet_n_layers_inner',
        None,
        [int, None],
        "Implement internal decode layers as residual layers with **resnet_n_layers_inner** internal layers each. If ``None``, do not use residual layers.",
        aliases=['resnet_n_layers_inner']
    ),
    Kwarg(
        'encoder_batch_normalization_decay',
        None,
        [float, None],
        "Decay rate to use for batch normalization in internal encoder layers. If ``None``, no batch normalization.",
        aliases=['batch_normalization_decay']
    ),
    Kwarg(
        'decoder_batch_normalization_decay',
        None,
        [float, None],
        "Decay rate to use for batch normalization in internal decoder layers. If ``None``, no batch normalization.",
        aliases=['batch_normalization_decay']
    ),
    Kwarg(
        'slope_annealing_rate',
        None,
        [float, None],
        "Whether to anneal the slopes of the boundary activations."
    ),
    Kwarg(
        'slope_annealing_max',
        10.,
        [float, None],
        "Maximum allowed value of the slope annealing coefficient. If ``None``, no maximum will be enforced."
    ),
    Kwarg(
        'slope_annealing_iteration_power',
        1,
        float,
        "Power to which the iteration number ``t`` should be raised when computing annealed slopes for binary stochastic neurons."
    ),
    Kwarg(
        'additive_morph_noise_level',
        None,
        [float, None],
        "Amount of additive noise (incorrect features being turned on) to add to the morph features. Must be in [0,1]."
    ),
    Kwarg(
        'pad_seqs',
        True,
        bool,
        "Whether to pad inputs and targets out to a fixed temporal dimensionality. Necessary in order to use minibatches larger than 1. If ``True``, sequences are padded and submitted to the network as batch arrays. If ``False``, sequences are not padded and are submitted to the network as minibatches of 1."
    ),
    Kwarg(
        'max_len',
        None,
        [int, None],
        "Maximum sequence length. If ``None``, no maximum length imposed."
    ),
    Kwarg(
        'mask_padding',
        True,
        bool,
        "Mask padding frames in reconstruction targets so that they are ignored in gradient updates."
    ),
    Kwarg(
        'n_iter',
        10000,
        int,
        "Expected number of training iterations. Used only for logging purposes."
    ),
    Kwarg(
        'optim_name',
        'Nadam',
        [str, None],
        """Name of the optimizer to use. Must be one of:
        
            - ``'SGD'``
            - ``'Momentum'``
            - ``'AdaGrad'``
            - ``'AdaDelta'``
            - ``'Adam'``
            - ``'FTRL'``
            - ``'RMSProp'``
            - ``'Nadam'``
            - ``None`` (DTSRBayes only; uses the default optimizer defined by Edward, which currently includes steep learning rate decay and is therefore not recommended in the general case)"""
    ),
    Kwarg(
        'epsilon',
        1e-8,
        float,
        "Epsilon to avoid boundary violations."
    ),
    Kwarg(
        'optim_epsilon',
        1e-8,
        float,
        "Epsilon parameter to use if **optim_name** in ``['Adam', 'Nadam']``, ignored otherwise."
    ),
    Kwarg(
        'learning_rate',
        0.001,
        float,
        "Initial value for the learning rate."
    ),
    Kwarg(
        'learning_rate_min',
        0.,
        float,
        "Minimum value for the learning rate."
    ),
    Kwarg(
        'lr_decay_family',
        None,
        [str, None],
        "Functional family for the learning rate decay schedule (no decay if ``None``)."
    ),
    Kwarg(
        'lr_decay_rate',
        1.,
        float,
        "coefficient by which to decay the learning rate every ``lr_decay_steps`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_steps',
        1,
        int,
        "Span of iterations over which to decay the learning rate by ``lr_decay_rate`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_staircase',
        False,
        bool,
        "Keep learning rate flat between ``lr_decay_steps`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_iteration_power',
        1,
        float,
        "Power to which the iteration number ``t`` should be raised when computing the learning rate decay."
    ),
    Kwarg(
        'max_global_gradient_norm',
        None,
        [float, None],
        'Maximum allowable value for the global norm of the gradient, which will be clipped as needed. If ``None``, no gradient clipping.'
    ),
    Kwarg(
        'regularizer_name',
        None,
        [str, None],
        "Name of global regularizer. If ``None``, no regularization."
    ),
    Kwarg(
        'regularizer_scale',
        0.01,
        float,
        "Scale of global regularizer (ignored if ``regularizer_name==None``)."
    ),
    Kwarg(
        'entropy_regularizer_scale',
        None,
        [float, None],
        "Scale of regularizer on classifier entropy. If ``None``, no entropy regularization."
    ),
    Kwarg(
        'input_dropout_rate',
        None,
        [float, None],
        "Rate at which to drop input data to the encoder. If ``None``, no input dropout."
    ),
    Kwarg(
        'ema_decay',
        0.999,
        [float, None],
        "Decay factor to use for exponential moving average for parameters (used in prediction)."
    ),
    Kwarg(
        'minibatch_size',
        128,
        [int, None],
        "Size of minibatches to use for fitting (full-batch if ``None``)."
    ),
    Kwarg(
        'eval_minibatch_size',
        5000,
        [int, None],
        "Size of minibatches to use for prediction/evaluation (full-batch if ``None``)."
    ),
    Kwarg(
        'float_type',
        'float32',
        str,
        "``float`` type to use throughout the network."
    ),
    Kwarg(
        'int_type',
        'int32',
        str,
        "``int`` type to use throughout the network (used for tensor slicing)."
    ),
    Kwarg(
        'save_freq',
        1,
        int,
        "Frequency (in iterations) with which to save model checkpoints."
    ),
    Kwarg(
        'log_graph',
        False,
        bool,
        "Log the network graph to Tensorboard"
    )
]


def dtsr_kwarg_docstring():
    out = ''

    for kwarg in ENCODER_DECODER_MORPH_LEARNER_INITIALIZATION_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    out += '\n'

    return out
