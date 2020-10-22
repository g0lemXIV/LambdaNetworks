import tensorflow as tf
from einops import rearrange


class LambdaNetwork1DBase(tf.keras.layers.Layer):
    def __init__(
        self,
        type_: str,
        kernel_out: int,
        key_depth: int,
        intra_depth: int = 1,
        heads: int = 4,
        size: int = None,
        receptive_kernel: int = None,
        data_format: str = "channels_last",
        norm_keys: bool = False,
        debug_mode=False,
        **kwargs
    ) -> None:
        super(LambdaNetwork1DBase, self).__init__()
        # check parameters
        self._validate_init_args(
            kernel_out=kernel_out,
            heads=heads,
            r=receptive_kernel,
            size=size,
            channel=data_format,
        )
        # parameters
        self.use_context = True if receptive_kernel is not None else False
        self.data_format = data_format
        self.norm_keys = norm_keys
        self.intra_depth = intra_depth
        self.heads = heads
        self.debug_mode = debug_mode
        self.kernel_out = kernel_out
        values_depth = kernel_out // heads

        # init layers

        if type_ == "dense":
            # init time distributed layers
            ts = tf.keras.layers.TimeDistributed
            self.q_layer = ts(
                tf.keras.layers.Dense(key_depth * heads, use_bias=False, **kwargs)
            )
            self.k_layer = ts(
                tf.keras.layers.Dense(key_depth * intra_depth, use_bias=False, **kwargs)
            )
            self.v_layer = ts(
                tf.keras.layers.Dense(
                    values_depth * intra_depth, use_bias=False, **kwargs
                )
            )
        elif type_ == "1dconv":
            # init convolution layers
            self.q_layer = tf.keras.layers.Conv1D(
                key_depth * heads, 1, use_bias=False, **kwargs
            )
            self.k_layer = tf.keras.layers.Conv1D(
                key_depth * intra_depth, 1, use_bias=False, **kwargs
            )
            self.v_layer = tf.keras.layers.Conv1D(
                values_depth * intra_depth, 1, use_bias=False, **kwargs
            )

        # init normalization layers
        self.normalization_q = tf.keras.layers.BatchNormalization()
        self.normalization_v = tf.keras.layers.BatchNormalization()
        # add normalization for keys
        if norm_keys is True:
            self.normalization_k = tf.keras.layers.BatchNormalization()

        if self.use_context:
            self.embedding = tf.keras.layers.Conv2D(
                key_depth, (receptive_kernel, receptive_kernel), padding="same"
            )
        else:
            self.embedding = tf.Variable(
                initial_value=tf.random.uniform(shape=[size, size, key_depth]),
                trainable=True,
                name="PositionalEmbedding",
            )

    def call(self, inputs):
        self._validate_call_args(inputs=inputs)

        # check for different context
        x = inputs[0]
        c = inputs[1] if len(inputs) > 1 else x

        # reshape to channels last
        if self.data_format == "channels_first":
            x = tf.einsum("b f t -> b t f", x)
            c = tf.einsum("b f t -> b t f", c)

        # initialization of base parameters
        q = self.q_layer(x)
        k = self.k_layer(c)
        v = self.v_layer(c)
        # normalization
        q = self.normalization_q(q)
        v = self.normalization_v(v)
        if self.norm_keys:
            k = self.normalization_k(k)

        k = tf.keras.activations.softmax(k, axis=-1)
        # rearrange dimensions
        q = rearrange(q, "b t (h k) -> b h t k", h=self.heads)
        k = rearrange(k, "b t (u k) -> b u t k", u=self.intra_depth)
        v = rearrange(v, "b t (u v) -> b u t v", u=self.intra_depth)
        # create context lambda and output
        lambda_c = tf.einsum("b u t k, b u t v -> b k v", k, v)
        Yc = tf.einsum("b h t q, b k v -> b h t v", q, lambda_c)
        # calculate positional lambda and output
        if self.use_context:
            v = tf.einsum("b u t v -> b v u t", v)
            lambda_p = self.embedding(v)
            lambda_p = rearrange(lambda_p, "b v u t -> b t (v u)")
            Yp = tf.einsum("b h t q, b k v -> b h t v", q, lambda_p)
        else:
            lambda_p = tf.einsum("t t k , b u t v -> b k v", self.embedding, v)
            Yp = tf.einsum("b h t q, b k v -> b h t v", q, lambda_p)

        Y = tf.keras.layers.Add()([Yc, Yp])
        out = rearrange(Y, "b h t c -> b t (h c)")

        if self.debug_mode:
            debug_dict = dict()
            debug_dict["x"] = x
            debug_dict["c"] = c
            debug_dict["q"] = q
            debug_dict["v"] = v
            debug_dict["k"] = k
            debug_dict["lambda_p"] = lambda_p
            debug_dict["lambda_c"] = lambda_c
            return out, debug_dict

        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.kernel_out)

    def _validate_call_args(self, inputs):
        """Validates arguments of the call method."""
        class_name = self.__class__.__name__
        if not isinstance(inputs, list):
            raise ValueError(
                "{} layer must be called on a list of inputs, namely [query]"
                "or [query, context].".format(class_name)
            )
        if len(inputs) < 1 or len(inputs) > 2:
            raise ValueError(
                "{} layer accepts inputs list of length 1 or 2, "
                "namely [query] or [query, context]. "
                "Given length: {}".format(class_name, len(inputs))
            )
        if len(inputs) > 1:
            if inputs[0].shape != inputs[1].shape:
                raise ValueError(
                    "{} layer should have same last dimension of inputs "
                    "and context tensor."
                    "Given dimension inputs:{}, context: {}".format(
                        class_name, inputs[0].shape, inputs[1].shape
                    )
                )

    def _validate_init_args(self, kernel_out, heads, r, size, channel):
        class_name = self.__class__.__name__
        if (kernel_out % heads) != 0:
            raise ValueError(
                "{} layer should have kernel_out dimension divisible by number of heads."
                "Given kernel dimension:{}, heads: {}".format(
                    class_name, kernel_out, heads
                )
            )
        if (r is not None) and (r % 2 != 1):
            raise ValueError(
                "{} layer should have odd receptive kernel size."
                "Given receptive kernel dimension:{}x{}".format(class_name, r, r)
            )
        if (r is None) and (size is None):
            raise ValueError(
                "{} layer should have specify total sequence length where "
                "size = w x h of the last layer"
                "or receptive kernel size.".format(class_name)
            )
        if channel not in ["channels_first", "channels_last"]:
            raise ValueError(
                "{} layer should have channels_first, channels_last format."
                "It is {}".format(class_name, channel)
            )


class LambdaNetwork1DConv(LambdaNetwork1DBase):
    def __init__(
        self,
        kernel_out,
        key_depth,
        intra_depth=1,
        heads=4,
        size=None,
        receptive_kernel=None,
        data_format="channels_last",
        norm_keys=False,
        debug_mode=False,
        **kwargs
    ):
        super(LambdaNetwork1DConv, self).__init__(
            type_="1dconv",
            kernel_out=kernel_out,
            key_depth=key_depth,
            intra_depth=intra_depth,
            heads=heads,
            size=size,
            receptive_kernel=receptive_kernel,
            data_format=data_format,
            norm_keys=norm_keys,
            debug_mode=debug_mode,
            **kwargs
        )


class LambdaNetwork1Dense(LambdaNetwork1DBase):
    def __init__(
        self,
        kernel_out,
        key_depth,
        intra_depth=1,
        heads=4,
        size=None,
        receptive_kernel=None,
        data_format="channels_last",
        norm_keys=False,
        debug_mode=False,
        **kwargs
    ):
        super(LambdaNetwork1Dense, self).__init__(
            type_="dense",
            kernel_out=kernel_out,
            key_depth=key_depth,
            intra_depth=intra_depth,
            heads=heads,
            size=size,
            receptive_kernel=receptive_kernel,
            data_format=data_format,
            norm_keys=norm_keys,
            debug_mode=debug_mode,
            **kwargs
        )
