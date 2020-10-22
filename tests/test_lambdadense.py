from lambda_layers import LambdaNetwork1Dense
import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LG_LEVEL"] = "2"


class LambdaNetwork1DConvTest(tf.test.TestCase):

    layer = LambdaNetwork1Dense(
        kernel_out=32,
        key_depth=16,
        intra_depth=1,
        heads=4,
        size=64,
        data_format="channels_last",
        norm_keys=False,
    )
    test_output_shape = (1, 64, 32)
    case_layer_out = layer([tf.random.uniform(test_output_shape)])
    case_out = layer.compute_output_shape(test_output_shape)
    assert case_layer_out.shape == case_out

    def testLambdaNetworkDenseWih2DConv(self):

        batch_size = 1
        size = 64
        n_channels = 32
        kernel_out = 32
        key_depth = 16
        intra_depth = 1
        heads = 4

        # init layer with embedding
        layers_1_test = LambdaNetwork1Dense(
            kernel_out=kernel_out,
            key_depth=key_depth,
            intra_depth=intra_depth,
            heads=heads,
            size=size,
            data_format="channels_last",
            norm_keys=False,
            debug_mode=True,
        )
        test_img = np.zeros((batch_size, size, n_channels))
        context_image = np.zeros((batch_size, size, n_channels))

        # test laer context and image stability
        _, debug_dict_1 = layers_1_test([test_img])
        _, debug_dict_2 = layers_1_test([test_img, test_img])
        outputs, debug_dict_3 = layers_1_test([test_img, context_image])
        # check context and image itself
        assert test_img.shape == context_image.shape
        # check layer 1 out
        self.assertShapeEqual(test_img, debug_dict_1["x"])
        self.assertAllEqual(test_img, debug_dict_1["x"])
        self.assertAllEqual(test_img, debug_dict_1["c"])
        # check layer 2 out
        self.assertShapeEqual(test_img, debug_dict_2["x"])
        self.assertAllEqual(test_img, debug_dict_2["x"])
        self.assertAllEqual(test_img, debug_dict_2["c"])
        # check layer 3 out
        self.assertShapeEqual(test_img, debug_dict_3["x"])
        self.assertAllEqual(test_img, debug_dict_3["x"])
        self.assertAllEqual(context_image, debug_dict_3["c"])

        # init test output layers here nothing change due to change context
        self.assertShapeEqual(
            np.zeros((batch_size, heads, size, key_depth)), debug_dict_3["q"]
        )
        self.assertShapeEqual(
            np.zeros((batch_size, intra_depth, size, key_depth)), debug_dict_3["k"]
        )
        self.assertShapeEqual(
            np.zeros((batch_size, intra_depth, size, kernel_out // heads)),
            debug_dict_3["v"],
        )
        self.assertShapeEqual(
            np.zeros((batch_size, key_depth, kernel_out // heads)),
            debug_dict_3["lambda_c"],
        )
        self.assertShapeEqual(
            np.zeros((batch_size, key_depth, kernel_out // heads)),
            debug_dict_3["lambda_p"],
        )
        self.assertShapeEqual(np.zeros((batch_size, size, n_channels)), outputs)

    def testLambdaNetworkDenseWihEmb(self):

        batch_size = 1
        size = 64
        n_channels = 32
        kernel_out = 32
        key_depth = 16
        intra_depth = 1
        heads = 4
        receptive_kernel = 13

        # init layer with embedding
        layers_2_test = LambdaNetwork1Dense(
            kernel_out=kernel_out,
            key_depth=key_depth,
            intra_depth=intra_depth,
            heads=heads,
            receptive_kernel=receptive_kernel,
            data_format="channels_last",
            norm_keys=False,
            debug_mode=True,
        )
        test_img = np.zeros((batch_size, size, n_channels))
        context_image = np.zeros((batch_size, size, n_channels))

        outputs, debug_dict = layers_2_test([test_img, context_image])
        # init test output layers here nothing change due to change context
        self.assertShapeEqual(
            np.zeros((batch_size, heads, size, key_depth)), debug_dict["q"]
        )
        self.assertShapeEqual(
            np.zeros((batch_size, intra_depth, size, key_depth)), debug_dict["k"]
        )
        self.assertShapeEqual(
            np.zeros((batch_size, kernel_out // heads, intra_depth, size)),
            debug_dict["v"],
        )
        self.assertShapeEqual(
            np.zeros((batch_size, key_depth, kernel_out // heads)),
            debug_dict["lambda_c"],
        )
        self.assertShapeEqual(
            np.zeros((batch_size, key_depth, (kernel_out // heads))),
            debug_dict["lambda_p"],
        )
        self.assertShapeEqual(np.zeros((batch_size, size, n_channels)), outputs)


if __name__ == "__main__":
    tf.test.main()
