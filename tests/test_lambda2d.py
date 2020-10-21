from lambda_layers import LambdaNetwork2D
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LG_LEVEL'] = '2'

class LambdaNetwork2DTest(tf.test.TestCase):

    layer = LambdaNetwork2D(
        kernel_out=32,
        key_depth=16,
        intra_depth=1,
        heads=4,
        size=64 * 64,
        data_format='channels_last',
        norm_keys=False,
    )
    test_output_shape = (1, 64, 64, 32)
    case_layer_out = layer([tf.random.uniform(test_output_shape)])
    case_out = layer.compute_output_shape(test_output_shape)
    assert case_layer_out.shape == case_out

    def testLambdaNetwork2DWih3Conv(self):

        batch_size = 1
        image_width = 64
        image_height = 64
        n_channels = 32
        kernel_out = 32
        key_depth = 16
        intra_depth = 1
        heads = 4
        size = image_width * image_height

        # init layer with embedding
        layers_1_test = LambdaNetwork2D(
                                        kernel_out=kernel_out,
                                        key_depth=key_depth,
                                        intra_depth=intra_depth,
                                        heads=heads,
                                        size=size,
                                        data_format='channels_last',
                                        norm_keys=False,
                                        debug_mode=True
                                        )
        test_img = np.zeros((batch_size, image_width, image_height, n_channels))
        context_image = np.zeros((batch_size, image_width, image_height, n_channels))

        # test laer context and image stability
        _, debug_dict_1 = layers_1_test([test_img])
        _, debug_dict_2 = layers_1_test([test_img, test_img])
        outputs, debug_dict_3 = layers_1_test([test_img, context_image])
        # check context and image itself
        assert test_img.shape == context_image.shape
        # check layer 1 out
        self.assertShapeEqual(test_img, debug_dict_1['x'])
        self.assertAllEqual(test_img, debug_dict_1['x'])
        self.assertAllEqual(test_img, debug_dict_1['c'])
        # check layer 2 out
        self.assertShapeEqual(test_img, debug_dict_2['x'])
        self.assertAllEqual(test_img, debug_dict_2['x'])
        self.assertAllEqual(test_img, debug_dict_2['c'])
        # check layer 3 out
        self.assertShapeEqual(test_img, debug_dict_3['x'])
        self.assertAllEqual(test_img, debug_dict_3['x'])
        self.assertAllEqual(context_image, debug_dict_3['c'])

        # init test output layers here nothing change due to change context
        self.assertShapeEqual(np.zeros((batch_size, heads, key_depth, size)),
                              debug_dict_3["q"])
        self.assertShapeEqual(np.zeros((batch_size, intra_depth, key_depth, size)),
                              debug_dict_3["k"])
        self.assertShapeEqual(np.zeros((batch_size, intra_depth, kernel_out // heads, size)),
                              debug_dict_3["v"])
        self.assertShapeEqual(np.zeros((batch_size, key_depth, kernel_out // heads)),
                              debug_dict_3["lambda_c"])
        self.assertShapeEqual(np.zeros((batch_size, size, key_depth, kernel_out // heads)),
                              debug_dict_3["lambda_p"])
        self.assertShapeEqual(np.zeros((batch_size, image_width, image_height, n_channels)),
                              outputs)

    def testLambdaNetwork2DWihEmb(self):
        batch_size = 1
        image_width = 64
        image_height = 64
        n_channels = 32
        kernel_out = 32
        key_depth = 16
        intra_depth = 1
        heads = 4
        receptive_kernel = 13
        size = image_width * image_height

        # init layer with embedding
        layers_2_test = LambdaNetwork2D(
            kernel_out=kernel_out,
            key_depth=key_depth,
            intra_depth=intra_depth,
            heads=heads,
            receptive_kernel=receptive_kernel,
            data_format='channels_last',
            norm_keys=False,
            debug_mode=True
        )
        test_img = tf.ones([batch_size, image_width, image_height, n_channels])
        context_image = tf.ones([batch_size, image_width, image_height, n_channels])

        outputs, debug_dict = layers_2_test([test_img, context_image])
        # init test output layers here nothing change due to change context
        self.assertShapeEqual(np.zeros((batch_size, heads, key_depth, size)),
                              debug_dict["q"])
        self.assertShapeEqual(np.zeros((batch_size, intra_depth, key_depth, size)),
                              debug_dict["k"])
        # reshape in code so check if it is correct
        self.assertShapeEqual(np.zeros((batch_size, kernel_out // heads,
                                        image_width, image_height, intra_depth)),
                              debug_dict["v"])
        self.assertShapeEqual(np.zeros((batch_size, key_depth, kernel_out // heads)),
                              debug_dict["lambda_c"])
        self.assertShapeEqual(np.zeros((batch_size, key_depth, kernel_out // heads, size)),
                              debug_dict["lambda_p"])
        self.assertShapeEqual(np.zeros((batch_size, image_width, image_height, n_channels)),
                              outputs)


if __name__ == '__main__':
    tf.test.main()
