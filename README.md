# LambdaNetworks
![Python tests](https://github.com/g0lemXIV/LambdaNetworks/workflows/Python%20tests/badge.svg?branch=main) ![License](https://img.shields.io/pypi/l/ansicolortags.svg)

Tensorflow implementation of [Lambda Network](https://openreview.net/forum?id=xTJEN-ggl1b) framework for capturing long-range interaction between input and structured information.

Paper review: [Yannic Kilcher's channel](https://www.youtube.com/watch?v=3qxJ2WD8p4w)
Pytorch implementation (I was based on): [lucidrains](https://github.com/lucidrains/lambda-networks)  

*However, I will implement 1D convolution lambda and lambda dense here, soon...*

## Installation  
```bash
git clone <repository>
cd LambdaNetworks
pip install .
```

## Examples of usage

**Using Lambda 2D**
```python
from lambda_layers import LambdaNetwork2DConv

layer = LambdaNetwork2DConv(kernel_out = 32,  # output of the layer
                            key_depth = 16, # depth of keys
                            intra_depth = 1, depth of 
                            heads = 4, # number of heads
                            size = 28 * 28, # total size of the input image (use for global embedding)
                            receptive_kernel = 7, # dimension of kernel if local embedding is using
                            data_format = "channels_last", # data format
                            norm_keys = False, # normalization of the key before activation function
                            **kwargs # additional args which can use in queries, keys, and values
                            )
```
**Using Lambda 1D/Dense**
```python
from lambda_layers import LambdaNetwork1DConv, LambdaNetwork1Dense

layer = LambdaNetwork1DConv(kernel_out = 32,  # output of the layer
                            key_depth = 16, # depth of keys
                            intra_depth = 1, # infra-depth of the layer
                            heads = 4, # number of heads
                            size = 28, # total number of timesteps
                            receptive_kernel = 7, # dimension of kernel if local embedding is using
                            data_format = "channels_last", # data format
                            norm_keys = False, # normalization of the key before activation function
                            **kwargs # additional args which can use in queries, keys, and values
                            )
                            
layer = LambdaNetwork1Dense(kernel_out = 32,  # output of the layer
                            key_depth = 16, # depth of keys
                            intra_depth = 1, # infra-depth of the layer
                            heads = 4, # number of heads
                            size = 28, # total number of timesteps
                            receptive_kernel = 7, # dimension of kernel if local embedding is using
                            data_format = "channels_last", # data format
                            norm_keys = False, # normalization of the key before activation function
                            **kwargs # additional args which can use in queries, keys, and values
                            )
```

## Citations

```
@inproceedings{
    anonymous2021lambdanetworks,
    title={LambdaNetworks: Modeling long-range Interactions without Attention},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=xTJEN-ggl1b},
    note={under review}
}
```
