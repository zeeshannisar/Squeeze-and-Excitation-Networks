# Squeeze and Excitation Networks
[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) implementation in Tensorflow 2.

<p align="center">
    <img src="https://github.com/zeeshannisar/CX_GAN/blob/master/ReadMe%20Images/integrated%20model.png" >
</p>

```
def squeeze_excite_block_(input, ratio=8, kernel_initializer='glorot_uniform', name=None):
    x = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = x.shape[channel_axis]
    x_shape = (1, 1, filters)

    x = GlobalAveragePooling2D(name=name + '_GAP')(x)
    x = Reshape(x_shape, name=name + '_reshape')(x)

    x = tf.keras.layers.Conv2D(filters=filters // ratio, kernel_size=1, strides=(1, 1), use_bias=True,
                               kernel_initializer=kernel_initializer, name=name + '_FC1')(x)
    x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0, name=name + '_relu')
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=(1, 1), use_bias=True,
                               kernel_initializer=kernel_initializer, name=name + '_FC2')(x)
    x = tf.keras.activations.sigmoid(x, name=name + '_sigmoid')
    if K.image_data_format() == 'channels_first':
        x = Permute((3, 1, 2))(x)
    output = multiply([input, x], name=name + '_multiply')
    return output
    
```
