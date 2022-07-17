import numpy as np


def get_mse(data1, data2, channel_axis=-1):
    assert data1.shape == data2.shape

    diff = np.square(data1 - data2)
    # channel-wise
    if channel_axis > 0:
        diff = np.transpose(diff, (channel_axis, 0))
    channel_dim = diff.shape[0]
    diff = diff.reshape(channel_dim, -1)
    mse = diff.mean(-1)

    if channel_axis == -1:
        mse = mse.mean()

    return mse


def get_scale_zp(output_low, output_high, levels):
    scale = (output_high - output_low) / (levels - 1)
    zero_point = -output_low / (output_high - output_low) * (levels - 1)
    return scale, zero_point
