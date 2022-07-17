import json

import numpy as np

from .utils import get_mse, get_scale_zp


def plot_text(data_str, title, **kwargs):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    sns.set()

    ax = kwargs.pop("ax")

    ax.set_facecolor((1, 1, 1))
    ax.axis("off")
    ax.text(0, 0, data_str, transform=ax.transAxes)
    ax.set_title(title)


def plot_hist(data, title, low=None, high=None, **kwargs):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    sns.set()

    ax = kwargs.pop("ax")
    bins = kwargs.pop("bins", "scott")

    sns.histplot(
        data,
        bins=bins,
        ax=ax,
    )
    if low is not None:
        ax.axvline(x=low, color="r")
    if high is not None:
        ax.axvline(x=high, color="r")
    ax.set_yscale("log")
    ax.set_ylabel("log(Count)")
    ax.set_title(title)


def _plot_fake_quantize_node(**kwargs):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    sns.set()

    before_stats = kwargs.pop("before_stats")
    after_stats = kwargs.pop("after_stats")
    fq_stats = kwargs.pop("fq_stats")

    fq_attrs = kwargs.pop("fq_attrs")
    fq_config = kwargs.pop("fq_config")
    input_low, input_high, output_low, output_high = kwargs.pop("fq_params")

    fq_name = kwargs.pop("fq_name")
    fq_input_name = kwargs.pop("fq_input_name")

    channel_axis = kwargs.pop("channel_axis")

    levels = fq_attrs["levels"]
    scale, zero_point = get_scale_zp(output_low, output_high, levels)

    min_index = before_stats < min(input_low, input_high)
    max_index = before_stats > max(input_low, input_high)

    before_stats_q = before_stats.copy()
    before_stats_q[min_index] = min(input_low, input_high)
    before_stats_q[max_index] = max(input_low, input_high)
    before_stats_q = np.round(
        (before_stats_q - input_low) / (input_high - input_low) * (levels - 1)
    )
    before_stats_dq = (
        before_stats_q / (levels - 1) * (output_high - output_low) + output_low
    )

    fq_mse = get_mse(fq_stats, before_stats, channel_axis=channel_axis)
    fq_rmse = np.sqrt(fq_mse)
    fq_rmse_scale = fq_rmse / scale

    err_mse = get_mse(before_stats, after_stats, channel_axis=channel_axis)
    err_rmse = np.sqrt(err_mse)
    err_rmse_scale = err_rmse / scale

    n_rows = 4
    fig = plt.figure(figsize=(30, n_rows * 5))
    gs = GridSpec(nrows=n_rows, ncols=4)
    fig.suptitle(fq_name, fontsize=20)

    data_str = json.dumps(fq_config, indent=8)
    ax = fig.add_subplot(gs[0, 0:2])
    plot_text(data_str, "FQ Configuration", ax=ax)

    data = {
        "Scale": scale.item(),
        "Zero Point": zero_point.item(),
        "Levels": levels,
        "Clipping ratio": {
            "Minimum": min_index.mean().item(),
            "Maximum": max_index.mean().item(),
        },
    }
    data_str = json.dumps(data, indent=8)
    ax = fig.add_subplot(gs[0, 2:4])
    plot_text(data_str, "Additional Info", ax=ax)

    data = {
        "Error": {
            "MSE": err_mse.item(),
            "RMSE": err_rmse.item(),
            "RMSE/scale": err_rmse_scale.item(),
        },
    }
    data_str = json.dumps(data, indent=8)
    ax = fig.add_subplot(gs[1, 3])
    plot_text(data_str, "Additional Info", ax=ax)

    ax = fig.add_subplot(gs[1, :3])
    plot_hist(
        {
            "before quantized": np.reshape(before_stats, -1),
            "after quantized": np.reshape(after_stats, -1),
        },
        f"Output disrbitution of node {fq_input_name}",
        input_low,
        input_high,
        ax=ax,
    )

    data = {
        "Error": {
            "MSE": fq_mse.item(),
            "RMSE": fq_rmse.item(),
            "RMSE/scale": fq_rmse_scale.item(),
        },
    }
    data_str = json.dumps(data, indent=8)
    ax = fig.add_subplot(gs[2, 3])
    plot_text(data_str, "Additional Info", ax=ax)

    ax = fig.add_subplot(gs[2, :3])
    plot_hist(
        {
            "before quantized": np.reshape(before_stats, -1),
            "after quantized": np.reshape(fq_stats, -1),
        },
        f"Fake quantized output distribution of node {fq_name}",
        output_low,
        output_high,
        ax=ax,
    )

    ax = fig.add_subplot(gs[3, :])
    plot_hist(
        np.reshape(before_stats_q, -1),
        f"Quantized output distribution of node {fq_name}",
        0,
        levels,
        bins=levels,
        ax=ax,
    )

    plt.tight_layout()

    return fig


def _plot_node(**kwargs):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    sns.set()

    before_stats = kwargs.pop("before_stats")
    after_stats = kwargs.pop("after_stats")

    node_name = kwargs.pop("node_name")

    channel_axis = kwargs.pop("channel_axis")

    mse = get_mse(before_stats, after_stats, channel_axis=channel_axis)
    rmse = np.sqrt(mse)

    n_rows = 1
    fig = plt.figure(figsize=(30, n_rows * 5))
    gs = GridSpec(nrows=n_rows, ncols=4)
    fig.suptitle(node_name, fontsize=20)

    data = {
        "input original - output dequantized": {
            "MSE": mse.item(),
            "RMSE": rmse.item(),
        }
    }
    data_str = json.dumps(data, indent=8)
    ax = fig.add_subplot(gs[0, 3])
    plot_text(data_str, "Error", ax=ax)

    ax = fig.add_subplot(gs[0, :3])
    plot_hist(
        {
            "before quantized": np.reshape(before_stats, -1),
            "after quantized": np.reshape(after_stats, -1),
        },
        f"Output distribution of node {node_name}",
        ax=ax,
    )

    plt.tight_layout()

    return fig


def _plot_weight_node(**kwargs):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    sns.set()

    weights = kwargs.pop("weights")

    fq_attrs = kwargs.pop("fq_attrs")
    fq_config = kwargs.pop("fq_config")
    input_low, input_high, output_low, output_high = kwargs.pop("fq_params")

    weight_name = kwargs.pop("weight_name")
    fq_name = kwargs.pop("fq_name")

    channel_axis = kwargs.pop("channel_axis")
    top_k = kwargs.pop("top_k")

    if channel_axis > 0:
        weights = np.transpose(weights, (channel_axis, 0))
        input_low = np.transpose(input_low, (channel_axis, 0))
        input_high = np.transpose(input_high, (channel_axis, 0))
        output_low = np.transpose(output_low, (channel_axis, 0))
        output_high = np.transpose(output_high, (channel_axis, 0))

    levels = fq_attrs["levels"]
    scale, zero_point = get_scale_zp(output_low, output_high, levels)

    min_index = weights < input_low
    max_index = weights > input_high

    weights_q = weights.copy()
    if min_index.sum() > 0:
        for w, v, i in zip(weights_q, input_low, min_index):
            if i.sum() == 0:
                continue
            w[i] = v
    if max_index.sum() > 0:
        for w, v, i in zip(weights_q, input_high, max_index):
            if i.sum() == 0:
                continue
            w[i] = v

    weights_q = np.round(
        (weights_q - input_low) / (input_high - input_low) * (levels - 1)
    )
    weights_dq = weights_q / (levels - 1) * (output_high - output_low) + output_low

    mse = get_mse(weights, weights_dq, channel_axis=channel_axis)
    rmse = np.sqrt(mse)
    rmse_scale = rmse / scale.squeeze()

    indices = np.argsort(mse)[::-1][:top_k]

    n_rows = top_k + 1
    fig = plt.figure(figsize=(30, n_rows * 5))
    gs = GridSpec(nrows=n_rows, ncols=4)
    fig.suptitle(weight_name, fontsize=20)

    data_str = json.dumps(fq_config, indent=8)
    ax = fig.add_subplot(gs[0, 0:2])
    plot_text(data_str, "FQ Configuration", ax=ax)

    data = {
        "Levels": levels,
        "Clipping ratio": {
            "Minimum": min_index.mean().item(),
            "Maximum": max_index.mean().item(),
        },
        "Mean error across channel": {
            "MSE": mse.mean().item(),
            "RMSE": rmse.mean().item(),
            "RMSE/scale": rmse_scale.mean().item(),
        },
    }
    data_str = json.dumps(data, indent=8)
    ax = fig.add_subplot(gs[0, 2:])
    plot_text(data_str, "Additional Info", ax=ax)

    #  level_low = fq_config["level_low"]
    #  level_high = fq_config["level_high"]
    #  ax = fig.add_subplot(gs[-1, :])
    #  sns.histplot(
    #      {
    #          "before quantized": weights_q[indices[-1]],
    #      },
    #      bins="scott",
    #      ax=ax,
    #  )
    #  ax.axvline(x=level_low, color="r")
    #  ax.axvline(x=level_high, color="r")
    #  ax.set_yscale("log")
    #  ax.set_ylabel("log(Count)")

    for index, k_index in enumerate(indices):
        index += 1

        data = {
            "Scale": scale[k_index].item(),
            "Zero Point": zero_point[k_index].item(),
            "Levels": levels,
            "Clipping ratio": {
                "Minimum": min_index[k_index].mean().item(),
                "Maximum": max_index[k_index].mean().item(),
            },
            "Error": {
                "MSE": mse[k_index].item(),
                "RMSE": rmse[k_index].item(),
                "RMSE/scale": rmse_scale[k_index].item(),
            },
        }
        data_str = json.dumps(data, indent=8)
        ax = fig.add_subplot(gs[index, 3])
        plot_text(data_str, "Additional Info", ax=ax)

        ax = fig.add_subplot(gs[index, :3])
        plot_hist(
            {
                "before quantized": weights[k_index],
                "after quantized": weights_dq[k_index],
            },
            f"Output distribution of weight node {fq_name} at channel {k_index}",
            output_low[k_index],
            output_high[k_index],
            ax=ax,
        )

    plt.tight_layout()

    return fig
