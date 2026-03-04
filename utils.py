import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

def inject_label_conv(input_seq, label, num_classes):
    """
    input_seq: [T, B, C, H, W]
    label:     [B]
    returns:   [T, B, C+num_classes, H, W]
    """

    T, B, C, H, W = input_seq.shape
    device = input_seq.device

    # One-hot
    one_hot = torch.zeros(B, num_classes, device=device)
    one_hot[torch.arange(B), label] = 1.0

    # Expand spatially
    label_map = one_hot.view(B, num_classes, 1, 1)
    label_map = label_map.expand(B, num_classes, H, W)

    # Repeat across time
    label_map = label_map.unsqueeze(0).repeat(T, 1, 1, 1, 1)

    # Concatenate on channel dimension
    return torch.cat([input_seq, label_map], dim=2)

def spike_count_goodness(spike_record):
    """
    spike_record: [T, B, N]
    returns: [B] goodness per sample
    """
    return spike_record.sum(dim=(0, 2))

def stdp_update(pre_spikes, post_spikes, weights, lr, modulator):
    """
    pre_spikes:  [T, B, N_pre]
    post_spikes: [T, B, N_post]
    weights:     [N_pre, N_post]
    modulator:   +1 or -1
    """
    T = pre_spikes.shape[0]

    correlation = torch.einsum("tbi,tbj->ij", pre_spikes, post_spikes)
    weights.data += lr * modulator * correlation / T

def homeostatic_threshold_update(threshold, spike_record, target_rate, lr):
    """
    threshold: [N]
    spike_record: [T, B, N]
    """
    firing_rate = spike_record.mean(dim=(0, 1))
    threshold.data += lr * (firing_rate - target_rate)


def visualize(event, label, binned=False, interval=200):

    fig, ax = plt.subplots()
    ax.set_title(f"Label: {label}")

    on_scatter = ax.scatter([], [], s=5, c='red', label='ON')
    off_scatter = ax.scatter([], [], s=5, c='blue', label='OFF')
    ax.legend()

    # =========================
    # CASE 1: RAW EVENT DATA
    # =========================
    if not binned:

        x = np.array(event['x'])
        y = np.array(event['y'])
        t = np.array(event['t'])
        p = np.array(event['p'])

        indices = np.argsort(t)
        x = x[indices]
        y = y[indices]
        t = t[indices]
        p = p[indices]

        ax.set_xlim(0, 127)
        ax.set_ylim(0, 127)
        ax.invert_yaxis()

        def update(frame):

            current_time = t[frame]
            mask = t <= current_time

            on_mask = mask & (p == 1)
            off_mask = mask & (p == 0)

            on_scatter.set_offsets(np.column_stack((x[on_mask], y[on_mask])))
            off_scatter.set_offsets(np.column_stack((x[off_mask], y[off_mask])))

            return on_scatter, off_scatter

        frames = len(t)

    # =========================
    # CASE 2: BINNED DATA
    # =========================
    else:

        # Remove batch dimension if exists
        if len(event.shape) == 5:
            event = event[0]

        T, C, H, W = event.shape

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.invert_yaxis()

        def update(frame):

            on_frame = event[frame, 0]
            off_frame = event[frame, 1]

            on_y, on_x = np.where(on_frame > 0)
            off_y, off_x = np.where(off_frame > 0)

            on_scatter.set_offsets(np.column_stack((on_x, on_y)))
            off_scatter.set_offsets(np.column_stack((off_x, off_y)))

            return on_scatter, off_scatter

        frames = T

    # =========================
    # Animation
    # =========================
    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        blit=False
    )

    plt.show()

