import torch
from torch.utils.data import DataLoader
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


def get_dvs128_dataloader(
    root="./Dataset",
    train=True,
    frames_number=20,
    batch_size=8,
    shuffle=True,
    num_workers=0
):
   
    dataset = DVS128Gesture(
        root=root,
        train=train,
        data_type='frame',
        frames_number=frames_number,
        split_by='number'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if train else False,
        num_workers=num_workers
    )

    return dataset ,dataloader