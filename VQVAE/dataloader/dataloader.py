# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from VQVAE.dataloader.sparse_utils import (
    load_processed_record,
    sparse_joint_indices,
    split_processed_paths,
    stable_sequence_name,
)


class TrainDataset(Dataset):
    def __init__(self, motions, sparses, all_info, input_motion_length=196,
                 train_dataset_repeat_times=1, normalization=True, sparse_dim=54):
        self.motions = motions
        self.sparses = sparses
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.input_motion_length = input_motion_length
        self.motion_54 = []
        self.up_idx = sparse_joint_indices(sparse_dim)
        self.n_joint = 22
        for sparse in self.sparses:
            rot_abs = sparse[:, :self.n_joint * 6].reshape(-1, self.n_joint, 6)
            rot_vel = sparse[:, self.n_joint * 6:self.n_joint * 12].reshape(-1, self.n_joint, 6)
            pos = sparse[:, self.n_joint * 12:self.n_joint * 15].reshape(-1, self.n_joint, 3)
            pos_vel = sparse[:, self.n_joint * 15:self.n_joint * 18].reshape(-1, self.n_joint, 3)
            self.motion_54.append(torch.cat((rot_abs, rot_vel, pos, pos_vel), dim=-1)[:, self.up_idx])

    def __len__(self):
        return len(self.motions) * self.train_dataset_repeat_times

    def __getitem__(self, idx):
        sparse = self.motion_54[idx % len(self.motions)].float()
        motion = self.motions[idx % len(self.motions)].float()
        seqlen = motion.shape[0]
        start = 0 if seqlen <= self.input_motion_length else torch.randint(
            0, int(seqlen - self.input_motion_length), (1,)
        )[0]
        return (
            motion[start:start + self.input_motion_length],
            sparse[start:start + self.input_motion_length],
        )


class TestDataset(Dataset):
    def __init__(self, all_info, filename_list, sparse_dim=54):
        self.filename_list = filename_list
        self.motions = [item["rotation_local_full_gt_list"] for item in all_info]
        self.sparses = [item["hmd_position_global_full_gt_list"] for item in all_info]
        self.body_params = [item["body_parms_list"] for item in all_info]
        self.head_motion = [item["head_global_trans_list"] for item in all_info]
        self.motion_54 = []
        self.up_idx = sparse_joint_indices(sparse_dim)
        self.n_joint = 22
        for sparse in self.sparses:
            rot_abs = sparse[:, :self.n_joint * 6].reshape(-1, self.n_joint, 6)
            rot_vel = sparse[:, self.n_joint * 6:self.n_joint * 12].reshape(-1, self.n_joint, 6)
            pos = sparse[:, self.n_joint * 12:self.n_joint * 15].reshape(-1, self.n_joint, 3)
            pos_vel = sparse[:, self.n_joint * 15:self.n_joint * 18].reshape(-1, self.n_joint, 3)
            self.motion_54.append(torch.cat((rot_abs, rot_vel, pos, pos_vel), dim=-1)[:, self.up_idx])

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        return (
            self.motions[idx],
            self.motion_54[idx],
            self.body_params[idx],
            self.head_motion[idx],
            self.filename_list[idx],
        )


def get_motion(motion_list):
    return (
        [item["rotation_local_full_gt_list"] for item in motion_list],
        [item["hmd_position_global_full_gt_list"] for item in motion_list],
    )


def get_path(dataset_path, split, protocol):
    split_files, _, processed = split_processed_paths(dataset_path, protocol, split)
    print(f"{split} using semantic splits: {', '.join(map(str, split_files))}")
    return [str(path) for path in processed]


def load_data(dataset_path, split, protocol, **kwargs):
    split_files, sources, processed = split_processed_paths(dataset_path, protocol, split)
    print(f"Loading {len(processed)} records from semantic splits: {', '.join(map(str, split_files))}")
    records = [load_processed_record(path, dataset_path) for path in tqdm(processed)]

    if split == "test":
        return [stable_sequence_name(source) for source in sources], records

    if split not in ("train", "val"):
        raise ValueError(f"Unsupported split: {split}")
    if "input_motion_length" not in kwargs:
        raise ValueError("Please specify input_motion_length")

    input_motion_length = kwargs["input_motion_length"]
    motions, sparses = get_motion(records)
    filtered_motions = []
    filtered_sparses = []
    filtered_records = []
    for motion, sparse, record in zip(motions, sparses, records):
        if motion.shape[0] < input_motion_length:
            continue
        filtered_motions.append(motion)
        filtered_sparses.append(sparse)
        filtered_records.append(record)
    return filtered_motions, filtered_sparses, filtered_records


def get_dataloader(dataset, split, batch_size, num_workers=32):
    is_train = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train,
        persistent_workers=num_workers > 0,
    )
