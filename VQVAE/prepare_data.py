"""Deterministically preprocess AMASS using semantic NPZ split manifests."""

import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from tqdm import tqdm
from yacs.config import CfgNode as CN

from VQVAE.dataloader.sparse_utils import normalize_source_path, protocol_split_files
from VQVAE.utils import utils_transform


def load_cfg(path):
    if path is None:
        return None
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(str(path))
    return cfg


def read_sources(split_files):
    sources = set()
    for split_file in split_files:
        for line_number, line in enumerate(split_file.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            source = normalize_source_path(line.strip())
            if source in sources:
                continue
            sources.add(source)
    return sorted(sources)


def create_body_model(support_dir, device):
    bm_fname = support_dir / "smplh" / "male" / "model.npz"
    dmpl_fname = support_dir / "dmpls" / "male" / "model.npz"
    if not bm_fname.is_file() or not dmpl_fname.is_file():
        raise FileNotFoundError(f"Missing male SMPL-H/DMPL assets under {support_dir}")
    return BodyModel(
        bm_fname=str(bm_fname),
        num_betas=16,
        num_dmpls=8,
        dmpl_fname=str(dmpl_fname),
    ).to(device)


def scalar_string(value):
    array = np.asarray(value)
    return str(array.item() if array.shape == () else value)


def preprocess_sequence(source_file, source_relative, body_model, device):
    raw = np.load(source_file, allow_pickle=True)
    if "mocap_framerate" not in raw:
        raise ValueError("missing mocap_framerate")
    framerate = float(np.asarray(raw["mocap_framerate"]).item())
    stride = 2 if framerate == 120 else 1 if framerate == 60 else round(framerate / 60)
    if stride < 1:
        raise ValueError(f"unsupported mocap framerate: {framerate}")

    poses_full = torch.as_tensor(raw["poses"][::stride], dtype=torch.float32)
    poses = poses_full[:, :66]
    trans = torch.as_tensor(raw["trans"][::stride], dtype=torch.float32)
    if poses.shape[0] < 5:
        raise ValueError(f"sequence is too short: {poses.shape[0]} frames")

    body_params = {
        "root_orient": poses[:, :3].clone(),
        "pose_body": poses[:, 3:66].clone(),
        "trans": trans.clone(),
    }
    with torch.no_grad():
        body_pose_world = body_model(**{key: value.to(device) for key, value in body_params.items()})

    frame_count = poses.shape[0]
    rotation_local_6d = utils_transform.aa2sixd(poses.reshape(-1, 3)).reshape(frame_count, -1)
    rotation_local_full = rotation_local_6d[1:].float().cpu()

    rotation_local_mat = aa2matrot(poses_full.reshape(-1, 3)).reshape(frame_count, -1, 9)
    rotation_global_mat = local2global_pose(
        rotation_local_mat, body_model.kintree_table[0].long().cpu()
    ).reshape(frame_count, -1, 3, 3)
    head_rotation = rotation_global_mat[:, 15]
    rotation_global_6d = utils_transform.matrot2sixd(
        rotation_global_mat.reshape(-1, 3, 3)
    ).reshape(frame_count, -1, 6)
    input_rotation = rotation_global_6d[1:, :22].float().cpu()

    rotation_velocity_mat = torch.matmul(
        torch.linalg.inv(rotation_global_mat[:-1]), rotation_global_mat[1:]
    )
    rotation_velocity_6d = utils_transform.matrot2sixd(
        rotation_velocity_mat.reshape(-1, 3, 3)
    ).reshape(frame_count - 1, -1, 6)
    input_rotation_velocity = rotation_velocity_6d[:, :22].float().cpu()

    positions = body_pose_world.Jtr[:, :22].detach().float().cpu()
    head_transform = torch.eye(4, dtype=torch.float32).repeat(frame_count, 1, 1)
    head_transform[:, :3, :3] = head_rotation.float().cpu()
    head_transform[:, :3, 3] = positions[:, 15]

    num_frames = frame_count - 1
    position_velocity = positions[1:] - positions[:-1]
    hmd_full = torch.cat(
        [
            input_rotation.reshape(num_frames, -1),
            input_rotation_velocity.reshape(num_frames, -1),
            positions[1:].reshape(num_frames, -1),
            position_velocity.reshape(num_frames, -1),
        ],
        dim=-1,
    ).float()

    return {
        "rotation_local_full_gt_list": rotation_local_full,
        "hmd_position_global_full_gt_list": hmd_full,
        "body_parms_list": {key: value.float().cpu() for key, value in body_params.items()},
        "head_global_trans_list": head_transform[1:].float(),
        "position_global_full_gt_world": positions[1:].float(),
        "framerate": 60,
        "gender": scalar_string(raw["gender"] if "gender" in raw else "unknown"),
        "filepath": PurePosixPath(source_relative).as_posix(),
    }


def validate_existing(path, expected_source):
    try:
        record = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        record = torch.load(path, map_location="cpu")
    actual = normalize_source_path(record.get("filepath", ""))
    if actual != expected_source:
        raise ValueError(f"Existing output metadata mismatch: {path}: {actual} != {expected_source}")
    return int(record["rotation_local_full_gt_list"].shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", "--root-dir", type=Path, required=True)
    parser.add_argument("--cfg", type=Path)
    parser.add_argument("--support_dir", "--support-dir", type=Path)
    parser.add_argument("--save_dir", "--save-dir", type=Path)
    parser.add_argument("--split", dest="split_files", type=Path, action="append")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-skips", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    if args.support_dir is None:
        args.support_dir = Path(cfg.SUPPORT_DIR) if cfg is not None else None
    if args.save_dir is None:
        args.save_dir = Path(cfg.DATASET_PATH) if cfg is not None else None
    if args.support_dir is None or args.save_dir is None:
        parser.error("--support-dir and --save-dir are required when --cfg is omitted")

    if args.split_files:
        split_files = args.split_files
        manifest_name = "manifest_custom.jsonl"
    elif cfg is not None:
        split_files = (
            protocol_split_files(cfg.PROTOCOL, "train")
            + protocol_split_files(cfg.PROTOCOL, "test")
        )
        manifest_name = f"manifest_{cfg.PROTOCOL}.jsonl"
    else:
        parser.error("provide at least one --split or a --cfg with PROTOCOL")

    split_files = [Path(path) for path in split_files]
    sources = read_sources(split_files)
    body_model = create_body_model(args.support_dir, torch.device(args.device))
    manifest = []
    failures = []

    for source in tqdm(sources, desc="Preprocessing AMASS"):
        source_file = args.root_dir.joinpath(*PurePosixPath(source).parts)
        output_file = args.save_dir.joinpath(*PurePosixPath(source).with_suffix(".pt").parts)
        try:
            if not source_file.is_file():
                raise FileNotFoundError(source_file)
            if output_file.is_file() and not args.overwrite:
                frames = validate_existing(output_file, source)
            else:
                record = preprocess_sequence(source_file, source, body_model, torch.device(args.device))
                output_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = output_file.with_name(output_file.name + ".tmp")
                torch.save(record, temporary)
                temporary.replace(output_file)
                frames = int(record["rotation_local_full_gt_list"].shape[0])
            manifest.append({"source": source, "processed": output_file.relative_to(args.save_dir).as_posix(),
                             "frames": frames})
        except Exception as error:
            failures.append({"source": source, "error": f"{type(error).__name__}: {error}"})

    args.save_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = args.save_dir / manifest_name
    manifest_file.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in manifest))
    if failures:
        failure_file = args.save_dir / (manifest_file.stem + "_failures.jsonl")
        failure_file.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in failures))
        if not args.allow_skips:
            raise RuntimeError(f"Failed to preprocess {len(failures)} files; see {failure_file}")
    print(f"processed={len(manifest)} failed={len(failures)} manifest={manifest_file}")


if __name__ == "__main__":
    main()
