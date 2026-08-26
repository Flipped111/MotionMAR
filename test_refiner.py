"""Evaluate VAR output after offline temporal refinement."""

import os
import math
import random
import numpy as np
import torch
from tqdm import tqdm
from VQVAE.utils import utils_transform
from VQVAE.utils.metrics import get_metric_function
from VQVAE.parser_util import get_args, get_inference_kwargs
from VQVAE.dataloader.dataloader_refiner import load_data, TestDataset
from VQVAE.utils.smplBody import BodyModel
from models import build_vae_var
device = "cuda" if torch.cuda.is_available() else "cpu"
from refinenet import Refinenet

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

pred_metrics = [
    "mpjre",
    "rootre",
    "mpjpe",
    "mpjve",
    "handpe",
    "upperpe",
    "lowerpe",
    "rootpe",
    "pred_jitter",
]
gt_metrics = [
    "gt_jitter",
]
all_metrics = pred_metrics + gt_metrics

metrics_coeffs = {
    "mpjre": RADIANS_TO_DEGREES,
    "rootre": RADIANS_TO_DEGREES,
    "mpjpe": METERS_TO_CENTIMETERS,
    "mpjve": METERS_TO_CENTIMETERS,
    "handpe": METERS_TO_CENTIMETERS,
    "upperpe": METERS_TO_CENTIMETERS,
    "lowerpe": METERS_TO_CENTIMETERS,
    "rootpe": METERS_TO_CENTIMETERS,
    "pred_jitter": 1.0,
    "gt_jitter": 1.0,
    "gt_mpjpe": METERS_TO_CENTIMETERS,
    "gt_mpjve": METERS_TO_CENTIMETERS,
    "gt_handpe": METERS_TO_CENTIMETERS,
    "gt_rootpe": METERS_TO_CENTIMETERS,
}


def load_var(args):
    vae_ckpt = args.VAR.VAE_CKPT
    var_ckpt = args.VAR.CKPT
    patch_nums = tuple(args.VAR.PATCH_NUMS)
    if 'vae' not in globals() or 'var' not in globals():
        vqcfg = args.VQVAE
        vae, var = build_vae_var(device,
            V=vqcfg.n_e, Cvae=vqcfg.e_dim, patch_nums=patch_nums, share_quant_resi=4,
            vqvae_in_dim=vqcfg.in_dim, vqvae_n_layers=tuple(vqcfg.n_layers),
            vqvae_hid_dim=vqcfg.hid_dim, vqvae_heads=vqcfg.heads,
            vqvae_dropout=vqcfg.dropout, vqvae_n_codebook=vqcfg.n_codebook,
            vqvae_beta=vqcfg.beta, sparse_input_dim=args.SPARSE_DIM,
            depth=args.VAR.DEPTH
        )

    var_checkpoint = torch.load(var_ckpt, map_location="cpu")
    if "epoch" in var_checkpoint:
        print(f"Loaded VAR checkpoint at epoch: {var_checkpoint['epoch']}")
    temp2 = var_checkpoint["trainer"]["var_wo_ddp"]
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    var.load_state_dict(temp2, strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    return var.to(device)


def overlapping_refine(args, data, model, refinenet, inference_kwargs, num_per_batch=16):
    sparse_original, body_param, head_motion, filename = (data[1], data[2], data[3], data[4])
    num_frames = head_motion.shape[0]
    sparse = sparse_original.to(device).float().reshape(num_frames, args.SPARSE_DIM)
    head_motion = head_motion.to(device).float()

    sparse_splits = []
    block_seq = args.INPUT_MOTION_LENGTH  # 32
    sparse_pad = sparse[:1].repeat(block_seq - 1, 1)
    sparse_pad = torch.cat((sparse_pad, sparse), dim=0)

    for i in range(num_frames):
        sparse_splits.append(sparse_pad[i: i + block_seq])

    sparse_splits = torch.stack(sparse_splits)

    n_steps = sparse_splits.shape[0] // num_per_batch
    if len(sparse_splits) % num_per_batch > 0:
        n_steps += 1

    output_samples = []
    num_joints = 22

    for step_index in range(n_steps):
        sparse_per_batch = sparse_splits[step_index * num_per_batch: (step_index + 1) * num_per_batch].to(device)
        with torch.no_grad():
            bs, seq = sparse_per_batch.shape[:2]
            sample = model.autoregressive_infer_cfg(
                B=bs,
                sparse_seq=sparse_per_batch,
                **inference_kwargs,
            )
        sample = sample[:, -1].reshape(-1, num_joints * 6)
        output_samples.append(sample.cpu().float())

    initial_res = torch.cat(output_samples, dim=0).to(device)
    with torch.no_grad():
        refined, _ = refinenet(initial_res[None], None)
    return refined.squeeze(0).cpu(), body_param, head_motion, filename

def evaluate_prediction(args, metrics, sample, body_model, head_motion, body_param, fps, filename, sample_index):
    seq = sample.shape[0]
    motion_pred = sample.squeeze().to(device)
    model_rot_input = (
        utils_transform.sixd2aa(motion_pred.reshape(-1, 6).detach()).reshape(motion_pred.shape[0], -1).float()
    )
    for k, v in body_param.items():
        body_param[k] = v.squeeze().to(device)
        body_param[k] = body_param[k][-model_rot_input.shape[0]:, ...]

    T_head2world = head_motion.clone().to(device)
    t_head2world = T_head2world[:, :3, 3].clone()

    pred_temp = torch.zeros((seq, 22, 3), device=device)
    pred_temp = model_rot_input.reshape((seq, -1, 3))

    pred_temp = pred_temp.reshape((seq, -1))
    body_pose_local = body_model(
        {
            "pose_body": pred_temp[..., 3:],
            "root_orient": pred_temp[..., :3],
        }
    ).Jtr

    t_head2root = -body_pose_local[:, 15, :]
    t_root2world = t_head2root + t_head2world.to(device)
    predicted_body = body_model(
        {
            "pose_body": pred_temp[..., 3:],
            "root_orient": pred_temp[..., :3],
            "trans": t_root2world,
        }
    )

    predicted_position = predicted_body.Jtr[:, :22, :]

    gt_pose = torch.cat((body_param["root_orient"], body_param["pose_body"]), dim=-1).reshape((seq, -1, 3))
    gt_pose_temp = torch.zeros((seq, 22, 3), device=device)
    gt_pose_temp= gt_pose
    gt_pose_temp = gt_pose_temp.reshape((seq, -1))
    gt_body = body_model({
        "pose_body": gt_pose_temp[..., 3:],
        "root_orient": gt_pose_temp[..., :3],
        "trans": body_param["trans"]
    })


    gt_position = gt_body.Jtr[:, :22, :]
    gt_root_angle = body_param["root_orient"]
    predicted_root_angle = pred_temp[..., :3]
    eval_log = {}
    for metric in metrics:
        eval_log[metric] = (
            get_metric_function(metric)(
                predicted_position,
                pred_temp,
                predicted_root_angle,
                gt_position,
                gt_pose_temp,
                gt_root_angle,
                fps,
            ).cpu().numpy()
        )

    torch.cuda.empty_cache()
    return eval_log


def test_process():
    args = get_args(
        include_inference=True,
        default_cfg="VQVAE/config_vqvae/refiner_S1.yaml",
    )
    inference_kwargs = get_inference_kwargs(args)
    torch.backends.cudnn.benchmark = False
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)

    fps = args.FPS
    body_model = BodyModel(args.SUPPORT_DIR).to(device)
    print("Loading dataset...")
    filename_list, all_info = load_data(
        args.DATASET_PATH,
        "test",
        protocol=args.PROTOCOL,
        input_motion_length=args.INPUT_MOTION_LENGTH,
    )
    dataset = TestDataset(all_info, filename_list, sparse_dim=args.SPARSE_DIM)

    log = {}
    for metric in all_metrics:
        log[metric] = 0

    model=load_var(args)

    refinenet = Refinenet(n_layers=args.REFINER.n_layers, hidden_dim=args.REFINER.hidden_dim)
    refinenet = refinenet.to(device)

    refine_file = args.REFINER.CKPT
    checkpoint_file = args.REFINER.CHECKPOINT
    if os.path.exists(refine_file):
        print("=> loading refine model '{}'".format(refine_file))
        refine_checkpoint = torch.load(refine_file, map_location="cpu")
        if "state_dict" in refine_checkpoint:
            refine_checkpoint = refine_checkpoint["state_dict"]
        refinenet.load_state_dict(refine_checkpoint)

        if os.path.exists(checkpoint_file):
            ckpt_info = torch.load(checkpoint_file, map_location='cpu')
            print(f"=> Model from epoch {ckpt_info.get('epoch', 'unknown')}")
            saved_var_ckpt = ckpt_info.get("var_ckpt")
            saved_inference = ckpt_info.get("inference")
            if saved_var_ckpt is not None and saved_var_ckpt != args.VAR.CKPT:
                raise ValueError(
                    f"Refiner checkpoint uses VAR {saved_var_ckpt}, not {args.VAR.CKPT}."
                )
            if saved_inference is not None and saved_inference != inference_kwargs:
                raise ValueError(
                    f"Refiner checkpoint uses inference settings {saved_inference}, "
                    f"not {inference_kwargs}."
                )
    else:
        raise FileNotFoundError(f"Refiner checkpoint not found: {refine_file}")
    refinenet.eval()
    for p in refinenet.parameters():
        p.requires_grad_(False)

    print(f"Inference settings: {inference_kwargs}")

    n_testframe = args.NUM_PER_BATCH
    for sample_index in tqdm(range(len(dataset))):
        output, body_param, head_motion, filename = \
        overlapping_refine(
            args, dataset[sample_index], model, refinenet,
            inference_kwargs, n_testframe)

        instance_log = evaluate_prediction(
            args, all_metrics, output, body_model, head_motion,
            body_param, fps, filename, sample_index)

        for key in instance_log:
            log[key] += instance_log[key]

    print("Metrics for the predictions")
    for metric in pred_metrics:
        print(metric, log[metric] / len(dataset) * metrics_coeffs[metric])
    print("Metrics for the ground truth")
    for metric in gt_metrics:
        print(metric, log[metric] / len(dataset) * metrics_coeffs[metric])


if __name__ == "__main__":
    test_process()
