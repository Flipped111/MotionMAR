"""Train the temporal refiner on VAR predictions."""

import os
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from accelerate import Accelerator

from VQVAE.parser_util import get_args, get_inference_kwargs
from VQVAE.dataloader.dataloader_refiner import load_data, TrainDataset, get_dataloader
from VQVAE.dataloader.sparse_utils import sparse_head_hand_indices
from VQVAE.utils.smplBody import BodyModel
from VQVAE.utils import utils_transform
from models import build_vae_var
from refinenet import Refinenet

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_var(args):
    """Load the pretrained VAR model."""
    vae_ckpt = args.VAR.VAE_CKPT
    var_ckpt = args.VAR.CKPT
    
    patch_nums = tuple(args.VAR.PATCH_NUMS)
    vqcfg = args.VQVAE
    vae, var = build_vae_var(
        device,
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
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)
    print(f'VAR model loaded.')
    return var.to(device)


def calc_rot_loss(pred_6d, gt_6d):
    """Rotation loss that directly optimises MPJRE."""
    pred_aa = utils_transform.sixd2aa(pred_6d.reshape(-1, 6)).reshape(-1, 22, 3)
    gt_aa = utils_transform.sixd2aa(gt_6d.reshape(-1, 6)).reshape(-1, 22, 3)
    diff = gt_aa - pred_aa
    # Wrap to [-pi, pi] to match the MPJRE definition in metrics.py.
    diff = utils_transform.matrot2aa(utils_transform.aa2matrot(diff.reshape(-1, 3)))
    return torch.mean(torch.abs(diff))


def calc_fk_loss(body_model, recover, gt, gt_pos, sparse_dim=54):
    """Compute FK loss, jitter loss, and hand-align loss."""
    recover = recover.reshape(-1, 22, 6)
    gt = gt.reshape(-1, 22, 6)
    pred_aa = utils_transform.sixd2aa(recover, batch=True).flatten(1, 2)
    gt_aa = utils_transform.sixd2aa(gt, batch=True).flatten(1, 2)

    pred_loc = body_model({
        "root_orient": pred_aa[:, :3],
        "pose_body": pred_aa[:, 3:]
    }).Jtr[:, :22, :]

    gt_loc = body_model({
        "root_orient": gt_aa[:, :3],
        "pose_body": gt_aa[:, 3:]
    }).Jtr[:, :22, :]

    head_idx, _ = sparse_head_hand_indices(sparse_dim)
    gt_head_pos = gt_pos[:, head_idx]
    head2root = pred_loc[:, 15].clone()
    root_trans = gt_head_pos - head2root
    global_pos = root_trans[:, None] + pred_loc

    fk_loss = torch.mean(torch.norm((pred_loc - gt_loc).reshape(-1, 3), p=2, dim=1))

    pred_jitter = ((pred_loc[3:] - 3 * pred_loc[2:-1] +
                    3 * pred_loc[1:-2] - pred_loc[:-3])).norm(dim=2).mean()

    root_trans_gt = -gt_loc[:, 15] + gt_head_pos
    global_pos_gt = gt_loc + root_trans_gt[:, None]

    align_loss = F.smooth_l1_loss(global_pos[:, [15, 20, 21]], global_pos_gt[:, [15, 20, 21]])

    return pred_jitter, fk_loss, align_loss


@torch.no_grad()
def generate_var_sequence(var_model, sparse, inference_kwargs, batch_size):
    """Generate one prediction per sliding window without retaining VAR activations."""
    outputs = []
    for start in range(0, sparse.shape[0], batch_size):
        sparse_batch = sparse[start:start + batch_size]
        sample = var_model.autoregressive_infer_cfg(
            B=sparse_batch.shape[0],
            sparse_seq=sparse_batch,
            **inference_kwargs,
        )
        outputs.append(sample[:, -1])
    return torch.cat(outputs, dim=0)


def do_train(args, var_model, refinenet, dataloader, body_model, inference_kwargs):
    accelerator = Accelerator(mixed_precision='fp16')

    optimizer = AdamW(refinenet.parameters(), lr=args.LR, weight_decay=args.WEIGHT_DECAY)
    begin_epoch = 0
    output_dir = args.SAVE_DIR

    refiner_model_file = os.path.join(output_dir, 'checkpoint.pth.tar')
    if os.path.exists(refiner_model_file):
        refiner_checkpoint = torch.load(refiner_model_file, map_location="cpu")
        saved_var_ckpt = refiner_checkpoint.get("var_ckpt")
        saved_inference = refiner_checkpoint.get("inference")
        if saved_var_ckpt is not None and saved_var_ckpt != args.VAR.CKPT:
            raise ValueError(
                f"Refiner checkpoint uses VAR {saved_var_ckpt}, not {args.VAR.CKPT}."
            )
        if saved_inference is not None and saved_inference != inference_kwargs:
            raise ValueError(
                f"Refiner checkpoint uses inference settings {saved_inference}, "
                f"not {inference_kwargs}."
            )
        begin_epoch = refiner_checkpoint['epoch']
        optimizer.load_state_dict(refiner_checkpoint['optimizer'])
        refinenet.load_state_dict(refiner_checkpoint['state_dict'])
        print(f"=> Resumed from epoch {begin_epoch}")

    var_model.eval()
    refinenet.train()

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.MILESTONES, gamma=args.GAMMA,
        last_epoch=begin_epoch if begin_epoch else -1
    )

    optimizer, dataloader, lr_scheduler, body_model, refinenet = accelerator.prepare(
        optimizer, dataloader, lr_scheduler, body_model, refinenet
    )

    loss_weight = args.REFINER.loss_weight
    l1loss = torch.nn.SmoothL1Loss()

    for epoch in range(begin_epoch, args.EPOCH):
        tqdm.write(f"Starting epoch {epoch}, lr: {lr_scheduler.get_lr()[0]}")
        train_dataloader = tqdm(dataloader, dynamic_ncols=True)

        for motion_132, sparse in train_dataloader:
            motion_132 = motion_132[0].to(device)
            sparse = sparse[0].to(device)

            bs, seq = motion_132.shape[:2]
            optimizer.zero_grad()

            recover_6d = generate_var_sequence(
                var_model,
                sparse,
                inference_kwargs,
                args.NUM_PER_BATCH,
            )

            recover_6d = recover_6d.reshape(1, -1, 132).float()
            pred_final, _ = refinenet(recover_6d, None)
            # reconstruction loss
            recons_loss = l1loss(pred_final, motion_132[None, :, -1])

            pred_final_sq = pred_final.squeeze(0)
            motion_gt = motion_132[:, -1]

            # rotation loss (directly optimises MPJRE)
            rot_loss = calc_rot_loss(pred_final_sq, motion_gt)

            # velocity loss
            vel_loss = l1loss(pred_final_sq[1:] - pred_final_sq[:-1], motion_gt[1:] - motion_gt[:-1])
            vel_loss2 = l1loss(pred_final_sq[3::3] - pred_final_sq[:-3:3], motion_gt[3::3] - motion_gt[:-3:3])

            # FK loss, jitter loss, hand align loss
            pred_jitter, fk_loss, hand_align_loss = calc_fk_loss(
                body_model, pred_final_sq, motion_gt, sparse[:, -1, :, 12:15], args.SPARSE_DIM
            )

            loss = (recons_loss * loss_weight.recons +
                    rot_loss * loss_weight.rot +
                    vel_loss * loss_weight.vel_1 +
                    vel_loss2 * loss_weight.vel_2 +
                    fk_loss * loss_weight.fk_loss +
                    hand_align_loss * loss_weight.hand_align +
                    pred_jitter * loss_weight.jitter)

            accelerator.backward(loss)

            train_dataloader.set_description(
                f"e:{epoch}, rec:{recons_loss:.2e}, rot:{rot_loss:.2e}, fk:{fk_loss:.2e}"
            )

            optimizer.step()

        lr_scheduler.step()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(refinenet)
            accelerator.save({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'var_ckpt': args.VAR.CKPT,
                'inference': inference_kwargs,
            }, os.path.join(output_dir, "checkpoint.pth.tar"))
            accelerator.save(state_dict, os.path.join(output_dir, "best.pth.tar"))

        print(f"Epoch {epoch} finished.")


def main():
    args = get_args(
        include_inference=True,
        default_cfg="VQVAE/config_vqvae/refiner_S1.yaml",
    )
    inference_kwargs = get_inference_kwargs(args)

    torch.backends.cudnn.benchmark = False
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)

    os.makedirs(args.SAVE_DIR, exist_ok=True)
    print(f"Saving to: {args.SAVE_DIR}")
    print(f"Inference settings: {inference_kwargs}")

    motions, sparses, all_info = load_data(
        args.DATASET_PATH, "train",
        protocol=args.PROTOCOL,
        input_motion_length=args.INPUT_MOTION_LENGTH,
    )

    train_dataset = TrainDataset(
        motions, sparses,
        input_motion_length=args.INPUT_MOTION_LENGTH,
        full_motion_len=args.FULL_MOTION_LENGTH,
        sparse_dim=args.SPARSE_DIM
    )

    train_dataloader = get_dataloader(train_dataset, "train", batch_size=1, num_workers=args.NUM_WORKERS)

    var_model = load_var(args)

    refinenet = Refinenet(n_layers=args.REFINER.n_layers, hidden_dim=args.REFINER.hidden_dim)
    refinenet = refinenet.to(device)

    body_model = BodyModel(args.SUPPORT_DIR).to(device)

    print("Training...")
    do_train(args, var_model, refinenet, train_dataloader, body_model, inference_kwargs)


if __name__ == "__main__":
    main()
