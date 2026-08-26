import copy
import os
import random
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import torch.multiprocessing as mp
from VQVAE.utils import utils_transform

from VQVAE.parser_util import get_args
from VQVAE.dataloader.dataloader import get_dataloader, load_data, TrainDataset
from VQVAE.dataloader.sparse_utils import sparse_head_hand_indices
from VQVAE.transformer_vqvae import TransformerVQVAE
from VQVAE.utils.smplBody import BodyModel
from VQVAE.test_vqvae import test_process
torch.autograd.set_detect_anomaly(True)


def loss_function(args, recover_6d, motion, loss_z, bodymodel, gt_pos):
    """
    recover_6d: [B, T, 22, 6]
    motion:     [B, T, 22, 6]
    gt_pos:     [B, T, J, 3] (sparse positions)
    """
    loss_func = torch.nn.SmoothL1Loss(reduction='mean')

    B, T, J,C = recover_6d.shape

    # 1. Reconstruction loss (full sequence)
    if args.ROOTLOSS:
        rec_root = loss_func(recover_6d[:, :, 0], motion[:, :, 0])
        rec_other = loss_func(recover_6d[:, :, 1:], motion[:, :, 1:])
        recons_loss = rec_root * 0.2 + rec_other
    else:
        recons_loss = loss_func(recover_6d, motion)

    # 2. Velocity loss for temporal smoothness.
    pred_vel = recover_6d[:, 1:] - recover_6d[:, :-1]
    gt_vel = motion[:, 1:] - motion[:, :-1]
    velocity_loss = loss_func(pred_vel, gt_vel)

    # 3. FK loss: flatten B*T because BodyModel expects [N, ...].
    pred_aa = utils_transform.sixd2aa(recover_6d.view(-1, 22, 6), batch=True).reshape(B*T, -1)
    gt_aa = utils_transform.sixd2aa(motion.view(-1, 22, 6), batch=True).reshape(B*T, -1)

    pred_output = bodymodel({
        "root_orient": pred_aa[:, :3],
        "pose_body": pred_aa[:, 3:]
    })
    gt_output = bodymodel({
        "root_orient": gt_aa[:, :3],
        "pose_body": gt_aa[:, 3:]
    })

    pred_loc = pred_output.Jtr[:, :22, :]
    gt_loc = gt_output.Jtr[:, :22, :]

    fk_loss = loss_func(pred_loc, gt_loc)

    # 4. Hand-align loss: align predicted skeleton to GT head, then compare hands.
    pred_loc = pred_loc.view(B, T, 22, 3)

    head_idx, hand_idx = sparse_head_hand_indices(args.SPARSE_DIM)
    gt_head_pos = gt_pos[:, :, head_idx]
    head2root = pred_loc[:, :, 15].clone()

    root_trans = gt_head_pos - head2root
    global_pos_pred = root_trans[:, :, None, :] + pred_loc

    hand_align_loss = loss_func(global_pos_pred[:, :, [20, 21]], gt_pos[:, :, hand_idx])

    vq_loss = torch.mean(loss_z)
    loss_w = args.LOSS

    vel_weight = getattr(loss_w, 'velocity_loss', 10.0)

    loss_all = (recons_loss +
                vq_loss * loss_w.alpha_codebook +
                fk_loss * loss_w.fk_loss +
                hand_align_loss * loss_w.hand_align_loss +
                velocity_loss * vel_weight)

    loss = {
        "loss": loss_all,
        "vq_loss": vq_loss,
        "recons_loss": recons_loss,
        "fk_loss": fk_loss,
        "hand_align_loss": hand_align_loss,
        "velocity_loss": velocity_loss
    }
    return loss


def save_checkpoint(states, output_dir):
    checkpoint_file = os.path.join(output_dir, "checkpoint.pth.tar")
    torch.save(states, checkpoint_file)
    if True:
        torch.save(
            states['state_dict'],
            os.path.join(output_dir, 'best.pth.tar')
        )


def do_train(args, model, train_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    begin_epoch = 0
    output_dir = args.SAVE_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, 'checkpoint.pth.tar')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.LR, betas=(0.9, 0.99), weight_decay=args.WEIGHT_DECAY)
    if os.path.exists(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.MILESTONES, gamma=1 / 4, last_epoch=begin_epoch if begin_epoch else -1)

    body_model = BodyModel(args.SUPPORT_DIR).to(device)
    
    model.train()
    
    for epoch in range(begin_epoch, args.EPOCH):
        tqdm.write(f"Starting epoch {epoch}")
        tqdm.write(f"current lr:{scheduler.get_last_lr()}")
        train_dataloader = tqdm(train_dataloader, dynamic_ncols=True)
        for motion, sparse in train_dataloader:
            bs, seq = motion.shape[:2]
            motion = motion.to(device)
            sparse = sparse.to(device)
            motion_input = copy.deepcopy(motion)  # (bs, 20, 396)
            motion_input = motion_input.reshape(bs, seq, 22, 6)
            motion = motion.reshape(bs, seq, 22, 6)

            recover_6d, usages ,loss_z= model(x=motion_input.reshape(bs, seq,-1),sparse=sparse.reshape(bs,seq,-1))  # (bs, 20, 132)``
            loss = loss_function(args, recover_6d, motion, loss_z,
                                 body_model, sparse[:, :, :, 12:15])  # units: cm
            optimizer.zero_grad()
            loss["loss"].backward()
            optimizer.step()
            formatted_usages = ", ".join([f"{u:.2f}" if isinstance(u, (int, float)) else str(u) for u in usages])
            train_dataloader.set_description(f"e:{epoch},rc:{loss['recons_loss']:.2e},vq:{loss['vq_loss']:.2e},"
                                             f"fk:{loss['fk_loss']:.2e},hd:{loss['hand_align_loss']:.2e},"
                                             f"usages:{formatted_usages}")

        scheduler.step()
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, output_dir)

        test_process()
        train_dataloader.close()


def main():
    args = get_args()
    torch.backends.cudnn.benchmark = False
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    if args.SAVE_DIR is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR)

    motions, sparses, all_info = load_data(
        args.DATASET_PATH,
        "train",
        protocol=args.PROTOCOL,
        input_motion_length=args.INPUT_MOTION_LENGTH,
    )
    train_dataset = TrainDataset(
        motions,
        sparses,
        all_info,
        args.INPUT_MOTION_LENGTH,
        args.TRAIN_DATASET_REPEAT_TIMES,
        sparse_dim=args.SPARSE_DIM,
    )
    train_dataloader = get_dataloader(
        train_dataset, "train", batch_size=args.BATCH_SIZE, num_workers=args.NUM_WORKERS
    )

    print("creating model...")
    print(f"{args.SAVE_DIR}")
    body_part_name = args.part

    vqcfg = args.VQVAE
    patch_nums = tuple(args.VAR.PATCH_NUMS)
    model = TransformerVQVAE(in_dim=vqcfg.in_dim, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim, heads=vqcfg.heads,
                             dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook, n_e=vqcfg.n_e, e_dim=vqcfg.e_dim,
                             beta=vqcfg.beta, sparse_dim=args.SPARSE_DIM, patch_nums=patch_nums)

    print("Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training...")
    do_train(args, model, train_dataloader)
    print("Done.")


if __name__ == '__main__':
    mp.set_sharing_strategy('file_system')
    main()
