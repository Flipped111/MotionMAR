import os
from argparse import ArgumentParser
from yacs.config import CfgNode as CN


def get_args(include_inference=False, default_cfg="VQVAE/config_vqvae/vqvae_S1.yaml"):
    parser = ArgumentParser(description='Train Motion Capture Network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=default_cfg,
                        type=str)
    if include_inference:
        parser.add_argument('--decode',
                            choices=('argmax', 'soft', 'sample'),
                            default=None,
                            help='token decoding mode')
        parser.add_argument('--cfg_scale', '--cfg-scale',
                            type=float,
                            default=None,
                            help='classifier-free guidance scale')
        parser.add_argument('--tau',
                            type=float,
                            default=None,
                            help='soft decoding temperature')
        parser.add_argument('--soft_top_k', '--soft-top-k',
                            type=int,
                            default=None,
                            help='soft decoding top-k; 0 uses the full vocabulary')
    args = parser.parse_args()
    if not args.cfg:
        raise ValueError("Please specify a config file with --cfg.")
    print(f"using config {args.cfg}")
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.cfg)
    name = os.path.splitext(os.path.basename(args.cfg))[0]
    if "SAVE_DIR" not in cfg or not cfg.SAVE_DIR:
        cfg.SAVE_DIR = os.path.join("outputs", name)
    if include_inference:
        if "INFERENCE" not in cfg:
            cfg.INFERENCE = CN(new_allowed=True)
        else:
            cfg.INFERENCE.set_new_allowed(True)
        cli_values = {
            "DECODE": args.decode,
            "CFG_SCALE": args.cfg_scale,
            "TAU": args.tau,
            "SOFT_TOP_K": args.soft_top_k,
        }
        for key, value in cli_values.items():
            if value is not None:
                cfg.INFERENCE[key] = value
    return cfg


def get_inference_kwargs(cfg):
    inference = cfg.INFERENCE if "INFERENCE" in cfg else None
    decode = inference.DECODE if inference is not None and "DECODE" in inference else "argmax"
    cfg_scale = inference.CFG_SCALE if inference is not None and "CFG_SCALE" in inference else 1.0
    tau = inference.TAU if inference is not None and "TAU" in inference else 1.0
    soft_top_k = inference.SOFT_TOP_K if inference is not None and "SOFT_TOP_K" in inference else 0

    if decode not in ("argmax", "soft", "sample"):
        raise ValueError(f"Unsupported INFERENCE.DECODE: {decode}")
    if tau <= 0:
        raise ValueError("INFERENCE.TAU must be greater than 0.")
    if soft_top_k < 0:
        raise ValueError("INFERENCE.SOFT_TOP_K must be greater than or equal to 0.")
    return {
        "cfg_scale": float(cfg_scale),
        "decode": str(decode),
        "tau": float(tau),
        "soft_top_k": int(soft_top_k),
    }
