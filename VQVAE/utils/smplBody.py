import os
import torch
from human_body_prior.body_model.body_model import BodyModel as BM


class BodyModel(torch.nn.Module):
    def __init__(self, support_dir):
        super().__init__()
        subject_gender = "male"
        bm_fname = os.path.join(
            support_dir, "smplh/{}/model.npz".format(subject_gender)
        )
        dmpl_fname = os.path.join(
            support_dir, "dmpls/{}/model.npz".format(subject_gender)
        )
        num_betas = 16  # number of body parameters
        num_dmpls = 8  # number of DMPL parameters
        self.body_model = BM(
            bm_fname=bm_fname,
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_fname,
        )

    def forward(self, body_params):
        body_pose = self.body_model(
            **{
                k: v
                for k, v in body_params.items()
                if k in ["pose_body", "trans", "root_orient"]
            }
        )
        return body_pose
