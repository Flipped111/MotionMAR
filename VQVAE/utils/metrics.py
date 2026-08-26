# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Metric functions with same inputs

import torch
from VQVAE.utils import utils_transform


def pred_jitter(predicted_position, predicted_angle, predicted_root_angle, gt_position,
                gt_angle, gt_root_angle, fps, ):
    pred_jitter = (
        ((predicted_position[3:] - 3 * predicted_position[2:-1] +
          3 * predicted_position[1:-2] - predicted_position[:-3]) * (fps ** 3)).norm(dim=2).mean()
    )
    return pred_jitter


def gt_jitter(predicted_position, predicted_angle, predicted_root_angle, gt_position, gt_angle,
              gt_root_angle, fps, ):
    gt_jitter = (
        (
                (gt_position[3:] - 3 * gt_position[2:-1] +
                 3 * gt_position[1:-2] - gt_position[:-3]) * (fps ** 3)).norm(dim=2).mean()
    )
    return gt_jitter


def mpjre(predicted_position, predicted_angle, predicted_root_angle, gt_position, gt_angle,
          gt_root_angle, fps, ):
    diff = gt_angle - predicted_angle
    diff = utils_transform.matrot2aa(utils_transform.aa2matrot(diff.reshape(-1, 3)))
    rot_error = torch.mean(torch.absolute(diff))
    return rot_error


def rootre(predicted_position, predicted_angle, predicted_root_angle, gt_position, gt_angle,
           gt_root_angle, fps, ):
    diff = gt_root_angle - predicted_root_angle
    diff = utils_transform.matrot2aa(utils_transform.aa2matrot(diff))
    rot_error = torch.mean(torch.absolute(diff))
    return rot_error


def mpjpe(predicted_position, predicted_angle, predicted_root_angle, gt_position, gt_angle,
          gt_root_angle, fps, ):
    pos_error = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1)))
    return pos_error


def handpe(predicted_position, predicted_angle, predicted_root_angle, gt_position,
           gt_angle, gt_root_angle, fps, ):
    pos_error_hands = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[..., [20, 21]]
    )
    return pos_error_hands


def upperpe(predicted_position, predicted_angle, predicted_root_angle, gt_position,
            gt_angle, gt_root_angle, fps, ):
    upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    upper_body_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[..., upper_index]
    )
    return upper_body_error


def lowerpe(predicted_position, predicted_angle, predicted_root_angle, gt_position,
            gt_angle, gt_root_angle, fps, ):
    lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[..., lower_index]
    )
    return lower_body_error


def rootpe(predicted_position, predicted_angle, predicted_root_angle, gt_position,
           gt_angle, gt_root_angle, fps, ):
    pos_error_root = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[..., [0]]
    )
    return pos_error_root


def mpjve(predicted_position, predicted_angle, predicted_root_angle, gt_position,
          gt_angle, gt_root_angle, fps, ):
    gt_velocity = (gt_position[1:, ...] - gt_position[:-1, ...]) * fps
    predicted_velocity = (predicted_position[1:, ...] - predicted_position[:-1, ...]) * fps
    vel_error = torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity - predicted_velocity), axis=-1)))
    return vel_error


metric_funcs_dict = {
    "mpjre": mpjre,
    "rootre": rootre,
    "mpjpe": mpjpe,
    "mpjve": mpjve,
    "handpe": handpe,
    "upperpe": upperpe,
    "lowerpe": lowerpe,
    "rootpe": rootpe,
    "pred_jitter": pred_jitter,
    "gt_jitter": gt_jitter,
}


def get_metric_function(metric):
    return metric_funcs_dict[metric]
