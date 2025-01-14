import copy
import hydra
import torch
import random
import numpy as np
from pathlib import Path
from typing import List
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
            )
        )

    return callbacks


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2


def get_watermarking_pattern(pipe, watermark, device, shape=None):
    set_random_seed(watermark.w_seed)
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        gt_init = pipe.get_random_latents()
    if 'seed_ring' in watermark.w_pattern:
        gt_patch = gt_init
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(watermark.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in watermark.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in watermark.w_pattern:
        gt_patch = gt_init
    elif 'rand' in watermark.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in watermark.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in watermark.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += watermark.w_pattern_const
    elif 'ring' in watermark.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(watermark.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    else:
        raise NotImplementedError(f'w_pattern: {watermark.w_mask_shape}')

    return gt_patch


def get_watermarking_mask(init_latents_w, watermark, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if watermark.w_mask_shape == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r=watermark.w_radius)
        torch_mask = torch.tensor(np_mask).to(device)
        if watermark.w_channel == -1:
            watermarking_mask[:, :] = torch_mask  # all channels
        else:
            watermarking_mask[:, watermark.w_channel] = torch_mask
    elif watermark.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if watermark.w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p - watermark.w_radius:anchor_p + watermark.w_radius,
            anchor_p - watermark.w_radius:anchor_p + watermark.w_radius] = True
        else:
            watermarking_mask[:, watermark.w_channel, anchor_p - watermark.w_radius:anchor_p + watermark.w_radius,
            anchor_p - watermark.w_radius:anchor_p + watermark.w_radius] = True
    elif watermark.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {watermark.w_mask_shape}')

    return watermarking_mask


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, watermark):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    if watermark.w_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif watermark.w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f'w_injection: {watermark.w_injection}')

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, watermark):
    if 'complex' in watermark.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif 'seed' in watermark.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {watermark.w_measurement}')

    if 'l1' in watermark.w_measurement:
        no_w_metric = torch.abs(
            reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f'w_measurement: {watermark.w_measurement}')

    return no_w_metric, w_metric
