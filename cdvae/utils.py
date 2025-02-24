import copy
import hydra
import torch
import torch.fft as fft
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


def get_watermarking_pattern(signal_length=256, num_patterns=5, seed=42):
    """
    生成水印模式，选择特定的低频分量位置

    Args:
        signal_length: 信号长度
        num_patterns: 要选择的模式数量
        seed: 随机种子，用于复现结果

    Returns:
        频域中的模式位置列表，和对应的幅值列表
    """
    torch.manual_seed(seed)

    # 只考虑低频部分，设定阈值为信号长度的20%
    low_freq_threshold = int(signal_length * 0.2)

    # 生成候选的低频位置（排除直流分量0）
    positions = torch.arange(1, low_freq_threshold)

    # 随机选择指定数量的位置
    selected_positions = positions[torch.randperm(len(positions))[:num_patterns]]
    selected_positions, _ = torch.sort(selected_positions)  # 排序以保持一定的顺序性

    # 为每个选定位置生成随机幅值（在0.1到0.3之间）
    amplitudes = 0.1 + 0.2 * torch.rand(num_patterns)

    return selected_positions, amplitudes


def add_watermark(latent_code, positions, amplitudes, strength=1.0):
    """
    在频域中添加水印

    Args:
        latent_code: 输入的潜码，形状为 (batch_size, 256)
        positions: 要添加水印的频率位置
        amplitudes: 每个位置的幅值
        strength: 水印强度系数

    Returns:
        添加了水印的潜码
    """
    positions = positions.to(latent_code.device)
    amplitudes = amplitudes.to(latent_code.device)

    freq_domain = fft.fft(latent_code, dim=1)

    for pos, amp in zip(positions, amplitudes):
        # 在正频率和负频率对称位置都添加
        freq_domain[:, pos] += strength * amp
        freq_domain[:, -pos] += strength * amp.conj()  # 共轭以保持实数性质

    watermarked = fft.ifft(freq_domain, dim=1).real

    return watermarked


def detect_watermark(latent_code, positions, amplitudes, threshold=0.8):
    """
    检测水印

    Args:
        latent_code: 待检测的潜码
        positions: 水印的频率位置
        amplitudes: 期望的幅值
        threshold: 检测阈值

    Returns:
        是否检测到水印，以及相似度分数
    """
    batch_size, signal_length = latent_code.shape
    device = latent_code.device

    # 将位置和幅值移到正确的设备
    positions = positions.to(device)
    amplitudes = amplitudes.to(device)

    # 应用低通滤波器
    freq_domain = fft.fft(latent_code, dim=1)

    # 创建低通滤波器（保留20%的低频）
    cutoff = int(signal_length * 0.2)
    low_pass = torch.zeros(signal_length, device=device)
    low_pass[:cutoff] = 1
    low_pass[-cutoff:] = 1

    # 应用滤波器
    freq_domain = freq_domain * low_pass.unsqueeze(0)

    # 检查指定位置的幅值
    detected_amplitudes = torch.zeros_like(amplitudes)
    for i, pos in enumerate(positions):
        detected_amplitudes[i] = torch.abs(freq_domain[:, pos]).mean()

    # 计算与期望幅值的余弦相似度
    similarity = torch.nn.functional.cosine_similarity(
        detected_amplitudes.unsqueeze(0),
        amplitudes.unsqueeze(0)
    ).item()

    # 根据阈值判断是否检测到水印
    is_watermarked = similarity > threshold

    return is_watermarked, similarity
