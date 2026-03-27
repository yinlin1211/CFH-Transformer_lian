"""
CFT v6 训练脚本 — 修复 Transformer PE 语义错误版
==============================

基于 v5 的核心修复：
  [修复1] FH Transformer 加入 temporal_embed（论文公式2：ε(t)）
  [修复2] HT Transformer 加入 freq_embed（论文公式3：H(f)）
  [修复3] TF Transformer 加入 harm_embed（论文公式4：T(h)）
  [修复4] dataset.py 加入 15% 负样本

评估逻辑（继承自 v3）：
  - frames_to_notes 的 onset 分支允许 2 帧间隙
  - compute_note_f1_single 使用 mir_eval transcription.evaluate()
  - pitch 转 Hz，onset_tolerance=50ms，pitch_tolerance=50cents
  - ref 直接从 JSON 标注读取

调度器：CosineAnnealingLR（无 warmup）
"""

import argparse
import logging
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import json
import yaml

try:
    import mir_eval
    from mir_eval import transcription as mir_transcription, util as mir_util
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False
    print("WARNING: mir_eval not found, F1 metrics will be 0")

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from model import CFT_v6, CFTLoss
from dataset import MIR_ST500_Dataset, MIDI_MIN, NUM_PITCHES


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(run_dir / 'train_stdout.log', mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 音符级 F1 评估（v3：对齐原论文 evaluate_github.py）
# ---------------------------------------------------------------------------
# 修正内容（相比 v2）：
#   1. frames_to_notes 的 onset 分支允许 2 帧间隙，避免长音符被抖动截断
#   2. compute_note_f1_single 改用 transcription.evaluate() 一次调用：
#      - pitch 转 Hz（原来直接用 MIDI 编号导致 pitch 距离计算错误）
#      - COn = Onset_F-measure（只看 onset，不看 pitch）
#      - COnP = F-measure_no_offset（onset + pitch）
#      - COnPOff = F-measure（onset + pitch + offset）
#   3. ref 直接从 JSON 标注读取，不再从帧标签反推
# ---------------------------------------------------------------------------

def frames_to_notes(frame_pred, onset_pred, hop_length, sample_rate,
                    onset_thresh=0.5, frame_thresh=0.5, min_note_len=2):
    """帧级预测 → 音符列表，返回 (intervals, pitches)，pitches 为 MIDI 编号"""
    frame_time = hop_length / sample_rate
    T, P = frame_pred.shape
    intervals = []
    pitches = []

    for p in range(P):
        midi = p + MIDI_MIN
        onset_frames = np.where(onset_pred[:, p] > onset_thresh)[0]

        if len(onset_frames) == 0:
            # 纯帧模式
            active = frame_pred[:, p] > frame_thresh
            in_note, note_start = False, 0
            for t in range(T):
                if active[t] and not in_note:
                    in_note, note_start = True, t
                elif not active[t] and in_note:
                    in_note = False
                    if t - note_start >= min_note_len:
                        intervals.append([note_start * frame_time, t * frame_time])
                        pitches.append(float(midi))
            if in_note and T - note_start >= min_note_len:
                intervals.append([note_start * frame_time, T * frame_time])
                pitches.append(float(midi))
        else:
            # onset 引导模式：允许最多 2 帧间隙，避免长音符被抖动截断
            for i, f_on in enumerate(onset_frames):
                next_onset = onset_frames[i + 1] if i + 1 < len(onset_frames) else T
                f_off, gap = f_on, 0
                for t in range(f_on, min(next_onset, T)):
                    if frame_pred[t, p] > frame_thresh:
                        f_off = t
                        gap = 0
                    else:
                        gap += 1
                        if gap > 2 and t > f_on + 1:
                            break
                if f_off - f_on + 1 >= min_note_len:
                    intervals.append([f_on * frame_time, (f_off + 1) * frame_time])
                    pitches.append(float(midi))

    if len(intervals) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    return np.array(intervals), np.array(pitches, dtype=float)


def compute_note_f1_single(pred_intervals, pred_pitches_midi,
                            ref_intervals, ref_pitches_midi,
                            onset_tolerance=0.05):
    """
    对齐原论文 evaluate_github.py 的评估逻辑。
    输入 pitches 为 MIDI 编号，内部转 Hz 后调用 transcription.evaluate()。
    返回 (COn_f1, COnP_f1, COnPOff_f1)
    """
    if not HAS_MIR_EVAL:
        return 0.0, 0.0, 0.0

    # 过滤 duration <= 0 的音符
    valid_pred = pred_intervals[:, 1] - pred_intervals[:, 0] > 0
    valid_ref  = ref_intervals[:, 1]  - ref_intervals[:, 0]  > 0
    pred_intervals  = pred_intervals[valid_pred]
    pred_pitches_midi = pred_pitches_midi[valid_pred]
    ref_intervals   = ref_intervals[valid_ref]
    ref_pitches_midi  = ref_pitches_midi[valid_ref]

    if len(ref_intervals) == 0:
        return None, None, None
    if len(pred_intervals) == 0:
        return 0.0, 0.0, 0.0

    # pitch 转 Hz（mir_eval 要求 Hz 输入）
    pred_pitches_hz = mir_util.midi_to_hz(pred_pitches_midi)
    ref_pitches_hz  = mir_util.midi_to_hz(ref_pitches_midi)

    try:
        raw = mir_transcription.evaluate(
            ref_intervals, ref_pitches_hz,
            pred_intervals, pred_pitches_hz,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=50,
        )
        con_f1     = raw['Onset_F-measure']
        conp_f1    = raw['F-measure_no_offset']
        conpoff_f1 = raw['F-measure']
    except Exception:
        con_f1 = conp_f1 = conpoff_f1 = 0.0

    return con_f1, conp_f1, conpoff_f1


# ---------------------------------------------------------------------------
# 训练 epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, epoch, logger,
                grad_clip=1.0, max_batches=None, scaler=None):
    model.train()
    total_loss = 0.0
    onset_loss_sum = 0.0
    frame_loss_sum = 0.0
    offset_loss_sum = 0.0
    n_batches = min(len(loader), max_batches) if max_batches else len(loader)

    for batch_idx, (cqt, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        cqt = cqt.to(device)
        onset_label = labels['onset'].to(device)
        frame_label = labels['frame'].to(device)
        offset_label = labels['offset'].to(device)

        optimizer.zero_grad()
        with autocast():
            onset_pred, frame_pred, offset_pred = model(cqt)
            loss, onset_loss, frame_loss, offset_loss = criterion(
                onset_pred, frame_pred, offset_pred,
                onset_label, frame_label, offset_label
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        onset_loss_sum += onset_loss.item()
        frame_loss_sum += frame_loss.item()
        offset_loss_sum += offset_loss.item()

        if (batch_idx + 1) % max(1, n_batches // 3) == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{n_batches}] "
                f"loss={loss.item():.4f} "
                f"onset={onset_loss.item():.4f} "
                f"frame={frame_loss.item():.4f} "
                f"offset={offset_loss.item():.4f}"
            )

    return {
        'total': total_loss / n_batches,
        'onset': onset_loss_sum / n_batches,
        'frame': frame_loss_sum / n_batches,
        'offset': offset_loss_sum / n_batches,
    }


# ---------------------------------------------------------------------------
# 验证（全曲评估）
# ---------------------------------------------------------------------------

def validate_full_song(model, val_dataset, criterion, device, hop_length, sample_rate,
                       onset_thresh=0.5, frame_thresh=0.5, infer_chunk=256,
                       gt_annotations=None):
    """
    全曲验证。
    gt_annotations: dict {song_id: [[onset, offset, midi], ...]}，用于 ref 音符。
    如果为 None，则从帧标签反推（不推荐）。
    """
    model.eval()
    total_loss = 0.0
    n_songs = 0
    con_f1_list = []
    conp_f1_list = []
    conpoff_f1_list = []
    onset_sig_list = []
    frame_sig_list = []

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            cqt, labels, song_id = val_dataset[idx]
            F_bins, T_total = cqt.shape

            onset_lbl = labels['onset'].numpy()
            frame_lbl = labels['frame'].numpy()
            offset_lbl = labels['offset'].numpy()

            onset_sig_chunks = []
            frame_sig_chunks = []
            offset_sig_chunks = []
            chunk_losses = []

            for start in range(0, T_total, infer_chunk):
                end = min(start + infer_chunk, T_total)
                cqt_chunk = cqt[:, start:end].unsqueeze(0).to(device)

                chunk_T = end - start
                if chunk_T < infer_chunk:
                    pad_len = infer_chunk - chunk_T
                    cqt_chunk = torch.nn.functional.pad(cqt_chunk, (0, pad_len))

                onset_pred, frame_pred, offset_pred = model(cqt_chunk)

                onset_pred = onset_pred[:, :chunk_T, :]
                frame_pred = frame_pred[:, :chunk_T, :]
                offset_pred = offset_pred[:, :chunk_T, :]

                ol_chunk = torch.from_numpy(onset_lbl[start:end]).unsqueeze(0).to(device)
                fl_chunk = torch.from_numpy(frame_lbl[start:end]).unsqueeze(0).to(device)
                ofl_chunk = torch.from_numpy(offset_lbl[start:end]).unsqueeze(0).to(device)
                loss, _, _, _ = criterion(onset_pred, frame_pred, offset_pred,
                                          ol_chunk, fl_chunk, ofl_chunk)
                chunk_losses.append(loss.item())

                onset_sig_chunks.append(torch.sigmoid(onset_pred[0]).cpu().numpy())
                frame_sig_chunks.append(torch.sigmoid(frame_pred[0]).cpu().numpy())
                offset_sig_chunks.append(torch.sigmoid(offset_pred[0]).cpu().numpy())

            onset_sig = np.concatenate(onset_sig_chunks, axis=0)
            frame_sig = np.concatenate(frame_sig_chunks, axis=0)

            total_loss += float(np.mean(chunk_losses))
            n_songs += 1

            onset_sig_list.append(onset_sig.mean())
            frame_sig_list.append(frame_sig.mean())

            # 预测音符
            pred_intervals, pred_pitches = frames_to_notes(
                frame_sig, onset_sig, hop_length, sample_rate, onset_thresh, frame_thresh
            )

            # ref 音符：优先从 JSON 标注读取，否则从帧标签反推
            if gt_annotations is not None and song_id in gt_annotations:
                raw = gt_annotations[song_id]
                ref_notes = [[float(n[0]), float(n[1]), float(n[2])] for n in raw
                             if float(n[1]) - float(n[0]) > 0]
                if len(ref_notes) == 0:
                    continue
                ref_intervals = np.array([[n[0], n[1]] for n in ref_notes])
                ref_pitches   = np.array([n[2] for n in ref_notes])
            else:
                # 备用：从帧标签反推（不推荐，仅当没有 JSON 时使用）
                ref_intervals, ref_pitches = frames_to_notes(
                    frame_lbl.astype(np.float32), onset_lbl.astype(np.float32),
                    hop_length, sample_rate, onset_thresh=0.5, frame_thresh=0.5
                )

            if len(ref_intervals) == 0:
                continue

            con_f1, conp_f1, conpoff_f1 = compute_note_f1_single(
                pred_intervals, pred_pitches,
                ref_intervals, ref_pitches
            )
            if con_f1 is not None:
                con_f1_list.append(con_f1)
                conp_f1_list.append(conp_f1)
                conpoff_f1_list.append(conpoff_f1)

    avg_loss = total_loss / max(n_songs, 1)
    avg_con_f1 = float(np.mean(con_f1_list)) if con_f1_list else 0.0
    avg_conp_f1 = float(np.mean(conp_f1_list)) if conp_f1_list else 0.0
    avg_conpoff_f1 = float(np.mean(conpoff_f1_list)) if conpoff_f1_list else 0.0
    avg_onset_sig = float(np.mean(onset_sig_list)) if onset_sig_list else 0.0
    avg_frame_sig = float(np.mean(frame_sig_list)) if frame_sig_list else 0.0

    return avg_loss, avg_con_f1, avg_conp_f1, avg_conpoff_f1, avg_onset_sig, avg_frame_sig


def find_best_threshold(model, val_dataset, criterion, device, hop_length, sample_rate,
                        logger, gt_annotations=None):
    n_search = min(10, len(val_dataset))
    best_conp = 0.0
    best_ot, best_ft = 0.3, 0.3

    model.eval()
    infer_chunk = 256
    preds = []  # (frame_sig, onset_sig, ref_intervals, ref_pitches)
    with torch.no_grad():
        for idx in range(n_search):
            cqt, labels, song_id = val_dataset[idx]
            T_total = cqt.shape[1]
            onset_chunks, frame_chunks = [], []
            for start in range(0, T_total, infer_chunk):
                end = min(start + infer_chunk, T_total)
                chunk_T = end - start
                cqt_chunk = cqt[:, start:end].unsqueeze(0).to(device)
                if chunk_T < infer_chunk:
                    cqt_chunk = torch.nn.functional.pad(cqt_chunk, (0, infer_chunk - chunk_T))
                op, fp, _ = model(cqt_chunk)
                onset_chunks.append(torch.sigmoid(op[0, :chunk_T]).cpu().numpy())
                frame_chunks.append(torch.sigmoid(fp[0, :chunk_T]).cpu().numpy())
            onset_sig = np.concatenate(onset_chunks, axis=0)
            frame_sig = np.concatenate(frame_chunks, axis=0)

            # ref 音符：优先从 JSON 标注读取
            if gt_annotations is not None and song_id in gt_annotations:
                raw = gt_annotations[song_id]
                ref_notes = [[float(n[0]), float(n[1]), float(n[2])] for n in raw
                             if float(n[1]) - float(n[0]) > 0]
                if len(ref_notes) == 0:
                    continue
                ref_intervals = np.array([[n[0], n[1]] for n in ref_notes])
                ref_pitches   = np.array([n[2] for n in ref_notes])
            else:
                onset_lbl = labels['onset'].numpy()
                frame_lbl = labels['frame'].numpy()
                ref_intervals, ref_pitches = frames_to_notes(
                    frame_lbl.astype(np.float32), onset_lbl.astype(np.float32),
                    hop_length, sample_rate, onset_thresh=0.5, frame_thresh=0.5
                )
            preds.append((frame_sig, onset_sig, ref_intervals, ref_pitches))

    onset_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    frame_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

    for ot in onset_thresholds:
        for ft in frame_thresholds:
            conp_list = []
            for frame_sig, onset_sig, ref_intervals, ref_pitches in preds:
                pred_intervals, pred_pitches = frames_to_notes(
                    frame_sig, onset_sig, hop_length, sample_rate, ot, ft
                )
                if len(ref_intervals) == 0:
                    continue
                _, conp, _ = compute_note_f1_single(
                    pred_intervals, pred_pitches, ref_intervals, ref_pitches
                )
                if conp is not None:
                    conp_list.append(conp)
            if conp_list:
                avg_conp = np.mean(conp_list)
                if avg_conp > best_conp:
                    best_conp = avg_conp
                    best_ot, best_ft = ot, ft

    logger.info(f"  Threshold search: best on={best_ot:.2f}, fr={best_ft:.2f}, COnP_f1={best_conp:.4f}")
    return best_ot, best_ft


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 按启动时间生成独立 run 目录，格式：run/<时间戳>_COnP/
    # 每次启动都会创建全新子目录，多个进程并发训练时互不干扰
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{run_timestamp}_COnP"
    run_base = Path(config['training'].get('run_dir', './run'))
    run_dir = run_base / run_name
    save_dir = run_dir / 'checkpoints'
    log_dir  = run_dir / 'logs'
    run_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pid_file = Path(f'/tmp/cft_v6_COnP_{run_timestamp}.pid')
    pid_file.write_text(str(os.getpid()))

    logger = setup_logger(log_dir)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")
    logger.info("=" * 60)
    logger.info("CFT v6 — 修复 Transformer 位置编码语义错误版")
    logger.info("=" * 60)

    # 加载 JSON 标注（用于验证时 ref 音符）
    gt_json_path = config['data']['label_path']
    with open(gt_json_path) as f:
        gt_annotations = json.load(f)
    logger.info(f"Loaded GT annotations from {gt_json_path}")

    # 数据集
    train_dataset = MIR_ST500_Dataset(config, split='train')
    val_dataset = MIR_ST500_Dataset(config, split='val')
    logger.info(f"Train samples: {len(train_dataset)}, Val songs: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # 模型
    model = CFT_v6(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    logger.info("Tokenization: continuous kernels [3,5,7] (paper-aligned, no dilation)")

    # 混合精度
    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler is not None:
        logger.info("Mixed precision (AMP) enabled")

    # 损失函数（论文公式1：均等权重 BCE）
    criterion = CFTLoss(
        onset_weight=config['loss']['onset_weight'],
        frame_weight=config['loss']['frame_weight'],
        offset_weight=config['loss']['offset_weight'],
    ).to(device)

    # 优化器（论文 Section 3.3：Adam, lr=3e-4）
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    # 学习率调度：论文未提及，使用 CosineAnnealingLR（无 warmup）
    total_epochs = config['training']['epochs']
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6
    )
    logger.info(f"Scheduler: CosineAnnealingLR (no warmup, T_max={total_epochs})")

    writer = None
    if HAS_TB:
        writer = SummaryWriter(str(log_dir / 'tensorboard'))

    start_epoch = 1
    best_conp_f1 = 0.0
    best_onset_thresh = 0.3
    best_frame_thresh = 0.3

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_conp_f1 = ckpt.get('best_conp_f1', 0.0)
        best_onset_thresh = ckpt.get('best_onset_thresh', 0.3)
        best_frame_thresh = ckpt.get('best_frame_thresh', 0.3)
        logger.info(f"Resumed from epoch {ckpt['epoch']}, best_COnP_f1={best_conp_f1:.4f}")

    hop_length = config['audio']['hop_length']
    sample_rate = config['data']['sample_rate']

    max_samples = config['data'].get('max_samples_per_epoch', None)
    max_batches = None
    if max_samples:
        max_batches = max(1, max_samples // config['training']['batch_size'])
        logger.info(f"Max batches per epoch: {max_batches}")

    for epoch in range(start_epoch, total_epochs + 1):
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger,
            grad_clip=config['training']['grad_clip'],
            max_batches=max_batches,
            scaler=scaler
        )
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        if epoch % 5 == 0 or epoch == 1:
            best_onset_thresh, best_frame_thresh = find_best_threshold(
                model, val_dataset, criterion, device, hop_length, sample_rate,
                logger, gt_annotations=gt_annotations
            )

        val_loss, con_f1, conp_f1, conpoff_f1, onset_sig, frame_sig = validate_full_song(
            model, val_dataset, criterion, device, hop_length, sample_rate,
            onset_thresh=best_onset_thresh, frame_thresh=best_frame_thresh,
            gt_annotations=gt_annotations
        )

        logger.info(
            f"Epoch {epoch}/{total_epochs} | "
            f"train_loss={train_losses['total']:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"COn_f1={con_f1:.4f} | "
            f"COnP_f1={conp_f1:.4f} | "
            f"COnPOff_f1={conpoff_f1:.4f} | "
            f"sig_onset={onset_sig:.4f} sig_frame={frame_sig:.4f} | "
            f"thresh(on={best_onset_thresh:.2f},fr={best_frame_thresh:.2f}) | "
            f"lr={lr:.2e}"
        )

        if writer:
            writer.add_scalar('Loss/train', train_losses['total'], epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/COn_f1', con_f1, epoch)
            writer.add_scalar('Metrics/COnP_f1', conp_f1, epoch)
            writer.add_scalar('Metrics/COnPOff_f1', conpoff_f1, epoch)
            writer.add_scalar('LR', lr, epoch)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'val_loss': val_loss,
            'COn_f1': con_f1,
            'COnP_f1': conp_f1,
            'COnPOff_f1': conpoff_f1,
            'best_conp_f1': best_conp_f1,
            'best_onset_thresh': best_onset_thresh,
            'best_frame_thresh': best_frame_thresh,
            'config': config
        }

        if conp_f1 > best_conp_f1:
            best_conp_f1 = conp_f1
            ckpt['best_conp_f1'] = best_conp_f1
            torch.save(ckpt, save_dir / 'best_model.pt')
            logger.info(f"  -> Best model saved! COnP_f1={best_conp_f1:.4f}")

        if epoch % config['training']['save_every'] == 0:
            torch.save(ckpt, save_dir / f'checkpoint_epoch{epoch:04d}.pt')

        torch.save(ckpt, save_dir / 'latest.pt')

    if writer:
        writer.close()

    logger.info(f"Training complete! Best COnP_f1: {best_conp_f1:.4f}")
    pid_file.unlink(missing_ok=True)
    logger.info(f"Run directory: {run_dir}")


if __name__ == '__main__':
    main()
