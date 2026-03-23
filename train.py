"""
CFT v2 训练脚本 — 严格对齐论文

变更：
  - 使用 CFT_v2 模型（TRIAD Tokenization + 单层 Linear 输出头）
  - 标准 BCE 损失，均等权重
  - 音高范围 N=48（C2~B5，MIDI 36~83）
  - 全曲验证 + 音符级 F1 评估
  - 优化器修正：AdamW → Adam（对齐论文 Section 3.3）
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import yaml

try:
    import mir_eval
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False
    print("WARNING: mir_eval not found, F1 metrics will be 0")

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from model_v2 import CFT_v2, CFTLoss
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


def setup_logger(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_dir / 'train_stdout.log', mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 音符级 F1 评估
# ---------------------------------------------------------------------------

def frames_to_notes(frame_pred, onset_pred, hop_length, sample_rate,
                    onset_thresh=0.5, frame_thresh=0.5, min_note_len=2):
    """
    将帧级预测转换为音符事件列表。

    frame_pred:  (T, 48) numpy array, sigmoid 输出
    onset_pred:  (T, 48) numpy array, sigmoid 输出

    返回:
        intervals: (N, 2) array of [onset_time, offset_time]
        pitches:   (N,) array of MIDI pitch numbers (36~83)
    """
    frame_time = hop_length / sample_rate
    T, P = frame_pred.shape

    intervals = []
    pitches = []

    for p in range(P):
        midi = p + MIDI_MIN  # C2=36

        onset_frames = np.where(onset_pred[:, p] > onset_thresh)[0]

        if len(onset_frames) == 0:
            active = frame_pred[:, p] > frame_thresh
            in_note = False
            note_start = 0
            for t in range(T):
                if active[t] and not in_note:
                    in_note = True
                    note_start = t
                elif not active[t] and in_note:
                    in_note = False
                    note_len = t - note_start
                    if note_len >= min_note_len:
                        intervals.append([note_start * frame_time, t * frame_time])
                        pitches.append(float(midi))
            if in_note:
                note_len = T - note_start
                if note_len >= min_note_len:
                    intervals.append([note_start * frame_time, T * frame_time])
                    pitches.append(float(midi))
        else:
            for i, f_on in enumerate(onset_frames):
                next_onset = onset_frames[i + 1] if i + 1 < len(onset_frames) else T
                f_off = f_on
                for t in range(f_on, min(next_onset, T)):
                    if frame_pred[t, p] > frame_thresh:
                        f_off = t
                    else:
                        if t > f_on + 1:
                            break
                note_len = f_off - f_on + 1
                if note_len >= min_note_len:
                    intervals.append([f_on * frame_time, (f_off + 1) * frame_time])
                    pitches.append(float(midi))

    if len(intervals) == 0:
        return np.zeros((0, 2)), np.zeros(0)

    return np.array(intervals), np.array(pitches, dtype=float)


def compute_note_f1_single(frame_pred, onset_pred, frame_label, onset_label,
                            hop_length, sample_rate, onset_thresh=0.5, frame_thresh=0.5):
    """对单首歌计算音符级 F1。返回 (COn_f1, COnP_f1, COnPOff_f1)。"""
    if not HAS_MIR_EVAL:
        return 0.0, 0.0, 0.0

    pred_intervals, pred_pitches = frames_to_notes(
        frame_pred, onset_pred, hop_length, sample_rate, onset_thresh, frame_thresh
    )
    ref_intervals, ref_pitches = frames_to_notes(
        frame_label, onset_label, hop_length, sample_rate,
        onset_thresh=0.5, frame_thresh=0.5
    )

    if len(ref_intervals) == 0:
        return None, None, None

    if len(pred_intervals) == 0:
        return 0.0, 0.0, 0.0

    try:
        _, _, con_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            pred_intervals, pred_pitches,
            onset_tolerance=0.05, pitch_tolerance=0.0,
            offset_ratio=None, offset_min_tolerance=0.05
        )
    except Exception:
        con_f1 = 0.0

    try:
        _, _, conp_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            pred_intervals, pred_pitches,
            onset_tolerance=0.05, pitch_tolerance=50.0,
            offset_ratio=None, offset_min_tolerance=0.05
        )
    except Exception:
        conp_f1 = 0.0

    try:
        _, _, conpoff_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            pred_intervals, pred_pitches,
            onset_tolerance=0.05, pitch_tolerance=50.0,
            offset_ratio=0.2, offset_min_tolerance=0.05
        )
    except Exception:
        conpoff_f1 = 0.0

    return con_f1, conp_f1, conpoff_f1


# ---------------------------------------------------------------------------
# 训练 epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, epoch, logger,
                grad_clip=1.0, max_batches=None):
    model.train()
    total_loss = 0.0
    onset_loss_sum = 0.0
    frame_loss_sum = 0.0
    offset_loss_sum = 0.0
    n_batches = min(len(loader), max_batches) if max_batches else len(loader)

    for batch_idx, (cqt, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        cqt = cqt.to(device)                          # (B, 288, T)
        onset_label = labels['onset'].to(device)       # (B, T, 48)
        frame_label = labels['frame'].to(device)
        offset_label = labels['offset'].to(device)

        optimizer.zero_grad()
        onset_pred, frame_pred, offset_pred = model(cqt)  # 各 (B, T, 48)

        loss, onset_loss, frame_loss, offset_loss = criterion(
            onset_pred, frame_pred, offset_pred,
            onset_label, frame_label, offset_label
        )

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
                       onset_thresh=0.5, frame_thresh=0.5, infer_chunk=256):
    """对验证集中每首歌进行分段推理，拼接后计算音符级 F1。"""
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
            cqt, labels, song_id = val_dataset[idx]  # cqt: (288, T)
            F_bins, T_total = cqt.shape

            onset_lbl = labels['onset'].numpy()   # (T, 48)
            frame_lbl = labels['frame'].numpy()
            offset_lbl = labels['offset'].numpy()

            onset_sig_chunks = []
            frame_sig_chunks = []
            offset_sig_chunks = []
            chunk_losses = []

            for start in range(0, T_total, infer_chunk):
                end = min(start + infer_chunk, T_total)
                cqt_chunk = cqt[:, start:end].unsqueeze(0).to(device)  # (1, 288, chunk_T)

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

            onset_sig = np.concatenate(onset_sig_chunks, axis=0)   # (T, 48)
            frame_sig = np.concatenate(frame_sig_chunks, axis=0)

            total_loss += float(np.mean(chunk_losses))
            n_songs += 1

            onset_sig_list.append(onset_sig.mean())
            frame_sig_list.append(frame_sig.mean())

            con_f1, conp_f1, conpoff_f1 = compute_note_f1_single(
                frame_sig, onset_sig, frame_lbl, onset_lbl,
                hop_length, sample_rate, onset_thresh, frame_thresh
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


def find_best_threshold(model, val_dataset, criterion, device, hop_length, sample_rate, logger):
    """在验证集上搜索最优阈值。"""
    n_search = min(10, len(val_dataset))
    best_conp = 0.0
    best_ot, best_ft = 0.3, 0.3

    model.eval()
    infer_chunk = 256
    preds = []
    with torch.no_grad():
        for idx in range(n_search):
            cqt, labels, _ = val_dataset[idx]
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
            onset_lbl = labels['onset'].numpy()
            frame_lbl = labels['frame'].numpy()
            preds.append((frame_sig, onset_sig, frame_lbl, onset_lbl))

    onset_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    frame_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

    for ot in onset_thresholds:
        for ft in frame_thresholds:
            conp_list = []
            for fp, op, fl, ol in preds:
                _, conp, _ = compute_note_f1_single(
                    fp, op, fl, ol, hop_length, sample_rate, ot, ft
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
    save_dir = Path(config['training']['save_dir'])
    log_dir = Path(config['training']['log_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pid_file = Path('/tmp/cft_train.pid')
    pid_file.write_text(str(os.getpid()))

    logger = setup_logger(log_dir)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")

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

    # 模型（CFT_v2）
    model = CFT_v2(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # 损失函数（标准 BCE，均等权重）
    criterion = CFTLoss().to(device)

    # 优化器（论文 Section 3.3：使用标准 Adam）
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    # 学习率调度：warmup + cosine annealing
    warmup_epochs = config['training'].get('warmup_epochs', 10)
    total_epochs = config['training']['epochs']

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    writer = None
    if HAS_TB:
        writer = SummaryWriter(log_dir / datetime.now().strftime('%Y%m%d_%H%M%S'))

    start_epoch = 1
    best_conp_f1 = 0.0
    best_onset_thresh = 0.3
    best_frame_thresh = 0.3

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
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

    logger.info("=" * 60)
    logger.info("Starting CFT v2 training (paper-aligned)")
    logger.info("=" * 60)

    for epoch in range(start_epoch, total_epochs + 1):
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger,
            grad_clip=config['training']['grad_clip'],
            max_batches=max_batches
        )
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        if epoch % 5 == 0 or epoch == 1:
            best_onset_thresh, best_frame_thresh = find_best_threshold(
                model, val_dataset, criterion, device, hop_length, sample_rate, logger
            )

        val_loss, con_f1, conp_f1, conpoff_f1, onset_sig, frame_sig = validate_full_song(
            model, val_dataset, criterion, device, hop_length, sample_rate,
            onset_thresh=best_onset_thresh, frame_thresh=best_frame_thresh
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
            writer.add_scalar('Sigmoid/onset', onset_sig, epoch)
            writer.add_scalar('Sigmoid/frame', frame_sig, epoch)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
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


if __name__ == '__main__':
    main()
