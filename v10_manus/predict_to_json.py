"""
CFT v9_manus 推理脚本：将模型预测结果输出为 JSON 文件
输出格式与原论文 evaluate_github.py 完全兼容：
  {"401": [[onset_sec, offset_sec, midi_pitch], ...], "402": [...], ...}

用法：
  cd /mnt/ssd/lian/论文复现/CFH-Transformer_v2

  python predict_to_json.py \
      --config config.yaml \
      --checkpoint checkpoints_v2/best_model.pt \
      --split test \
      --onset_thresh 0.15 \
      --frame_thresh 0.3 \
      --output pred_test_v2.json

评估（直接调用原论文脚本）：
  python evaluate_github.py \
      /mnt/ssd/lian/论文复现/CFH-Transformer/MIR-ST500_corrected.json \
      pred_test_v2.json \
      0.05
"""
import sys
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

# 关键：优先加载本目录的 model.py
sys.path.insert(0, str(Path(__file__).parent))
from model import CFT_v9 as CFT

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

MIDI_MIN = 36  # C2，与训练时一致


# ---------------------------------------------------------------------------
# 后处理：帧级预测 -> 音符列表
# 返回格式：[[onset_sec, offset_sec, midi_pitch], ...]
# ---------------------------------------------------------------------------

def frames_to_notes(frame_pred, onset_pred, hop_length, sample_rate,
                    onset_thresh=0.5, frame_thresh=0.5, min_note_len=2):
    """
    帧级概率图 -> 音符列表 [[onset_sec, offset_sec, midi_pitch], ...]

    onset 分支：onset 触发音符起点，frame 概率追踪终点，允许最多 2 帧间隙
    无 onset 分支：纯帧级连续区间追踪
    """
    frame_time = hop_length / sample_rate
    T, P = frame_pred.shape
    notes = []

    for p in range(P):
        midi = p + MIDI_MIN
        onset_frames = np.where(onset_pred[:, p] > onset_thresh)[0]

        if len(onset_frames) == 0:
            # 无 onset：纯帧级追踪
            active = frame_pred[:, p] > frame_thresh
            in_note, note_start = False, 0
            for t in range(T):
                if active[t] and not in_note:
                    in_note, note_start = True, t
                elif not active[t] and in_note:
                    in_note = False
                    if t - note_start >= min_note_len:
                        notes.append([note_start * frame_time,
                                      t * frame_time,
                                      float(midi)])
            if in_note and T - note_start >= min_note_len:
                notes.append([note_start * frame_time,
                               T * frame_time,
                               float(midi)])
        else:
            # 有 onset：onset 触发，frame 追踪结束，允许 2 帧间隙
            for i, f_on in enumerate(onset_frames):
                next_onset = onset_frames[i + 1] if i + 1 < len(onset_frames) else T
                f_off = f_on
                gap = 0
                for t in range(f_on, min(next_onset, T)):
                    if frame_pred[t, p] > frame_thresh:
                        f_off = t
                        gap = 0
                    else:
                        gap += 1
                        if gap > 2 and t > f_on + 1:
                            break
                if f_off - f_on + 1 >= min_note_len:
                    notes.append([f_on * frame_time,
                                  (f_off + 1) * frame_time,
                                  float(midi)])

    return notes  # [[onset_sec, offset_sec, midi_pitch], ...]


# ---------------------------------------------------------------------------
# 推理：npy -> 概率图
# ---------------------------------------------------------------------------

def predict_from_npy(model, npy_path, config, device):
    """从 npy CQT 文件推理，返回 (frame_pred, onset_pred)，shape=(T, 48)"""
    cqt = np.load(npy_path)                                              # (F, T)
    cqt_tensor = torch.from_numpy(cqt).float().unsqueeze(0).to(device)  # (1, F, T)
    segment_frames = config['data']['segment_frames']
    T = cqt.shape[1]

    onset_map = np.zeros((T, 48), dtype=np.float32)
    frame_map = np.zeros((T, 48), dtype=np.float32)
    count_map = np.zeros(T,       dtype=np.float32)
    step = segment_frames // 2   # 50% 重叠，减少边界效应

    model.eval()
    with torch.no_grad():
        for start in range(0, T, step):
            end = start + segment_frames
            seg = cqt_tensor[:, :, start:end]
            if seg.shape[2] < segment_frames:
                pad = segment_frames - seg.shape[2]
                seg = torch.nn.functional.pad(seg, (0, pad), value=-80.0)

            # CFT_v9.forward 返回 (onset, frame, offset)，各 (B, T, 48)
            onset_logit, frame_logit, _ = model(seg)
            onset_prob = torch.sigmoid(onset_logit[0]).cpu().numpy()  # (seg, 48)
            frame_prob = torch.sigmoid(frame_logit[0]).cpu().numpy()

            actual = min(segment_frames, T - start)
            onset_map[start:start + actual] += onset_prob[:actual]
            frame_map[start:start + actual] += frame_prob[:actual]
            count_map[start:start + actual] += 1

    count_map = np.maximum(count_map, 1)
    onset_map /= count_map[:, np.newaxis]
    frame_map  /= count_map[:, np.newaxis]
    return frame_map, onset_map


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='CFT v9_manus 推理：输出与原论文 evaluate_github.py 兼容的预测 JSON')
    parser.add_argument('--config',       type=str, default='config.yaml')
    parser.add_argument('--checkpoint',   type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--split',        type=str, default='test',
                        help='评估集名称（test / val / train）')
    parser.add_argument('--onset_thresh', type=float, default=0.15,
                        help='onset 阈值（best_model.pt 记录的最优值 0.15）')
    parser.add_argument('--frame_thresh', type=float, default=0.3,
                        help='frame 阈值（best_model.pt 记录的最优值 0.3）')
    parser.add_argument('--output',       type=str, default='pred_test_v2.json',
                        help='输出 JSON 路径')
    args = parser.parse_args()

    # 读取配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    # 加载模型（使用本目录的 model.py / CFT_v9 / PaperHarmConvBlock）
    model = CFT(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    log.info(f'Checkpoint: epoch={ckpt.get("epoch", "?")}, '
             f'best_val_f1={ckpt.get("best_conp_f1", ckpt.get("best_val_f1", "N/A"))}')

    # 读取 split 文件
    splits_dir = Path(config['data']['splits_dir'])
    with open(splits_dir / f'{args.split}.txt') as f:
        song_ids = [line.strip() for line in f if line.strip()]

    npy_dir     = Path(config['data']['cqt_cache_dir'])
    hop_length  = config['audio']['hop_length']
    sample_rate = config['data']['sample_rate']

    log.info(f'Split={args.split}, 共 {len(song_ids)} 首')
    log.info(f'onset_thresh={args.onset_thresh}, frame_thresh={args.frame_thresh}')
    log.info('=' * 60)

    predictions = {}   # {song_id: [[onset, offset, midi], ...]}
    skipped = 0

    for idx, song_id in enumerate(song_ids):
        npy_path = npy_dir / f'{song_id}.npy'
        if not npy_path.exists():
            log.warning(f'[{idx+1}/{len(song_ids)}] {song_id}: npy 不存在，跳过')
            skipped += 1
            continue

        # 推理
        frame_pred, onset_pred = predict_from_npy(
            model, str(npy_path), config, device)

        # 后处理：帧级 -> 音符列表
        notes = frames_to_notes(
            frame_pred, onset_pred, hop_length, sample_rate,
            onset_thresh=args.onset_thresh,
            frame_thresh=args.frame_thresh)

        predictions[song_id] = notes

        log.info(f'[{idx+1:3d}/{len(song_ids)}] song {song_id:>4s}: '
                 f'{len(notes):4d} notes predicted')

    log.info('=' * 60)
    log.info(f'完成: {len(predictions)} 首成功，{skipped} 首跳过')

    # 保存 JSON
    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    log.info(f'预测结果已保存: {args.output}')

    log.info('')
    log.info('下一步评估命令：')
    log.info(f'  python evaluate_github.py '
             f'{config["data"]["label_path"]} {args.output} 0.05')


if __name__ == '__main__':
    main()
