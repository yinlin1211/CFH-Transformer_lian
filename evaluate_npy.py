"""
CFT 测试集评估脚本（使用预计算 npy 特征）
用法: python evaluate_npy.py --config config.yaml --checkpoint checkpoints/best_model.pt --split test
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from model_v2 import CFT_v2 as CFT

try:
    import mir_eval
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False
    print("ERROR: mir_eval 未安装，请运行: pip install mir_eval")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

MIDI_MIN = 36  # C2


def frames_to_notes(frame_pred, onset_pred, hop_length, sample_rate,
                    onset_thresh=0.5, frame_thresh=0.5, min_note_len=2):
    """帧级预测 -> 音符事件列表"""
    frame_time = hop_length / sample_rate
    T, P = frame_pred.shape
    intervals, pitches = [], []

    for p in range(P):
        midi = p + MIDI_MIN
        onset_frames = np.where(onset_pred[:, p] > onset_thresh)[0]

        if len(onset_frames) == 0:
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
            for i, f_on in enumerate(onset_frames):
                next_onset = onset_frames[i + 1] if i + 1 < len(onset_frames) else T
                f_off = f_on
                for t in range(f_on, min(next_onset, T)):
                    if frame_pred[t, p] > frame_thresh:
                        f_off = t
                    else:
                        if t > f_on + 1:
                            break
                if f_off - f_on + 1 >= min_note_len:
                    intervals.append([f_on * frame_time, (f_off + 1) * frame_time])
                    pitches.append(float(midi))

    if len(intervals) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    return np.array(intervals), np.array(pitches, dtype=float)


def predict_from_npy(model, npy_path, config, device, onset_thresh, frame_thresh):
    """从 npy 文件推理，返回 (frame_pred, onset_pred) numpy arrays"""
    cqt = np.load(npy_path)          # (F, T) float32
    cqt_tensor = torch.from_numpy(cqt).float().unsqueeze(0).to(device)  # (1, F, T)

    segment_frames = config['data']['segment_frames']
    hop_length = config['audio']['hop_length']
    sample_rate = config['data']['sample_rate']
    F, T = cqt.shape

    onset_map = np.zeros((T, 48), dtype=np.float32)
    frame_map = np.zeros((T, 48), dtype=np.float32)
    count_map = np.zeros(T, dtype=np.float32)

    step = segment_frames // 2  # 50% 重叠，与 evaluate.py 一致

    model.eval()
    with torch.no_grad():
        for start in range(0, T, step):
            end = start + segment_frames
            if end > T:
                seg = cqt_tensor[:, :, start:T]
                pad_len = end - T
                seg = torch.nn.functional.pad(seg, (0, pad_len), value=-80.0)
            else:
                seg = cqt_tensor[:, :, start:end]

            onset_pred, frame_pred, offset_pred = model(seg)
            onset_prob = torch.sigmoid(onset_pred[0]).cpu().numpy()  # (seg, 48)
            frame_prob = torch.sigmoid(frame_pred[0]).cpu().numpy()

            actual_len = min(segment_frames, T - start)
            onset_map[start:start + actual_len] += onset_prob[:actual_len]
            frame_map[start:start + actual_len] += frame_prob[:actual_len]
            count_map[start:start + actual_len] += 1

    count_map = np.maximum(count_map, 1)
    onset_map /= count_map[:, np.newaxis]
    frame_map /= count_map[:, np.newaxis]

    return frame_map, onset_map


def evaluate_song(pred_intervals, pred_pitches, ref_intervals, ref_pitches):
    """计算单首歌的 COn / COnP / COnPOff"""
    if len(pred_intervals) == 0 or len(ref_intervals) == 0:
        return 0.0, 0.0, 0.0

    try:
        _, _, con_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches, pred_intervals, pred_pitches,
            onset_tolerance=0.05, pitch_tolerance=0.0,
            offset_ratio=None, offset_min_tolerance=0.05)
    except Exception:
        con_f1 = 0.0

    try:
        _, _, conp_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches, pred_intervals, pred_pitches,
            onset_tolerance=0.05, pitch_tolerance=50.0,
            offset_ratio=None, offset_min_tolerance=0.05)
    except Exception:
        conp_f1 = 0.0

    try:
        _, _, conpoff_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches, pred_intervals, pred_pitches,
            onset_tolerance=0.05, pitch_tolerance=50.0,
            offset_ratio=0.2, offset_min_tolerance=0.05)
    except Exception:
        conpoff_f1 = 0.0

    return con_f1, conp_f1, conpoff_f1


def ref_json_to_intervals(song_notes, hop_length, sample_rate):
    """将 JSON 标注转换为 mir_eval 格式（intervals + pitches）"""
    if len(song_notes) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[n[0], n[1]] for n in song_notes])
    pitches = np.array([float(n[2]) for n in song_notes])
    return intervals, pitches


def main():
    parser = argparse.ArgumentParser(description='CFT 测试集评估（npy 版）')
    parser.add_argument('--config',     type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split',      type=str, default='test')
    parser.add_argument('--onset_thresh', type=float, default=0.10,
                        help='onset 阈值（训练时最优值）')
    parser.add_argument('--frame_thresh', type=float, default=0.35,
                        help='frame 阈值（训练时最优值）')
    parser.add_argument('--output',     type=str, default='eval_results.json')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    # 加载模型
    model = CFT(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    log.info(f'加载 checkpoint: epoch {ckpt["epoch"]}, best COnP_f1={ckpt.get("best_val_f1", "N/A")}')

    # 加载标注
    with open(config['data']['label_path']) as f:
        annotations = json.load(f)

    # 加载 split 文件
    splits_dir = Path(config['data']['splits_dir'])
    with open(splits_dir / f'{args.split}.txt') as f:
        file_list = [line.strip() for line in f if line.strip()]

    npy_dir = Path(config['data']['cqt_cache_dir'])
    hop_length = config['audio']['hop_length']
    sample_rate = config['data']['sample_rate']

    log.info(f'评估 {len(file_list)} 首歌曲（split={args.split}）')
    log.info(f'阈值: onset={args.onset_thresh}, frame={args.frame_thresh}')
    log.info('=' * 60)

    con_list, conp_list, conpoff_list = [], [], []
    skipped = 0

    for idx, song_id in enumerate(file_list):
        npy_path = npy_dir / f'{song_id}.npy'
        if not npy_path.exists():
            log.warning(f'[{idx+1}/{len(file_list)}] {song_id}: npy 文件不存在，跳过')
            skipped += 1
            continue

        if song_id not in annotations:
            log.warning(f'[{idx+1}/{len(file_list)}] {song_id}: 标注不存在，跳过')
            skipped += 1
            continue

        # 推理
        frame_pred, onset_pred = predict_from_npy(
            model, str(npy_path), config, device,
            args.onset_thresh, args.frame_thresh)

        # 预测音符
        pred_intervals, pred_pitches = frames_to_notes(
            frame_pred, onset_pred, hop_length, sample_rate,
            onset_thresh=args.onset_thresh, frame_thresh=args.frame_thresh)

        # 参考音符
        ref_intervals, ref_pitches = ref_json_to_intervals(
            annotations[song_id], hop_length, sample_rate)

        # 计算 F1
        con, conp, conpoff = evaluate_song(
            pred_intervals, pred_pitches, ref_intervals, ref_pitches)

        con_list.append(con)
        conp_list.append(conp)
        conpoff_list.append(conpoff)

        log.info(f'[{idx+1:3d}/{len(file_list)}] {song_id:>6s} | '
                 f'pred={len(pred_intervals):4d} ref={len(ref_intervals):4d} | '
                 f'COn={con:.4f} COnP={conp:.4f} COnPOff={conpoff:.4f}')

    if not con_list:
        log.error('没有成功评估任何歌曲，请检查 npy 路径和标注文件')
        return

    log.info('=' * 60)
    log.info(f'评估完成: {len(con_list)} 首成功，{skipped} 首跳过')
    log.info(f'COn_f1      = {np.mean(con_list):.4f}')
    log.info(f'COnP_f1     = {np.mean(conp_list):.4f}')
    log.info(f'COnPOff_f1  = {np.mean(conpoff_list):.4f}')

    results = {
        'split': args.split,
        'checkpoint': args.checkpoint,
        'onset_thresh': args.onset_thresh,
        'frame_thresh': args.frame_thresh,
        'num_songs': len(con_list),
        'COn_f1':     float(np.mean(con_list)),
        'COnP_f1':    float(np.mean(conp_list)),
        'COnPOff_f1': float(np.mean(conpoff_list)),
        'per_song': {
            file_list[i]: {
                'COn': float(con_list[i]),
                'COnP': float(conp_list[i]),
                'COnPOff': float(conpoff_list[i])
            }
            for i in range(len(con_list))
        }
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f'结果已保存: {args.output}')


if __name__ == '__main__':
    main()
