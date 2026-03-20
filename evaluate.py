"""
CFT 评估脚本 - 使用 mir_eval 计算 COn, COnP, COnPOff 指标
用法: CUDA_VISIBLE_DEVICES=2 python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pt
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
import librosa
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
from model import CFT

try:
    import mir_eval
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False
    print("Warning: mir_eval not installed. Install with: pip install mir_eval")


def load_audio_cqt(audio_path: str, config: dict) -> torch.Tensor:
    """加载音频并计算 CQT，返回 (F, T) tensor"""
    waveform, sr = torchaudio.load(audio_path)
    if sr != config['data']['sample_rate']:
        waveform = torchaudio.transforms.Resample(sr, config['data']['sample_rate'])(waveform)
    waveform = torch.mean(waveform, dim=0).numpy()

    cqt = librosa.cqt(
        waveform,
        sr=config['data']['sample_rate'],
        hop_length=config['audio']['hop_length'],
        fmin=config['audio']['fmin'],
        n_bins=config['audio']['cqt_bins']
    )
    cqt = np.abs(cqt)
    cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    return torch.from_numpy(cqt).float()  # (F, T)


def predict_notes(model, cqt: torch.Tensor, config: dict, device: torch.device,
                  onset_threshold: float = 0.5, frame_threshold: float = 0.5) -> list:
    """
    对完整音频进行推理，返回预测的音符列表 [(onset, offset, pitch_midi), ...]
    使用滑动窗口处理长音频
    """
    model.eval()
    hop_length = config['audio']['hop_length']
    sample_rate = config['data']['sample_rate']
    frame_time = hop_length / sample_rate
    segment_frames = config['data']['segment_frames']

    F, T = cqt.shape
    # 滑动窗口推理，步长 = segment_frames // 2（50% 重叠）
    step = segment_frames // 2
    onset_map = np.zeros((T, 88), dtype=np.float32)
    frame_map = np.zeros((T, 88), dtype=np.float32)
    count_map = np.zeros(T, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, T, step):
            end = start + segment_frames
            if end > T:
                # 对最后一段进行 padding
                seg = cqt[:, start:T]
                pad_len = end - T
                seg = torch.nn.functional.pad(seg, (0, pad_len), value=-80.0)
            else:
                seg = cqt[:, start:end]

            seg = seg.unsqueeze(0).to(device)  # (1, F, segment_frames)
            onset_pred, frame_pred, offset_pred = model(seg)

            onset_prob = torch.sigmoid(onset_pred[0]).cpu().numpy()  # (segment_frames, 88)
            frame_prob = torch.sigmoid(frame_pred[0]).cpu().numpy()

            actual_len = min(segment_frames, T - start)
            onset_map[start:start + actual_len] += onset_prob[:actual_len]
            frame_map[start:start + actual_len] += frame_prob[:actual_len]
            count_map[start:start + actual_len] += 1

    # 平均重叠区域
    count_map = np.maximum(count_map, 1)
    onset_map /= count_map[:, np.newaxis]
    frame_map /= count_map[:, np.newaxis]

    # 后处理：从 onset/frame map 提取音符
    notes = []
    for pitch_idx in range(88):
        midi_pitch = pitch_idx + 21
        onset_frames = np.where(onset_map[:, pitch_idx] > onset_threshold)[0]
        frame_active = frame_map[:, pitch_idx] > frame_threshold

        if len(onset_frames) == 0:
            continue

        for onset_f in onset_frames:
            # 找到 onset 之后 frame 结束的位置
            end_f = onset_f
            while end_f < T and frame_active[end_f]:
                end_f += 1

            if end_f > onset_f:
                onset_time = onset_f * frame_time
                offset_time = end_f * frame_time
                notes.append((onset_time, offset_time, midi_pitch))

    return sorted(notes, key=lambda x: x[0])


def evaluate_song(pred_notes: list, ref_notes: list) -> dict:
    """使用 mir_eval 计算 COn, COnP, COnPOff"""
    if not HAS_MIR_EVAL:
        return {}

    if len(pred_notes) == 0 or len(ref_notes) == 0:
        return {'COn': 0.0, 'COnP': 0.0, 'COnPOff': 0.0}

    # 转换为 mir_eval 格式
    pred_intervals = np.array([[n[0], n[1]] for n in pred_notes])
    pred_pitches = np.array([librosa.midi_to_hz(n[2]) for n in pred_notes])

    ref_intervals = np.array([[n[0], n[1]] for n in ref_notes])
    ref_pitches = np.array([librosa.midi_to_hz(n[2]) for n in ref_notes])

    # COn: onset only (50ms tolerance)
    prec, rec, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, pred_intervals, pred_pitches,
        onset_tolerance=0.05, pitch_tolerance=0.0, offset_ratio=None
    )
    con = f1

    # COnP: onset + pitch
    prec, rec, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, pred_intervals, pred_pitches,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
    )
    conp = f1

    # COnPOff: onset + pitch + offset (0.2x note duration)
    prec, rec, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, pred_intervals, pred_pitches,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=0.2
    )
    conpoff = f1

    return {'COn': con * 100, 'COnP': conp * 100, 'COnPOff': conpoff * 100}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--onset_threshold', type=float, default=0.5)
    parser.add_argument('--frame_threshold', type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = CFT(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Load annotations
    with open(config['data']['label_path']) as f:
        annotations = json.load(f)

    # Load split file list
    splits_dir = Path(config['data']['splits_dir'])
    split_file = splits_dir / f'{args.split}.txt'
    with open(split_file) as f:
        file_list = [line.strip() for line in f]

    audio_dir = Path(config['data']['audio_dir'])

    all_metrics = {'COn': [], 'COnP': [], 'COnPOff': []}

    for song_id in file_list:
        audio_path = audio_dir / f'{song_id}_vocals.mp3'
        if not audio_path.exists():
            print(f"Warning: {audio_path} not found, skipping.")
            continue

        cqt = load_audio_cqt(str(audio_path), config)
        pred_notes = predict_notes(model, cqt, config, device,
                                   args.onset_threshold, args.frame_threshold)

        ref_notes = [(n[0], n[1], int(n[2])) for n in annotations[song_id]]
        metrics = evaluate_song(pred_notes, ref_notes)

        if metrics:
            for k in all_metrics:
                all_metrics[k].append(metrics[k])
            print(f"Song {song_id}: COn={metrics['COn']:.2f} COnP={metrics['COnP']:.2f} COnPOff={metrics['COnPOff']:.2f}")

    if all_metrics['COnP']:
        print("\n=== Final Results ===")
        for k in ['COn', 'COnP', 'COnPOff']:
            avg = np.mean(all_metrics[k])
            print(f"{k}: {avg:.2f}")

        results = {k: float(np.mean(v)) for k, v in all_metrics.items()}
        with open('eval_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved to eval_results.json")


if __name__ == '__main__':
    main()
