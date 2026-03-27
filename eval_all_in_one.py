"""
一体化批量评估脚本
- 对 checkpoints_v2 下所有 checkpoint 依次推理 + 评估
- 推理使用大 batch（充分利用 V100 32GB 显存）
- 结果保存到「今天的推理情况.txt」

用法（在 CFH-Transformer_v2 目录下运行）：
    CUDA_VISIBLE_DEVICES=0 python eval_all_in_one.py
"""

import sys
import glob
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# 优先加载本目录的 model_v2.py（PaperHarmConvBlock）
sys.path.insert(0, str(Path(__file__).parent))
from model_v2 import CFT_v2 as CFT

# ── mir_eval ────────────────────────────────────────────────────────────────
try:
    import mir_eval
    from mir_eval import transcription, util
except ImportError:
    print('ERROR: pip install mir_eval')
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# ── 常量 ────────────────────────────────────────────────────────────────────
MIDI_MIN       = 36       # C2
ONSET_THRESH   = 0.15     # 固定阈值
FRAME_THRESH   = 0.35
INFER_BATCH    = 64       # V100 32GB，一次推理 64 个 segment
ONSET_TOL      = 0.05     # 50ms，与原论文一致


# ── 后处理：帧级预测 → 音符列表 ─────────────────────────────────────────────

def frames_to_notes(frame_pred, onset_pred, hop_length, sample_rate,
                    onset_thresh=ONSET_THRESH, frame_thresh=FRAME_THRESH,
                    min_note_len=2):
    """返回 [[onset_sec, offset_sec, midi_pitch], ...]"""
    frame_time = hop_length / sample_rate
    T, P = frame_pred.shape
    notes = []
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
                        notes.append([note_start * frame_time, t * frame_time, float(midi)])
            if in_note and T - note_start >= min_note_len:
                notes.append([note_start * frame_time, T * frame_time, float(midi)])
        else:
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
                    notes.append([f_on * frame_time, (f_off + 1) * frame_time, float(midi)])
    return notes


# ── 推理：npy → 概率图（大 batch） ──────────────────────────────────────────

def predict_from_npy(model, npy_path, config, device, infer_batch=INFER_BATCH):
    """从 npy CQT 推理，返回 (frame_pred, onset_pred)，shape=(T, 48)"""
    cqt = np.load(npy_path)                          # (F, T)
    segment_frames = config['data']['segment_frames']
    T = cqt.shape[1]
    step = segment_frames // 2                       # 50% 重叠

    # 预切所有 segment
    segments = []
    starts = list(range(0, T, step))
    for start in starts:
        seg = cqt[:, start:start + segment_frames]
        if seg.shape[1] < segment_frames:
            seg = np.pad(seg, ((0, 0), (0, segment_frames - seg.shape[1])),
                         constant_values=-80.0)
        segments.append(seg)

    # 分批推理
    onset_map = np.zeros((T, 48), dtype=np.float32)
    frame_map  = np.zeros((T, 48), dtype=np.float32)
    count_map  = np.zeros(T,       dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(segments), infer_batch):
            batch_segs = segments[batch_start:batch_start + infer_batch]
            batch_tensor = torch.from_numpy(
                np.stack(batch_segs, axis=0)).float().to(device)  # (B, F, seg)
            onset_logit, frame_logit, _ = model(batch_tensor)
            onset_prob = torch.sigmoid(onset_logit).cpu().numpy()  # (B, seg, 48)
            frame_prob = torch.sigmoid(frame_logit).cpu().numpy()

            for j, start in enumerate(starts[batch_start:batch_start + infer_batch]):
                actual = min(segment_frames, T - start)
                onset_map[start:start + actual] += onset_prob[j, :actual]
                frame_map[start:start + actual]  += frame_prob[j, :actual]
                count_map[start:start + actual]  += 1

    count_map = np.maximum(count_map, 1)
    onset_map /= count_map[:, np.newaxis]
    frame_map  /= count_map[:, np.newaxis]
    return frame_map, onset_map


# ── 评估：对齐原论文 evaluate_github.py ─────────────────────────────────────

def eval_one_song(est_notes, ref_notes, onset_tolerance=ONSET_TOL):
    """
    est_notes / ref_notes: [[onset_sec, offset_sec, midi_pitch], ...]
    返回 {'COn': f1, 'COnP': f1, 'COnPOff': f1}
    """
    # 过滤 duration <= 0
    est_notes = [n for n in est_notes if n[1] - n[0] > 0]
    ref_notes = [n for n in ref_notes if n[1] - n[0] > 0]

    if len(est_notes) == 0 or len(ref_notes) == 0:
        return {'COn': 0.0, 'COnP': 0.0, 'COnPOff': 0.0}

    est_intervals = np.array([[n[0], n[1]] for n in est_notes])
    est_pitches   = util.midi_to_hz(np.array([n[2] for n in est_notes]))
    ref_intervals = np.array([[n[0], n[1]] for n in ref_notes])
    ref_pitches   = util.midi_to_hz(np.array([n[2] for n in ref_notes]))

    raw = transcription.evaluate(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50,
    )
    return {
        'COnPOff': raw['F-measure'],
        'COnP':    raw['F-measure_no_offset'],
        'COn':     raw['Onset_F-measure'],
    }


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    project_dir = Path(__file__).parent
    config_path = project_dir / 'config.yaml'
    ckpt_dir    = project_dir / 'checkpoints_v2'
    gt_json_path = Path('/mnt/ssd/lian/论文复现/CFH-Transformer/MIR-ST500_corrected.json')
    output_txt  = project_dir / '今天的推理情况.txt'

    # 读取配置
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')
    log.info(f'INFER_BATCH={INFER_BATCH}, ONSET_THRESH={ONSET_THRESH}, FRAME_THRESH={FRAME_THRESH}')

    # 读取标注
    with open(gt_json_path) as f:
        gt_all = json.load(f)

    # 读取测试集 song_id 列表
    splits_dir  = Path(config['data']['splits_dir'])
    with open(splits_dir / 'test.txt') as f:
        test_ids = [l.strip() for l in f if l.strip()]
    npy_dir     = Path(config['data']['cqt_cache_dir'])
    hop_length  = config['audio']['hop_length']
    sample_rate = config['data']['sample_rate']

    log.info(f'测试集: {len(test_ids)} 首')

    # 收集所有 checkpoint，按 epoch 排序，best_model 放最前
    ckpt_files = sorted(
        glob.glob(str(ckpt_dir / 'checkpoint_epoch*.pt')),
        key=lambda x: int(Path(x).stem.replace('checkpoint_epoch', ''))
    )
    best_model = ckpt_dir / 'best_model.pt'
    if best_model.exists():
        ckpt_files = [str(best_model)] + ckpt_files

    log.info(f'共 {len(ckpt_files)} 个 checkpoint')
    log.info('=' * 70)

    all_results = []

    for ckpt_idx, ckpt_path in enumerate(ckpt_files):
        ckpt_name = Path(ckpt_path).stem
        log.info(f'[{ckpt_idx+1}/{len(ckpt_files)}] 加载 {ckpt_name}')

        # 读取元数据
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        epoch   = ckpt.get('epoch', -1)
        val_f1  = ckpt.get('best_conp_f1', ckpt.get('best_val_f1', float('nan')))

        # 加载模型
        model = CFT(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        del ckpt
        torch.cuda.empty_cache()

        # 逐首推理 + 评估
        song_metrics = []
        for song_id in test_ids:
            npy_path = npy_dir / f'{song_id}.npy'
            if not npy_path.exists():
                continue

            frame_pred, onset_pred = predict_from_npy(
                model, str(npy_path), config, device)
            est_notes = frames_to_notes(frame_pred, onset_pred, hop_length, sample_rate)

            ref_raw = gt_all.get(song_id, gt_all.get(str(song_id), []))
            ref_notes = [[float(n[0]), float(n[1]), float(n[2])] for n in ref_raw]

            m = eval_one_song(est_notes, ref_notes)
            song_metrics.append(m)

        # 宏平均（每首等权，与原论文一致）
        n = len(song_metrics)
        if n == 0:
            log.warning(f'  {ckpt_name}: 无有效歌曲，跳过')
            continue

        con_f1     = sum(m['COn']     for m in song_metrics) / n
        conp_f1    = sum(m['COnP']    for m in song_metrics) / n
        conpoff_f1 = sum(m['COnPOff'] for m in song_metrics) / n

        log.info(f'  epoch={epoch:>5d}  val_COnP(旧)={val_f1:.4f}  '
                 f'COn={con_f1:.4f}  COnP={conp_f1:.4f}  COnPOff={conpoff_f1:.4f}')
        log.info('-' * 70)

        all_results.append({
            'checkpoint':   ckpt_name,
            'epoch':        epoch,
            'val_COnP_old': round(float(val_f1), 6),
            'COn':          round(con_f1, 6),
            'COnP':         round(conp_f1, 6),
            'COnPOff':      round(conpoff_f1, 6),
        })

        # 每跑完一个立即写入，防止中断丢失
        _write_txt(all_results, output_txt)

        del model
        torch.cuda.empty_cache()

    # 最终写入
    _write_txt(all_results, output_txt)
    log.info(f'\n结果已保存到: {output_txt}')


def _write_txt(results, path):
    """写入今天的推理情况.txt"""
    if not results:
        return
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    best = max(results, key=lambda x: x['COnP'])

    lines = []
    lines.append(f'CFH-Transformer_v2 批量评估结果')
    lines.append(f'生成时间: {now}')
    lines.append(f'阈值: onset_thresh={ONSET_THRESH}, frame_thresh={FRAME_THRESH}')
    lines.append(f'评估方式: 对齐原论文 evaluate_github.py（pitch 转 Hz，mir_eval transcription.evaluate）')
    lines.append('')
    lines.append(f'{"Checkpoint":<35} {"Epoch":>6} {"val_COnP(旧)":>12} {"COn":>8} {"COnP":>8} {"COnPOff":>10}')
    lines.append('-' * 85)

    # 按 COnP 降序排列
    for r in sorted(results, key=lambda x: x['COnP'], reverse=True):
        marker = ' ← 最佳' if r['checkpoint'] == best['checkpoint'] else ''
        lines.append(
            f'{r["checkpoint"]:<35} {r["epoch"]:>6} '
            f'{r["val_COnP_old"]:>12.4f} '
            f'{r["COn"]:>8.4f} {r["COnP"]:>8.4f} {r["COnPOff"]:>10.4f}'
            f'{marker}'
        )

    lines.append('')
    lines.append(f'最佳 checkpoint: {best["checkpoint"]}  (epoch {best["epoch"]})')
    lines.append(f'  COn     = {best["COn"]:.4f}')
    lines.append(f'  COnP    = {best["COnP"]:.4f}')
    lines.append(f'  COnPOff = {best["COnPOff"]:.4f}')
    lines.append('')
    lines.append(f'注：val_COnP(旧) 是训练时用旧评估代码算的验证集 F1，仅供参考，不可信。')
    lines.append(f'    COn / COnP / COnPOff 是本次用正确评估代码在测试集上算的结果。')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
