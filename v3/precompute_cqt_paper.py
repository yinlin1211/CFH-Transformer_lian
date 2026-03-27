"""
CQT 预处理脚本 —— 严格对齐 CFT 论文 + 多进程并行加速
======================================================
注意：CQT 是纯 CPU 计算（librosa），GPU 无法加速。
      改用多进程并行（multiprocessing），充分利用多核 CPU。

论文参数：
  - 采样率：16kHz
  - CQT bins：288（6个八度 × 48 bins/octave）
  - fmin：G1 ≈ 49.0 Hz（人声音域起点，覆盖 G1~F7）
  - hop_length：320 样本 = 20ms/帧

输出：
  cqt_cache/npy/1.npy ... 500.npy   ← 模型训练用，float32 dB幅度
  cqt_cache/png/1.png ... 500.png   ← 可视化，叠加标注音符（青色）

用法：
  python precompute_cqt_paper.py            # 默认用全部CPU核心
  python precompute_cqt_paper.py --workers 8  # 指定8个进程
"""

import os
import json
import argparse
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ── 严格对齐论文的参数 ──────────────────────────────────────
AUDIO_DIR   = "/mnt/ssd/lian/st500假的旧的/vocal_mp3/"
LABEL_PATH  = "/mnt/ssd/lian/st500假的旧的/MIR-ST500_corrected.json"
OUTPUT_DIR  = "/mnt/ssd/lian/CFHTransformer/cqt_cache/"

SAMPLE_RATE     = 16000
HOP_LENGTH      = 320
CQT_BINS        = 288
BINS_PER_OCTAVE = 48
FMIN            = librosa.note_to_hz('G1')   # ≈ 49.0 Hz


def save_png(cqt_db, notes, song_id, out_path):
    F, T = cqt_db.shape
    frame_sec = HOP_LENGTH / SAMPLE_RATE
    duration  = T * frame_sec

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(cqt_db, aspect='auto', origin='lower', cmap='magma',
              extent=[0, duration, 0, F], interpolation='nearest')

    for onset, offset, pitch in notes:
        freq    = librosa.midi_to_hz(pitch)
        bin_idx = int(round(np.log2(freq / FMIN) * BINS_PER_OCTAVE))
        if bin_idx < 0 or bin_idx >= F:
            continue
        t0 = onset / frame_sec / T * duration
        t1 = offset / frame_sec / T * duration
        ax.plot([t0, t1], [bin_idx, bin_idx],
                color='cyan', linewidth=1.2, alpha=0.85)

    octave_ticks  = [i * BINS_PER_OCTAVE for i in range(7)]
    octave_labels = ['G1','G2','G3','G4','G5','G6','G7']
    ax.set_yticks(octave_ticks)
    ax.set_yticklabels(octave_labels, fontsize=8)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Pitch', fontsize=9)
    ax.set_title(f'Song {song_id}  {T}frames  {duration:.1f}s  fmin=G1  cyan=notes', fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=80, bbox_inches='tight')
    plt.close(fig)


def process_one(args):
    """单进程处理一首歌，供 Pool.map 调用。"""
    song_id, annotations = args

    npy_path = Path(OUTPUT_DIR) / "npy" / f"{song_id}.npy"
    png_path = Path(OUTPUT_DIR) / "png" / f"{song_id}.png"

    if npy_path.exists() and png_path.exists():
        return (song_id, 'skip', None)

    audio_path = Path(AUDIO_DIR) / f"{song_id}_vocals.mp3"
    if not audio_path.exists():
        return (song_id, 'error', f"音频不存在: {audio_path}")

    try:
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

        cqt_complex = librosa.cqt(
            audio,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            fmin=FMIN,
            n_bins=CQT_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
        )
        cqt_db = librosa.amplitude_to_db(
            np.abs(cqt_complex), ref=np.max
        ).astype(np.float32)   # (288, T)

        if not npy_path.exists():
            np.save(str(npy_path), cqt_db)

        if not png_path.exists():
            notes = annotations[song_id]
            save_png(cqt_db, notes, song_id, png_path)

        return (song_id, 'ok', None)

    except Exception as e:
        return (song_id, 'error', str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int,
                        default=min(cpu_count(), 16),
                        help='并行进程数，默认=min(CPU核心数, 16)')
    args = parser.parse_args()

    print(f"fmin = {FMIN:.4f} Hz  (G1，覆盖 G1~F7，严格对齐论文)")
    print(f"并行进程数: {args.workers}  (服务器 CPU 核心数: {cpu_count()})")

    # 创建输出目录
    (Path(OUTPUT_DIR) / "npy").mkdir(parents=True, exist_ok=True)
    (Path(OUTPUT_DIR) / "png").mkdir(parents=True, exist_ok=True)

    # 加载标注
    with open(LABEL_PATH) as f:
        annotations = json.load(f)
    song_ids = sorted(annotations.keys(), key=lambda x: int(x))
    print(f"共 {len(song_ids)} 首歌\n")

    # 构造任务列表
    tasks = [(sid, annotations) for sid in song_ids]

    # 多进程并行处理
    ok_count = skip_count = err_count = 0
    errors = []

    with Pool(processes=args.workers) as pool:
        for song_id, status, msg in tqdm(
            pool.imap_unordered(process_one, tasks),
            total=len(tasks), desc="CQT precompute"
        ):
            if status == 'ok':
                ok_count += 1
            elif status == 'skip':
                skip_count += 1
            else:
                err_count += 1
                errors.append(f"Song {song_id}: {msg}")
                tqdm.write(f"[ERROR] {song_id}: {msg}")

    print(f"\n完成！处理: {ok_count}  跳过: {skip_count}  错误: {err_count}")
    if errors:
        print("错误列表：")
        for e in errors:
            print(f"  {e}")

    # 验证
    sample = Path(OUTPUT_DIR) / "npy" / "1.npy"
    if sample.exists():
        arr = np.load(str(sample))
        print(f"\n验证 1.npy: shape={arr.shape}  dtype={arr.dtype}  range=[{arr.min():.1f}, {arr.max():.1f}] dB")


if __name__ == '__main__':
    main()
