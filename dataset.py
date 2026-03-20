"""
MIR-ST500 数据集加载器（v2 — 对齐论文）

变更：
  - 输出音高 N=48（C2~B5，MIDI 36~83），不再是 88 键钢琴
  - 去掉 onset/offset 的高斯平滑（论文未提及）
  - cqt_tensor shape: (F=288, T)，DataLoader 自动加 batch 维度

训练模式：每个 __getitem__ 随机截取一段 segment_frames 帧的片段
验证/测试模式：__getitem__ 返回整首歌的完整 CQT 和标签
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
import random


# 论文音高范围：C2~B5，MIDI 36~83，共 48 个音
MIDI_MIN = 36   # C2
MIDI_MAX = 83   # B5
NUM_PITCHES = MIDI_MAX - MIDI_MIN + 1  # 48


class MIR_ST500_Dataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self.cqt_cache_dir = Path(config["data"]["cqt_cache_dir"])
        self.label_path = Path(config["data"]["label_path"])
        self.splits_dir = Path(config["data"]["splits_dir"])
        self.segment_frames = config["data"]["segment_frames"]
        self.hop_length = config["audio"]["hop_length"]
        self.sample_rate = config["data"]["sample_rate"]

        with open(self.label_path, "r") as f:
            self.annotations = json.load(f)

        split_file = self.splits_dir / f"{split}.txt"
        with open(split_file, "r") as f:
            self.file_list = [line.strip() for line in f if line.strip()]

        # 过滤掉没有缓存文件的歌曲
        valid = []
        for sid in self.file_list:
            if (self.cqt_cache_dir / f"{sid}.npy").exists():
                valid.append(sid)
        if len(valid) < len(self.file_list):
            print(f"Warning: {len(self.file_list) - len(valid)} songs missing CQT cache in {split} split")
        self.file_list = valid

        if split == "train":
            self._build_train_index()

    def _build_train_index(self):
        """
        为训练集构建 (song_id, start_frame) 索引。
        每首歌按 segment_frames//2 的步长切片（50%重叠），
        确保每个片段至少包含1个音符。
        """
        self._train_index = []
        stride = self.segment_frames // 2

        for sid in self.file_list:
            cqt_path = self.cqt_cache_dir / f"{sid}.npy"
            cqt = np.load(str(cqt_path))
            num_frames = cqt.shape[1]

            # 预计算该歌曲的音符帧位置（只统计 C2~B5 范围内的音符）
            notes = self.annotations.get(sid, [])
            frame_time = self.hop_length / self.sample_rate
            note_frames = set()
            for note in notes:
                midi = int(note[2])
                if not (MIDI_MIN <= midi <= MIDI_MAX):
                    continue
                f_on = int(round(float(note[0]) / frame_time))
                f_off = int(round(float(note[1]) / frame_time))
                for f in range(f_on, min(f_off + 1, num_frames)):
                    note_frames.add(f)

            if num_frames < self.segment_frames:
                self._train_index.append((sid, 0))
                continue

            for start in range(0, num_frames - self.segment_frames + 1, stride):
                end = start + self.segment_frames
                segment_has_note = any(start <= f < end for f in note_frames)
                if segment_has_note:
                    self._train_index.append((sid, start))

            # 确保每首歌至少有一个样本
            if not any(s == sid for s, _ in self._train_index[-20:]):
                self._train_index.append((sid, 0))

    def __len__(self):
        if self.split == "train":
            return len(self._train_index)
        else:
            return len(self.file_list)

    def __getitem__(self, idx):
        if self.split == "train":
            return self._get_train_item(idx)
        else:
            return self._get_full_song(idx)

    def _get_train_item(self, idx):
        """训练：返回固定长度片段"""
        song_id, start = self._train_index[idx]

        # 随机抖动（±stride/4）
        jitter = random.randint(-self.segment_frames // 8, self.segment_frames // 8)
        cqt = np.load(str(self.cqt_cache_dir / f"{song_id}.npy"))
        num_frames = cqt.shape[1]

        start = max(0, min(start + jitter, num_frames - self.segment_frames))
        end = start + self.segment_frames

        cqt_seg = cqt[:, start:end]                   # (288, segment_frames)
        labels = self._create_labels(song_id, num_frames)
        labels_seg = {k: v[start:end] for k, v in labels.items()}

        cqt_tensor = torch.from_numpy(cqt_seg).float()  # (288, T)
        label_tensors = {k: torch.from_numpy(v).float() for k, v in labels_seg.items()}
        return cqt_tensor, label_tensors

    def _get_full_song(self, idx):
        """验证/测试：返回整首歌"""
        song_id = self.file_list[idx]
        cqt = np.load(str(self.cqt_cache_dir / f"{song_id}.npy"))
        num_frames = cqt.shape[1]

        MAX_FRAMES = 4096
        if num_frames > MAX_FRAMES:
            start = (num_frames - MAX_FRAMES) // 4
            num_frames = MAX_FRAMES
            cqt = cqt[:, start:start + MAX_FRAMES]
        else:
            start = 0

        labels = self._create_labels(song_id, cqt.shape[1] + start)
        labels_seg = {k: v[start:start + cqt.shape[1]] for k, v in labels.items()}

        cqt_tensor = torch.from_numpy(cqt).float()
        label_tensors = {k: torch.from_numpy(v).float() for k, v in labels_seg.items()}
        return cqt_tensor, label_tensors, song_id

    def _create_labels(self, song_id, num_frames):
        """
        创建标签矩阵。

        音高范围：C2~B5（MIDI 36~83），共 48 个音。
        不做高斯平滑（论文未提及）。
        """
        notes = self.annotations.get(song_id, [])
        frame_time = self.hop_length / self.sample_rate

        onset  = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)
        offset = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)
        frame  = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)

        for note in notes:
            t_on, t_off, midi = float(note[0]), float(note[1]), int(note[2])
            pitch_idx = midi - MIDI_MIN
            if not (0 <= pitch_idx < NUM_PITCHES):
                continue
            f_on = int(round(t_on / frame_time))
            f_off = int(round(t_off / frame_time))

            if f_on < num_frames:
                onset[f_on, pitch_idx] = 1.0
            if f_off < num_frames:
                offset[f_off, pitch_idx] = 1.0
            for f in range(f_on, min(f_off + 1, num_frames)):
                frame[f, pitch_idx] = 1.0

        return {"onset": onset, "offset": offset, "frame": frame}
