"""
生成 MIR-ST500 的 train/val/test 划分文件。
论文设定：400首训练，100首测试。
我们从400首中再取40首作为验证集。
"""

import json
import os
from pathlib import Path

label_path = "/mnt/ssd/lian/st500假的旧的/MIR-ST500_corrected.json"
audio_dir = "/mnt/ssd/lian/st500假的旧的/vocal_mp3"
output_dir = "/mnt/ssd/lian/CFHTransformer/splits"

with open(label_path) as f:
    annotations = json.load(f)

# 获取所有有音频文件的 song_id
all_ids = []
for song_id in annotations.keys():
    audio_path = Path(audio_dir) / f"{song_id}_vocals.mp3"
    if audio_path.exists():
        all_ids.append(song_id)

# 按数字排序
all_ids_sorted = sorted(all_ids, key=lambda x: int(x))
print(f"Total songs with audio: {len(all_ids_sorted)}")

# 论文划分：前400训练，后100测试
# 我们从前400中取最后40作为验证集
train_ids = all_ids_sorted[:360]
val_ids = all_ids_sorted[360:400]
test_ids = all_ids_sorted[400:]

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

# 写入文件
os.makedirs(output_dir, exist_ok=True)
for split_name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
    out_file = Path(output_dir) / f"{split_name}.txt"
    with open(out_file, 'w') as f:
        for sid in ids:
            f.write(sid + '\n')
    print(f"Wrote {out_file}")
