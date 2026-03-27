"""
批量测试脚本：对 checkpoints_v2 目录下所有 checkpoint 依次推理 + 评估
结果保存到 eval_all_results.json 和 eval_all_results.csv

用法（在 CFH-Transformer_v2 目录下运行）：
    CUDA_VISIBLE_DEVICES=1 python eval_all_checkpoints.py

所有 checkpoint 统一使用固定阈值：onset_thresh=0.15, frame_thresh=0.35

可选参数：
    --config        config.yaml 路径（默认 config.yaml）
    --ckpt_dir      checkpoint 目录（默认 checkpoints_v2）
    --gt_json       标注 JSON 路径
    --split         评估集（默认 test）
    --onset_thresh  固定 onset 阈值（默认 0.15）
    --frame_thresh  固定 frame 阈值（默认 0.35）
    --output_dir    临时预测 JSON 存放目录（默认 eval_tmp）
    --results_file  汇总结果文件前缀（默认 eval_all_results）
    --gpu           CUDA 设备编号（默认 0）
    --skip_existing 跳过已有预测 JSON 的 checkpoint（断点续跑）
"""

import argparse
import csv
import glob
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)


def get_ckpt_meta(ckpt_path):
    """从 checkpoint 文件读取 epoch 和最优阈值"""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', -1)
        onset_thresh = ckpt.get('best_onset_thresh', None)
        frame_thresh = ckpt.get('best_frame_thresh', None)
        val_f1 = ckpt.get('best_conp_f1', ckpt.get('best_val_f1', None))
        return epoch, onset_thresh, frame_thresh, val_f1
    except Exception as e:
        log.warning(f'读取 checkpoint 元数据失败: {e}')
        return -1, None, None, None


def run_predict(predict_script, config, ckpt_path, split,
                onset_thresh, frame_thresh, output_json, gpu):
    """调用 predict_to_json.py 推理"""
    cmd = [
        sys.executable, str(predict_script),
        '--config', str(config),
        '--checkpoint', str(ckpt_path),
        '--split', split,
        '--onset_thresh', str(onset_thresh),
        '--frame_thresh', str(frame_thresh),
        '--output', str(output_json),
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    log.info(f'  推理命令: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        log.error(f'  推理失败:\n{result.stderr[-2000:]}')
        return False
    return True


def run_evaluate(evaluate_script, gt_json, pred_json, onset_tolerance=0.05):
    """调用 evaluate_github.py 评估，解析输出"""
    cmd = [
        sys.executable, str(evaluate_script),
        str(gt_json),
        str(pred_json),
        str(onset_tolerance),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f'  评估失败:\n{result.stderr[-2000:]}')
        return None

    # 解析输出，格式：
    #          Precision Recall F1-score
    # COnPOff  0.469155 0.456367 0.461692
    # COnP     0.719532 0.700462 0.708356
    # COn      0.757586 0.737199 0.745656
    metrics = {}
    for line in result.stdout.splitlines():
        for key in ['COnPOff', 'COnP', 'COn']:
            if line.strip().startswith(key):
                parts = line.split()
                if len(parts) >= 4:
                    metrics[key] = {
                        'precision': float(parts[1]),
                        'recall':    float(parts[2]),
                        'f1':        float(parts[3]),
                    }
    return metrics if metrics else None


def main():
    parser = argparse.ArgumentParser(description='批量测试所有 checkpoint')
    parser.add_argument('--config',        type=str, default='config.yaml')
    parser.add_argument('--ckpt_dir',      type=str, default='checkpoints_v2')
    parser.add_argument('--gt_json',       type=str,
                        default='/mnt/ssd/lian/论文复现/CFH-Transformer/MIR-ST500_corrected.json')
    parser.add_argument('--split',         type=str, default='test')
    parser.add_argument('--onset_thresh',  type=float, default=0.15,
                        help='固定 onset 阈值（默认 0.15）')
    parser.add_argument('--frame_thresh',  type=float, default=0.35,
                        help='固定 frame 阈值（默认 0.35）')
    parser.add_argument('--output_dir',    type=str, default='eval_tmp')
    parser.add_argument('--results_file',  type=str, default='eval_all_results')
    parser.add_argument('--gpu',           type=int, default=0)
    parser.add_argument('--skip_existing', action='store_true',
                        help='跳过已有预测 JSON 的 checkpoint（断点续跑）')
    args = parser.parse_args()

    # 路径解析（相对于当前工作目录）
    project_dir   = Path.cwd()
    predict_script  = project_dir / 'predict_to_json.py'
    evaluate_script = project_dir / 'evaluate_github.py'
    config          = project_dir / args.config
    ckpt_dir        = project_dir / args.ckpt_dir
    output_dir      = project_dir / args.output_dir
    gt_json         = Path(args.gt_json)

    # 检查必要文件
    for p, name in [(predict_script, 'predict_to_json.py'),
                    (evaluate_script, 'evaluate_github.py'),
                    (config, 'config.yaml'),
                    (ckpt_dir, 'checkpoints_v2'),
                    (gt_json, 'MIR-ST500_corrected.json')]:
        if not p.exists():
            log.error(f'找不到 {name}: {p}')
            sys.exit(1)

    output_dir.mkdir(exist_ok=True)

    # 收集所有 checkpoint，按 epoch 排序
    # 排除 latest.pt（与最后一个 epoch 重复）
    ckpt_files = sorted(
        [f for f in glob.glob(str(ckpt_dir / 'checkpoint_epoch*.pt'))],
        key=lambda x: int(Path(x).stem.replace('checkpoint_epoch', ''))
    )
    # 加入 best_model.pt
    best_model = ckpt_dir / 'best_model.pt'
    if best_model.exists():
        ckpt_files = [str(best_model)] + ckpt_files

    log.info(f'共找到 {len(ckpt_files)} 个 checkpoint')
    log.info('=' * 70)

    all_results = []

    for idx, ckpt_path in enumerate(ckpt_files):
        ckpt_name = Path(ckpt_path).stem
        log.info(f'[{idx+1}/{len(ckpt_files)}] {ckpt_name}')

        # 读取元数据
        epoch, ot_meta, ft_meta, val_f1 = get_ckpt_meta(ckpt_path)

        # 使用固定阈值
        onset_thresh = args.onset_thresh
        frame_thresh = args.frame_thresh

        log.info(f'  epoch={epoch}  val_COnP_f1={val_f1:.4f}  '
                 f'onset_thresh={onset_thresh}  frame_thresh={frame_thresh}')

        # 预测 JSON 路径
        pred_json = output_dir / f'{ckpt_name}_pred.json'

        # 断点续跑：已有预测则跳过推理
        if args.skip_existing and pred_json.exists():
            log.info(f'  跳过推理（已有 {pred_json.name}）')
        else:
            ok = run_predict(
                predict_script, config, ckpt_path, args.split,
                onset_thresh, frame_thresh, pred_json, args.gpu)
            if not ok:
                log.warning(f'  {ckpt_name} 推理失败，跳过')
                continue

        # 评估
        metrics = run_evaluate(evaluate_script, gt_json, pred_json)
        if metrics is None:
            log.warning(f'  {ckpt_name} 评估失败，跳过')
            continue

        con_f1     = metrics.get('COn',     {}).get('f1', 0.0)
        conp_f1    = metrics.get('COnP',    {}).get('f1', 0.0)
        conpoff_f1 = metrics.get('COnPOff', {}).get('f1', 0.0)

        log.info(f'  测试集结果  COn={con_f1:.4f}  COnP={conp_f1:.4f}  COnPOff={conpoff_f1:.4f}')
        log.info('-' * 70)

        all_results.append({
            'checkpoint':    ckpt_name,
            'epoch':         epoch,
            'val_COnP_f1':   round(val_f1, 6) if val_f1 else None,
            'onset_thresh':  onset_thresh,
            'frame_thresh':  frame_thresh,
            'test_COn':      round(con_f1, 6),
            'test_COnP':     round(conp_f1, 6),
            'test_COnPOff':  round(conpoff_f1, 6),
            'test_COn_P':    round(metrics.get('COn',     {}).get('precision', 0.0), 6),
            'test_COn_R':    round(metrics.get('COn',     {}).get('recall',    0.0), 6),
            'test_COnP_P':   round(metrics.get('COnP',    {}).get('precision', 0.0), 6),
            'test_COnP_R':   round(metrics.get('COnP',    {}).get('recall',    0.0), 6),
            'test_COnPOff_P':round(metrics.get('COnPOff', {}).get('precision', 0.0), 6),
            'test_COnPOff_R':round(metrics.get('COnPOff', {}).get('recall',    0.0), 6),
        })

        # 每跑完一个就保存一次，防止中途中断丢失结果
        _save_results(all_results, project_dir / (args.results_file + '.json'),
                      project_dir / (args.results_file + '.csv'))

    # 最终汇总
    log.info('=' * 70)
    log.info('全部完成！汇总（按 test_COnP 降序）：')
    log.info(f'{"Checkpoint":<35} {"Epoch":>6} {"val_COnP":>9} {"COn":>7} {"COnP":>7} {"COnPOff":>9}')
    log.info('-' * 70)
    for r in sorted(all_results, key=lambda x: x['test_COnP'], reverse=True):
        log.info(f'{r["checkpoint"]:<35} {r["epoch"]:>6} '
                 f'{(r["val_COnP_f1"] or 0):>9.4f} '
                 f'{r["test_COn"]:>7.4f} {r["test_COnP"]:>7.4f} {r["test_COnPOff"]:>9.4f}')

    _save_results(all_results, project_dir / (args.results_file + '.json'),
                  project_dir / (args.results_file + '.csv'))
    log.info(f'\n结果已保存：')
    log.info(f'  JSON: {args.results_file}.json')
    log.info(f'  CSV:  {args.results_file}.csv')


def _save_results(results, json_path, csv_path):
    """保存结果到 JSON 和 CSV"""
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    main()
