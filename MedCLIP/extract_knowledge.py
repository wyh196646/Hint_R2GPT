# -*- coding: utf-8 -*-
import os
import json
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import torch
import torch.distributed as dist
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

from medclip import PromptClassifier
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts


# -------------------------
# PromptBank
# -------------------------
@dataclass
class PromptBank:
    raw_cls_prompts: Dict[str, List[str]]                         # {class: [prompt, ...]}
    prompt_map_by_class: Dict[str, Dict[str, Tuple[str, str, str]]]  # {class: {prompt: (sev,sub,loc)}}
    cls_prompt_inputs: Dict[str, Dict[str, torch.Tensor]]         # tokenizer 结果（可直接复用）

def build_prompt_bank(n_prompts_per_class: int = 10,
                      seed: Optional[int] = None) -> PromptBank:
    raw_cls_prompts, prompt_map_by_class = generate_chexpert_class_prompts(
        n=n_prompts_per_class, return_map=True, seed=seed
    )
    cls_prompt_inputs = process_class_prompts(raw_cls_prompts)
    return PromptBank(
        raw_cls_prompts=raw_cls_prompts,
        prompt_map_by_class=prompt_map_by_class,
        cls_prompt_inputs=cls_prompt_inputs
    )


# -------------------------
# 单张图 → K 个三元组
# -------------------------
@torch.inference_mode()
def infer_triples_via_best_prompt(
    image_path: str,
    processor: MedCLIPProcessor,
    clf: PromptClassifier,
    prompt_bank: PromptBank,
    device: torch.device,
    k_findings: int = 5,
    threshold_finding: float = 0.35,
) -> Dict[str, Any]:

    raw_cls_prompts = prompt_bank.raw_cls_prompts
    prompt_map_by_class = prompt_bank.prompt_map_by_class
    cls_prompt_inputs = prompt_bank.cls_prompt_inputs

    # 1) 读图（with 确保及时释放文件句柄）
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    # 2) 前向：类级 logits + 每类最佳 prompt 的索引
    inputs_img = processor(images=image, return_tensors="pt")
    inputs_img = {k: v.to(device) for k, v in inputs_img.items()}
    inputs_img["prompt_inputs"] = cls_prompt_inputs  # classifier 内部会 .to(device)

    out = clf(**inputs_img, return_prompt_details=True)
    class_names: List[str] = out["class_names"]
    class_logits: torch.Tensor = out["logits"].squeeze(0)  # [C]
    p_finding_all = class_logits.tolist()

    # 3) Top-K（带阈值）
    cls_with_prob = list(zip(class_names, p_finding_all))
    cls_with_prob.sort(key=lambda x: x[1], reverse=True)
    filtered = [c for c in cls_with_prob if c[1] >= threshold_finding]
    selected = filtered[:k_findings] if filtered else cls_with_prob[:k_findings]

    # “No Finding”独占（可选）
    if len(selected) >= 2 and selected[0][0].lower() in {"no finding", "no findings"}:
        if (selected[0][1] - selected[1][1]) > 0.20:
            selected = [selected[0]]

    # 4) 映射三元组
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    per_class_best_idx: List[int] = out["per_class_best_idx"]

    triples_out: List[Dict[str, Any]] = []
    for finding, p_f in selected:
        cls_i = name_to_idx[finding]
        best_idx = per_class_best_idx[cls_i]

        best_prompt_text = raw_cls_prompts[finding][best_idx]
        sev, sub, loc = prompt_map_by_class[finding][best_prompt_text]

        triples_out.append({
            "finding": finding,
            "p_finding": float(p_f),
            "best_prompt": best_prompt_text,
            "severity": (sev or None) if (sev and sev.strip()) else None,
            "subtype":  (sub or None) if (sub and sub.strip()) else None,
            "location": (loc or None) if (loc and loc.strip()) else None
        })

    triples_out.sort(key=lambda x: x["p_finding"], reverse=True)

    return {
        "image": image_path,
        "topk_findings": [{"name": n, "p": float(p)} for (n, p) in selected],
        "triples": triples_out
    }


# -------------------------
# 分布式/设备初始化
# -------------------------
def init_distributed(distributed: bool):
    """
    返回: (is_dist, world_size, rank, local_rank, master_rank)
    """
    if distributed and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, world_size, rank, local_rank, 0
    else:
        return False, 1, 0, 0, 0

def cleanup_distributed(is_dist: bool):
    if is_dist and dist.is_initialized():
        dist.destroy_process_group()


# -------------------------
# I/O 与进度工具
# -------------------------
def load_split_samples(annotation_path: str, split: str) -> List[Dict[str, Any]]:
    """
    期望 annotation 结构为：
    {
      "train": [{"id": "...", "image_path": ["a.jpg","b.jpg"], ...}, ...],
      "val":   [...],
      "test":  [...]
    }
    只保留 id 与 image_path（标准化为 list）
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        meta_all = json.load(f)
    items = meta_all[split]
    out = []
    for it in items:
        sid = it.get("id", None)
        # 兼容 image_path 与 image_paths 两种字段
        paths = it.get("image_path", it.get("image_paths", []))
        if not isinstance(paths, list):
            paths = [paths]
        out.append({"id": sid, "image_paths": paths})
    return out

def partition_by_rank(samples: List[Any], world_size: int, rank: int) -> List[Any]:
    """简单按步长切分，确保各 rank 读相同列表但只处理自己的切片。"""
    return samples[rank::world_size]

def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    cnt = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            cnt += 1
    return cnt


# -------------------------
# 快速测试（仅 rank0，本地顺序执行）
# -------------------------
def quick_test(args, processor, clf, prompt_bank, device, samples: List[Dict[str, Any]]):
    n = min(args.quick_test, len(samples))
    subset = samples[:n]
    test_part = os.path.join(args.save_dir, "part_quick.jsonl")
    if _HAS_TQDM:
        pbar = tqdm(total=n, desc="[quick_test]", dynamic_ncols=True)
    else:
        pbar = None
        print(f"[quick_test] running {n} samples...")

    os.makedirs(args.save_dir, exist_ok=True)
    with open(test_part, "w", encoding="utf-8") as fw:
        for i, sample in enumerate(subset, 1):
            sid = sample["id"]
            rel_paths: List[str] = sample["image_paths"] or []
            per_view_results: List[Dict[str, Any]] = []

            for rel in rel_paths:
                abs_path = os.path.join(args.base_dir, rel)
                try:
                    res = infer_triples_via_best_prompt(
                        image_path=abs_path,
                        processor=processor,
                        clf=clf,
                        prompt_bank=prompt_bank,
                        device=device,
                        k_findings=args.k_findings,
                        threshold_finding=args.threshold_finding,
                    )
                except Exception as e:
                    res = {"image": abs_path, "error": f"{type(e).__name__}: {str(e)}"}
                per_view_results.append(res)

            out_obj = {"id": sid, "num_views": len(rel_paths), "views": per_view_results}
            fw.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fw.flush()
            if pbar:
                pbar.update(1)
            elif i % 5 == 0:
                print(f"[quick_test] {i}/{n}")

    # 合并为测试结果 JSON
    test_out = args.test_output_json or (os.path.splitext(args.output_json)[0] + ".quick.json")
    merged: List[Dict[str, Any]] = []
    with open(test_part, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if line:
                try:
                    merged.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    with open(test_out, "w", encoding="utf-8") as fw:
        json.dump(merged, fw, ensure_ascii=False, indent=2)
    if pbar:
        pbar.close()
    print(f"[quick_test] done → {test_out}")


# -------------------------
# 主流程
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--base_dir", type=str, default='/data2/yuhaowang/iu_xray/images', help="图像根目录")
    parser.add_argument("--annotation", type=str, default='/data2/yuhaowang/iu_xray/annotation.json', help="注释 JSON（包含 train/val/test）")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--output_json", type=str, default='./iuxray_train.json', help="最终合并输出 JSON 路径")
    parser.add_argument("--save_dir", type=str, default="./", help="各rank临时JSONL目录")

    # Prompt / 推理
    parser.add_argument("--k_findings", type=int, default=5)
    parser.add_argument("--threshold_finding", type=float, default=0.35)
    parser.add_argument("--n_prompts_per_class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # 并行 & 日志 & 刷新
    parser.add_argument("--distributed", action="store_true", help="使用 torchrun 的 DDP 推理")
    parser.add_argument("--log_every", type=int, default=100, help="每N个样本打印一次进度（无tqdm时生效）")
    parser.add_argument("--flush_every", type=int, default=20, help="每N个样本显式 flush 一次 JSONL")
    parser.add_argument("--progress_poll_sec", type=float, default=2.0, help="rank0 统计全局进度的轮询秒数")

    # 快速测试
    parser.add_argument("--quick_test", type=int, default=0, help=">0 时仅在 rank0 跑前 N 个样本并退出")
    parser.add_argument("--test_output_json", type=str, default=None, help="快速测试的输出 JSON 路径")

    # 调试：仅取前 K 个样本做全量流程
    parser.add_argument("--max_samples", type=int, default=-1, help="全量流程中截断样本数，-1 表示全部")

    args = parser.parse_args()

    # 分布式初始化
    is_dist, world_size, rank, local_rank, master_rank = init_distributed(args.distributed)

    # 设备
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    # 模型与处理器（每个进程各持有一份）
    processor = MedCLIPProcessor()
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()
    model = model.to(device).eval()
    clf = PromptClassifier(model, ensemble=True).to(device).eval()

    # PromptBank（每个进程各建一次）
    prompt_bank = build_prompt_bank(n_prompts_per_class=args.n_prompts_per_class, seed=args.seed)

    # 读取全部样本，仅保留路径
    all_samples = load_split_samples(args.annotation, args.split)
    if args.max_samples is not None and args.max_samples > 0:
        all_samples = all_samples[:args.max_samples]

    # ---------- 快速测试：仅 rank0 执行并直接退出 ----------
    if args.quick_test and rank == master_rank:
        quick_test(args, processor, clf, prompt_bank, device, all_samples)
        # 退出，不进入全量流程
        return
    elif args.quick_test and rank != master_rank:
        # 其他 rank 直接返回
        return

    # ---------- 全量流程 ----------
    # 按 rank 切分
    local_samples = partition_by_rank(all_samples, world_size, rank)

    # 准备临时目录/文件
    os.makedirs(args.save_dir, exist_ok=True)
    part_path = os.path.join(args.save_dir, f"part_rank{rank}.jsonl")
    if rank == master_rank:
        print(f"[rank {rank}] writing part file → {part_path}")
        print(f"[rank {rank}] final merge output → {args.output_json}")
        total_global = len(all_samples)
        # rank0: 全局进度条（根据各分片文件行数统计）
        if _HAS_TQDM:
            pbar = tqdm(total=total_global, desc="[global]", dynamic_ncols=True)
        else:
            pbar = None
            print(f"[global] total samples: {total_global}")
        last_poll = 0.0
    else:
        pbar = None

    processed = 0
    t0 = time.time()
    with open(part_path, "w", encoding="utf-8") as fw:
        for sample in local_samples:
            sid = sample["id"]
            rel_paths: List[str] = sample["image_paths"] or []

            per_view_results: List[Dict[str, Any]] = []
            for rel in rel_paths:
                abs_path = os.path.join(args.base_dir, rel)
                try:
                    res = infer_triples_via_best_prompt(
                        image_path=abs_path,
                        processor=processor,
                        clf=clf,
                        prompt_bank=prompt_bank,
                        device=device,
                        k_findings=args.k_findings,
                        threshold_finding=args.threshold_finding,
                    )
                except Exception as e:
                    res = {"image": abs_path, "error": f"{type(e).__name__}: {str(e)}"}
                per_view_results.append(res)

            out_obj = {
                "id": sid,
                "num_views": len(rel_paths),
                "views": per_view_results,
            }
            fw.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            processed += 1

            # flush 周期，保证 rank0 能准确统计全局行数
            if (processed % max(1, args.flush_every)) == 0:
                fw.flush()

            # 无 tqdm 时的简易日志
            if (not _HAS_TQDM) and (rank == master_rank) and (processed % max(1, args.log_every) == 0):
                dt = time.time() - t0
                print(f"[rank {rank}] processed={processed}/{len(local_samples)} elapsed={dt:.1f}s")

            # rank0 定期统计全局进度（汇总所有 part_rank*.jsonl 的行数）
            if rank == master_rank and pbar is not None:
                now = time.time()
                if (now - last_poll) >= args.progress_poll_sec:
                    total_done = 0
                    for r in range(world_size):
                        pr = os.path.join(args.save_dir, f"part_rank{r}.jsonl")
                        total_done += count_lines(pr)
                    pbar.n = min(total_done, pbar.total)
                    pbar.refresh()
                    last_poll = now

        # 完成后再 flush 一次
        fw.flush()

    # 同步
    if is_dist:
        dist.barrier()

    # rank0 进度条补齐并关闭
    if rank == master_rank and pbar is not None:
        # 最终统计一次
        total_done = 0
        for r in range(world_size):
            pr = os.path.join(args.save_dir, f"part_rank{r}.jsonl")
            total_done += count_lines(pr)
        pbar.n = min(total_done, pbar.total)
        pbar.refresh()
        pbar.close()

    # 合并（仅rank0）
    if rank == master_rank:
        merged: List[Dict[str, Any]] = []
        for r in range(world_size):
            p = os.path.join(args.save_dir, f"part_rank{r}.jsonl")
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        merged.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        out_path = args.output_json
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(merged, fw, ensure_ascii=False, indent=2)
        print(f"[rank {rank}] 合并完成，共 {len(merged)} 条样本 → {out_path}")

    cleanup_distributed(is_dist)


if __name__ == "__main__":
    main()
