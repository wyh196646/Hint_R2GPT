
# from __future__ import annotations

# import argparse
# import math
# from pathlib import Path
# from typing import List, Tuple

# import lightning.pytorch as pl
# import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF
# from tqdm import tqdm

# # -----------------------------------------------------------------------------
# # Project‑specific imports – make sure these are on PYTHONPATH
# # -----------------------------------------------------------------------------
# from dataset.data_module import DataModule  # noqa: E402
# from models.R2GenGPT import R2GenGPT       # noqa: E402
# from configs.config import parser as base_parser  # noqa: E402

# # ================================ Defaults ===================================
# DEFAULTS = dict(
#     # --- data ----------------------------------------------------------------
#     dataset="mimic_cxr",
#     annotation="/data2/yuhaowang/MIMIC-CXR/mimic_annotation_all.json",
#     base_dir="/data2/yuhaowang/MIMIC-CXR/files/",
#     # --- checkpoint ----------------------------------------------------------
#     delta_file="/home/yuhaowang/project/report_generation/TRRG/R2GenGPT/deep_checkpoint_step42310.pth",
#     # --- inference hyper‑params ---------------------------------------------
#     batch_size=16,
#     max_length=100,
#     min_new_tokens=80,
#     max_new_tokens=120,
#     repetition_penalty=2.0,
#     length_penalty=2.0,
#     # --- device / output -----------------------------------------------------
#     device="cuda:0",
#     output_dir="./outputs/mimic_cxr/attn_v2",  # 🆕
#     # --- precision -----------------------------------------------------------
#     fp16=False,
#     # --- layers to visualise --------------------------------------------------
#     layers=(4, 8, 12, 16, 20, 24, 28, 32),  # 1‑based layer indices
# )

# # ========================== Helper functions =================================

# def factorize_grid(n_patch: int) -> Tuple[int, int]:
#     """Return (gh, gw) so that gh * gw == n_patch and |gh‑gw| minimal."""
#     root = int(math.sqrt(n_patch))
#     for h in range(root, 0, -1):
#         if n_patch % h == 0:
#             return h, n_patch // h
#     return 1, n_patch


# def cls_attention_to_heatmap(cls_attn: torch.Tensor,
#                              hw: Tuple[int, int],
#                              grid_hw: Tuple[int, int] | None = None) -> torch.Tensor:
#     """Convert CLS‑to‑patch attention to a [0,1] heat‑map of size ``hw``."""
#     if grid_hw is None:
#         grid_hw = factorize_grid(cls_attn.numel())
#     gh, gw = grid_hw
#     grid = cls_attn.view(1, 1, gh, gw)
#     heat = F.interpolate(grid, size=hw, mode="bilinear", align_corners=False)
#     heat = heat.squeeze().clamp_(0, 1)
#     heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-5)
#     return heat


# def save_overlay(img_tensor: torch.Tensor,
#                  heatmap: torch.Tensor,
#                  labels: List[str] | None,
#                  save_path: Path,
#                  grid_hw: Tuple[int, int],
#                  alpha: float = 0.45) -> None:
#     """Save PIL image with heat‑map and optional patch labels."""
#     img = TF.to_pil_image(img_tensor.cpu().clamp(0, 1))
#     plt.figure(figsize=(6, 6))
#     plt.imshow(img)
#     plt.imshow(heatmap.cpu(), cmap="inferno", alpha=alpha)

#     if labels is not None:
#         gh, gw = grid_hw
#         H, W = img.height, img.width
#         pw, ph = W / gw, H / gh
#         for idx, token in enumerate(labels):
#             i, j = divmod(idx, gw)
#             x = j * pw + pw / 2
#             y = i * ph + ph / 2
#             plt.text(x, y, token, fontsize=6, ha="center", va="center",
#                      color="white", bbox=dict(boxstyle="round,pad=0.1",
#                                               fc="black", alpha=0.6))

#     plt.axis("off")
#     plt.tight_layout(pad=0)
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(save_path, dpi=300)
#     plt.close()

# # ======================= Monkey‑patch generate_reports =======================

# def _add_generate_reports():
#     """Attach `generate_reports` to **R2GenGPT** if it's missing."""
#     if hasattr(R2GenGPT, "generate_reports"):
#         return

#     def _generate_reports(self: R2GenGPT, images, **gen_kwargs):
#         img_embeds, atts_img = self.encode_img(images)
#         img_embeds = self.layer_norm(img_embeds)
#         img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

#         dtype = self.llama_model.dtype
#         img_embeds = img_embeds.to(dtype)

#         batch_size = img_embeds.shape[0]
#         bos = (torch.ones([batch_size, 1], dtype=torch.long, device=img_embeds.device)
#                * self.llama_tokenizer.bos_token_id)
#         bos_embeds = self.embed_tokens(bos).to(dtype)
#         atts_bos = atts_img[:, :1]

#         inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
#         attention_mask = torch.cat([atts_bos, atts_img], dim=1)

#         outputs = self.llama_model.generate(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             **gen_kwargs,
#         )
#         return [self.decode(o) for o in outputs]

#     R2GenGPT.generate_reports = _generate_reports  # type: ignore


# # ========================== Attention extraction ============================

# def most_attended_tokens(attn: torch.Tensor,
#                          input_ids: torch.Tensor,
#                          patch_range: slice,
#                          text_range: slice,
#                          tokenizer) -> List[str]:
#     """Return the *single* text token that attends most to each patch.

#     * ``attn`` has shape (num_heads, seq_len_q, seq_len_k).
#     * ``patch_range`` and ``text_range`` specify slices into the *key* and *query*
#       axes respectively, assuming queries=text tokens, keys=patch tokens.
#     """
#     # Mean over heads → (seq_q, seq_k)
#     attn_mean = attn.mean(dim=0)
#     labels: List[str] = []
#     for k in range(patch_range.start, patch_range.stop):
#         scores = attn_mean[text_range, k]  # (num_text_q,)
#         if scores.numel() == 0:
#             labels.append("<s>")
#             continue
#         q_rel = scores.argmax().item()
#         q_idx = text_range.start + q_rel
#         token_id = input_ids[q_idx].item()
#         token = tokenizer.decode([token_id]).strip()
#         labels.append(token if token else "<unk>")
#     return labels


# # ================================ Inference ==================================

# def run_inference(args: argparse.Namespace):
#     pl.seed_everything(42)

#     # 1️⃣ Data‑module
#     dm = DataModule(args)
#     dm.setup("test")
#     test_loader = dm.test_dataloader()

#     # 2️⃣ Model
#     precision = torch.float16 if args.fp16 else torch.float32
#     model = R2GenGPT(args).to(dtype=precision, device=args.device).eval()

#     # 3️⃣ Output folders
#     out_root = Path(args.output_dir)
#     attn_dir = out_root / "attn_maps" / "test"
#     rpt_dir = out_root / "reports" / "test"
#     attn_dir.mkdir(parents=True, exist_ok=True)
#     rpt_dir.mkdir(parents=True, exist_ok=True)

#     # 4️⃣ Generation hyper‑parameters
#     gen_kwargs = dict(
#         max_new_tokens=args.max_new_tokens,
#         min_new_tokens=args.min_new_tokens,
#         repetition_penalty=args.repetition_penalty,
#         length_penalty=args.length_penalty,
#         num_beams=3,
#         do_sample=False,
#     )

#     inspect_layers: Tuple[int, ...] = tuple(int(l) for l in args.layers)

#     with torch.no_grad():
#         for batch in tqdm(test_loader, total=len(test_loader), ncols=100):

#             images = [img.to(args.device, dtype=precision) for img in batch["image"]]
#             first_view = images[0]                        # Tensor, shape (B, C, H, W)
#             study_ids = batch["id"]                      # list[str]

#             # 1️⃣ Generate report text
#             reports = model.generate_reports(images=images, **gen_kwargs)

#             # 2️⃣ Build a *single* forward pass to grab attentions (no KV‑cache)
#             tokenizer = model.llama_tokenizer
#             txt_tokens = tokenizer(reports, return_tensors="pt", padding=True,
#                                     truncation=True, max_length=args.max_length).input_ids.to(args.device)
#             txt_embeds = model.embed_tokens(txt_tokens).to(dtype=precision)

#             # Compose full sequence: BOS + IMG + TXT
#             img_embeds, atts_img = model.encode_img(images)
#             img_embeds = model.layer_norm(img_embeds)
#             img_embeds, _ = model.prompt_wrap(img_embeds, atts_img)
#             bos = torch.full((1, 1), tokenizer.bos_token_id, device=args.device)
#             bos_embeds = model.embed_tokens(bos).to(dtype=precision)

#             inputs_embeds = torch.cat([bos_embeds, img_embeds, txt_embeds], dim=1)
#             attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long,
#                                          device=args.device)

#             llama_out = model.llama_model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 output_attentions=True,
#                 use_cache=False,
#                 return_dict=True,
#             )

#             attentions = llama_out.attentions  # tuple[num_layers] of (B, nH, L, L)

#             # 3️⃣ Save layer‑wise overlays
#             _, _, H, W = first_view.shape
#             N = img_embeds.shape[1]
#             gh, gw = factorize_grid(N)
#             patch_slice = slice(1, 1 + N)               # keys
#             text_slice = slice(1 + N, inputs_embeds.size(1))  # queries

#             for b_idx, (img_tensor, fname, rpt) in enumerate(zip(first_view,
#                                                                   study_ids,
#                                                                   reports)):
#                 # Write report text to disk
#                 (rpt_dir / f"{fname}.txt").write_text(rpt + "\n", encoding="utf-8")

#                 input_ids_full = torch.cat([bos, torch.full((1, N), tokenizer.pad_token_id, device=args.device), txt_tokens[b_idx:b_idx+1]], dim=1).squeeze(0)

#                 for layer_idx in inspect_layers:
#                     if layer_idx < 1 or layer_idx > len(attentions):
#                         continue  # skip invalid indices
#                     attn_mat = attentions[layer_idx - 1][b_idx]  # (nH, L, L)
#                     labels = most_attended_tokens(attn_mat, input_ids_full,
#                                                   patch_slice, text_slice,
#                                                   tokenizer)

#                     ve_out = model.visual_encoder(
#                         img_tensor.unsqueeze(0),
#                         output_hidden_states=True,
#                         return_dict=True,
#                     )
#                     patch_embeds = ve_out.last_hidden_state
#                     q_global = patch_embeds.mean(dim=1, keepdim=True)
#                     logits = torch.matmul(q_global, patch_embeds.transpose(-1, -2)) / math.sqrt(patch_embeds.size(-1))
#                     global2patch = torch.softmax(logits, dim=-1).squeeze(0)
#                     heat = cls_attention_to_heatmap(global2patch.float(), (H, W), (gh, gw))

#                     save_overlay(img_tensor.float(), heat, labels,
#                                  attn_dir / f"{fname}_layer{layer_idx}.png",
#                                  grid_hw=(gh, gw))

# # =============================== CLI parsing =================================

# def build_parser() -> argparse.ArgumentParser:
#     infer_parser = argparse.ArgumentParser(parents=[base_parser], add_help=False)

#     existing_opts = {opt for act in infer_parser._actions for opt in act.option_strings}
#     for k, v in DEFAULTS.items():
#         flag = f"--{k}"
#         if flag in existing_opts:
#             infer_parser.set_defaults(**{k: v})
#         else:
#             if isinstance(v, bool):
#                 infer_parser.add_argument(flag, action="store_true" if not v else "store_false")
#             elif isinstance(v, tuple):
#                 infer_parser.add_argument(flag, nargs="+", type=int, default=v)
#             else:
#                 infer_parser.add_argument(flag, default=v, type=type(v))
#     return infer_parser

# # =================================== Main ===================================
# if __name__ == "__main__":
#     _add_generate_reports()
#     parser = build_parser()
#     cli_args = parser.parse_args()
#     run_inference(cli_args)



## 用来观测整体的attention map和相应的report 
import argparse
import math
from pathlib import Path
from typing import Tuple

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Project‑specific imports – make sure these are on PYTHONPATH
# -----------------------------------------------------------------------------
from dataset.data_module import DataModule  # noqa: E402
from models.R2GenGPT import R2GenGPT       # noqa: E402
from configs.config import parser as base_parser  # noqa: E402

# ================================ Defaults ===================================
DEFAULTS = dict(
    # --- data ----------------------------------------------------------------
    dataset="mimic_cxr",
    annotation="/data2/yuhaowang/MIMIC-CXR/mimic_annotation_all.json",
    base_dir="/data2/yuhaowang/MIMIC-CXR/files/",
    # --- checkpoint ----------------------------------------------------------
    delta_file="/home/yuhaowang/project/report_generation/TRRG/R2GenGPT/deep_checkpoint_step42310.pth",  # supply your own .ckpt or .pth
    # --- inference hyper‑params ---------------------------------------------
    batch_size=16,
    max_length=100,
    min_new_tokens=80,
    max_new_tokens=120,
    repetition_penalty=2.0,
    length_penalty=2.0,
    # --- device / output -----------------------------------------------------
    device="cuda:0",
    output_dir="/home/yuhaowang/project/report_generation/TRRG/R2GenGPT/outputs/mimic_cxr/attn_v1",
    # --- precision -----------------------------------------------------------
    fp16=False,  # if True → runs model & data in float16
)

# ========================== Helper functions =================================

def factorize_grid(n_patch: int) -> Tuple[int, int]:
    """Return (gh, gw) so that gh * gw == n_patch and |gh‑gw| minimal."""
    root = int(math.sqrt(n_patch))
    for h in range(root, 0, -1):
        if n_patch % h == 0:
            return h, n_patch // h
    return 1, n_patch  # fallback (unlikely)


def cls_attention_to_heatmap(cls_attn: torch.Tensor,
                             hw: Tuple[int, int],
                             grid_hw: Tuple[int, int] | None = None) -> torch.Tensor:
    """Convert CLS‑to‑patch attention to a [0,1] heat‑map of size ``hw``.

    * ``cls_attn`` – (N,) tensor, where N = #patches.
    * ``hw``        – (H, W) size of original image.
    * ``grid_hw``   – optional (gh, gw) patch grid size; if ``None`` it is
                      inferred automatically.
    """
    if grid_hw is None:
        grid_hw = factorize_grid(cls_attn.numel())
    gh, gw = grid_hw
    grid = cls_attn.view(1, 1, gh, gw)
    heat = F.interpolate(grid, size=hw, mode="bilinear", align_corners=False)
    heat = heat.squeeze().clamp_(0, 1)  # (H, W)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-5)
    return heat


def save_overlay(img_tensor: torch.Tensor,
                 heatmap: torch.Tensor,
                 save_path: Path,
                 alpha: float = 0.45) -> None:
    """Save PIL image with heat‑map overlay."""
    img = TF.to_pil_image(img_tensor.cpu().clamp(0, 1))
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(heatmap.cpu(), cmap="inferno", alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

# ======================= Monkey‑patch generate_reports =======================

def _add_generate_reports():
    """Attach `generate_reports` to **R2GenGPT** if it's missing."""
    if hasattr(R2GenGPT, "generate_reports"):
        return

    def _generate_reports(self: R2GenGPT, images, **gen_kwargs):
        # Encode images → embeddings
        img_embeds, atts_img = self.encode_img(images)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        # match dtype with LLaMA
        dtype = self.llama_model.dtype
        img_embeds = img_embeds.to(dtype)

        batch_size = img_embeds.shape[0]
        bos = (torch.ones([batch_size, 1], dtype=torch.long, device=img_embeds.device)
               * self.llama_tokenizer.bos_token_id)
        bos_embeds = self.embed_tokens(bos).to(dtype)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        return [self.decode(o) for o in outputs]

    R2GenGPT.generate_reports = _generate_reports  # type: ignore

# ================================ Inference ==================================

def run_inference(args):
    pl.seed_everything(42)

    # 1️⃣ Data‑module
    dm = DataModule(args)
    dm.setup("test")
    test_loader = dm.test_dataloader()

    # 2️⃣ Model
    precision = torch.float16 if args.fp16 else torch.float32
    model = R2GenGPT(args).to(dtype=precision, device=args.device).eval()

    # 3️⃣ Output folders
    out_root = Path(args.output_dir)
    attn_dir = out_root / "attn_maps" / "test"
    rpt_dir = out_root / "reports" / "test"
    attn_dir.mkdir(parents=True, exist_ok=True)
    rpt_dir.mkdir(parents=True, exist_ok=True)

    # 4️⃣ Generation hyper‑parameters
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        num_beams=3,
        do_sample=False,  # greedy / beam search – no top_p/temperature
    )

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), ncols=100):

            images = [img.to(args.device, dtype=precision) for img in batch["image"]]
            first_view = images[0]                        # Tensor, shape (B, C, H, W)
            study_ids  = batch["id"]                     # list[str]

            # 1️⃣ 生成报告文本 —— 原逻辑不变
            reports = model.generate_reports(images=images, **gen_kwargs)

            # 2️⃣ 计算全局注意力 (no-CLS) -------------------------------------- 🔧②
            # 让 visual_encoder 返回 hidden states
            ve_out = model.visual_encoder(
                first_view,
                output_hidden_states=True,   # 👈 关键
                return_dict=True
            )
            patch_embeds = ve_out.last_hidden_state       # (B, N, D)
            q_global = patch_embeds.mean(dim=1, keepdim=True)        # (B, 1, D)
            attn_logits = torch.matmul(                   # (B, 1, N)
                q_global, patch_embeds.transpose(-1, -2)
            ) / math.sqrt(patch_embeds.size(-1))
            global2patch = torch.softmax(attn_logits, dim=-1).squeeze(1)  # (B, N)

            # ------------------------------------------------------------------
            # (3) 保存热图 & 报告 —— 用新向量替换 cls2patch                 🔧③
            # ------------------------------------------------------------------
            _, _, H, W = first_view.shape
            N = global2patch.shape[1]
            gh, gw = factorize_grid(N)

            for img_tensor, heat_vec, fname, rpt in zip(first_view,
                                                        global2patch,
                                                        study_ids,
                                                        reports):
                heat = cls_attention_to_heatmap(heat_vec.float(),
                                                (H, W),
                                                (gh, gw))
                save_overlay(img_tensor.float(), heat,
                             attn_dir / f"{fname}.png")
                (rpt_dir / f"{fname}.txt").write_text(rpt + "\n",
                                                      encoding="utf-8")
     
# =============================== CLI parsing =================================

def build_parser() -> argparse.ArgumentParser:
    infer_parser = argparse.ArgumentParser(parents=[base_parser], add_help=False)

    # Merge defaults & add `--fp16` flag
    existing_opts = {opt for act in infer_parser._actions for opt in act.option_strings}
    for k, v in DEFAULTS.items():
        flag = f"--{k}"
        if flag in existing_opts:
            infer_parser.set_defaults(**{k: v})
        else:
            if isinstance(v, bool):
                infer_parser.add_argument(flag, action="store_true" if not v else "store_false")
            else:
                infer_parser.add_argument(flag, default=v, type=type(v))
    return infer_parser

# =================================== Main ===================================
if __name__ == "__main__":
    _add_generate_reports()
    parser = build_parser()
    cli_args = parser.parse_args()
    run_inference(cli_args)
