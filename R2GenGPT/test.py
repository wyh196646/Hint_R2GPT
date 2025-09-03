import os
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    SwinModel,
)

# ============ 可选：视觉侧 LoRA（原入口保留） ============
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# ==============================================================
# 1) 多分支 LoRA（默认 num_branches=2）
# ==============================================================

class MultiBranchLoRALinear(nn.Module):

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0, num_branches: int = 2):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.base = base_linear
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.r = r
        self.scale = alpha / max(r, 1)
        self.num_branches = num_branches
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A/B 列表
        self.As = nn.ModuleList([nn.Linear(self.in_features, r, bias=False) for _ in range(num_branches)])
        self.Bs = nn.ModuleList([nn.Linear(r, self.out_features, bias=False) for _ in range(num_branches)])

        self._reset_parameters()

        # 合成权重（不参与梯度）
        self.merge_weights = [1.0] + [0.0] * (num_branches - 1)  # 默认只开分支0

        # 哪一条分支可训练（int 或 None）
        self.train_branch: Optional[int] = 0
        self._apply_train_switch()

    def _reset_parameters(self):
        for A, B in zip(self.As, self.Bs):
            nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
            nn.init.zeros_(B.weight)

    @torch.no_grad()
    def set_merge_weights(self, weights: List[float]):
        assert len(weights) == self.num_branches
        self.merge_weights = [float(w) for w in weights]

    @torch.no_grad()
    def set_train_branch(self, branch: Optional[int]):
        # branch ∈ {0..num_branches-1} or None
        self.train_branch = branch
        self._apply_train_switch()

    def _apply_train_switch(self):
        # 先全部冻结
        for A, B in zip(self.As, self.Bs):
            for p in list(A.parameters()) + list(B.parameters()):
                p.requires_grad = False
        # 打开指定分支
        if self.train_branch is not None:
            for p in list(self.As[self.train_branch].parameters()) + list(self.Bs[self.train_branch].parameters()):
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 关键修复：将输入激活 x 转成与底层权重相同的 dtype（通常 float16）
        if x.dtype != self.base.weight.dtype:
            x = x.to(self.base.weight.dtype)

        y = F.linear(x, self.base.weight, self.base.bias)
        if self.r == 0:
            return y
        z = self.dropout(x)
        # 累加每条分支的增量
        for w, A, B in zip(self.merge_weights, self.As, self.Bs):
            if w != 0.0:
                y = y + w * (B(A(z)) * self.scale)
        return y


def _replace_module(parent: nn.Module, name: str, new: nn.Module):
    setattr(parent, name, new)


def apply_multi_branch_lora(
    llama_model: nn.Module,
    target_keywords=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    num_branches: int = 2,
):
    """
    遍历 LLaMA，凡是子模块名包含 target_keywords 且为 nn.Linear 的，替换为 MultiBranchLoRALinear。
    """
    replaced = 0
    for _, module in llama_model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(k in child_name for k in target_keywords):
                dual = MultiBranchLoRALinear(child, r=r, alpha=alpha, dropout=dropout, num_branches=num_branches)
                _replace_module(module, child_name, dual)
                replaced += 1
    print(f"[MB-LoRA] Injected into {replaced} linear layers.")


def set_all_mb_lora_train_branch(model: nn.Module, branch: Optional[int]):
    for m in model.modules():
        if isinstance(m, MultiBranchLoRALinear):
            m.set_train_branch(branch)


def set_all_mb_lora_merge_weights(model: nn.Module, weights: List[float]):
    for m in model.modules():
        if isinstance(m, MultiBranchLoRALinear):
            m.set_merge_weights(weights)


# ==============================================================
# 2) 模型
# ==============================================================

class MMReportModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        print(f"Loading vision encoder: {args.vision_model}")
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)

        # 视觉侧 LoRA（可选）
        if args.vis_use_lora and PEFT_AVAILABLE:
            peft_config_visual = LoraConfig(
                r=args.vis_r,
                lora_alpha=args.vis_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                modules_to_save=["classifier"],
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print("Loading vision encoder with LoRA -- Done")
        elif args.freeze_vm:
            for _, p in self.visual_encoder.named_parameters():
                p.requires_grad = False
            print(f"Loading Frozen vision encoder: {args.vision_model} -- Done")
        else:
            print(f"Loading Trainable vision encoder: {args.vision_model} -- Done")

        print("Loading LLAMA")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        # 若 pad_token_id 为空，则对齐 EOS，避免 pad=0 导致 label masking 误伤
        if self.llama_tokenizer.pad_token_id is None:
            self.llama_tokenizer.pad_token_id = self.llama_tokenizer.eos_token_id

        if args.low_resource:
            # 8bit 加载（需要 GPU + bitsandbytes）
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                load_in_8bit=True,
                device_map="auto",
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype= torch.float32,
            )

        # 自定义多分支 LoRA（文本端）
        if args.use_mblora:
            apply_multi_branch_lora(
                self.llama_model.model,  # 注意：LlamaForCausalLM.model 是 decoder 栈
                target_keywords=args.target_keywords,
                r=args.llm_r,
                alpha=args.llm_alpha,
                dropout=args.lora_dropout,
                num_branches=args.num_lora_branches,
            )
            # 激活默认分支与合成权重
            self._activate_branch(args.active_lora_branch)
        else:
            # 不使用 mblora 时：冻结 LLaMA 主干
            for _, p in self.llama_model.named_parameters():
                p.requires_grad = False

        self.embed_tokens = self.llama_model.get_input_embeddings()

        # 关键修正：SwinModel 的维度来自 config.hidden_size，而不是 .num_features
        self.llama_proj = nn.Linear(self.visual_encoder.config.hidden_size, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)

        # 让新建层直接采用与 LLaMA 嵌入一致的 dtype（通常 fp16），减少来回 cast
        proj_dtype = self.embed_tokens.weight.dtype
        self.llama_proj = self.llama_proj.to(proj_dtype)
        self.layer_norm = self.layer_norm.to(proj_dtype)

        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'

        # 兼容选项
        self.global_only = getattr(args, "global_only", False)

    # —— 保持你的 encode_img / prompt_wrap / forward 主逻辑 ——
    def encode_img(self, images):
        """
        images: List[Tensor]，每个 Tensor 形状 [B,3,H,W]
        """
        image_embeds = []
        for image in images:
            device = image.device
            if self.global_only:
                image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
            else:
                image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
            image_embeds.append(image_embed)

        # 如果有多张，取平均融合（按你原逻辑）
        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.llama_proj(image_embeds)
        # 关键修复：投影后对齐到 LLaMA 嵌入 dtype（通常 float16）
        inputs_llama = inputs_llama.to(self.embed_tokens.weight.dtype)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img):
        prompt = f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples):
        # 期望 samples["image"] 是 List[Tensor(B,3,H,W)]
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        # 关键修复：某些 LN 会回到 float32，这里再对齐一次
        if img_embeds.dtype != self.embed_tokens.weight.dtype:
            img_embeds = img_embeds.to(self.embed_tokens.weight.dtype)

        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length,
            add_special_tokens=False
        ).to(img_embeds.device)

        targets = to_regress_tokens.input_ids.masked_fill(to_regress_tokens.input_ids == 0, -100)

        empty_targets = torch.ones(
            [atts_img.shape[0], atts_img.shape[1] + 1], dtype=torch.long, device=img_embeds.device
        ).fill_(-100)  # plus one for bos
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1], dtype=to_regress_tokens.input_ids.dtype, device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    # —— 方便外部切换/融合两个 LoRA 分支 ——
    def _activate_branch(self, branch_idx: int):
        """
        - 阶段一：branch=0，merge=[1,0]
        - 阶段二：branch=1，merge=[1,1]（冻结0，只训1，但前向使用 0+1）
        """
        set_all_mb_lora_train_branch(self.llama_model, branch=branch_idx)
        if branch_idx == 0:
            set_all_mb_lora_merge_weights(self.llama_model, [1.0, 0.0])
        elif branch_idx == 1:
            set_all_mb_lora_merge_weights(self.llama_model, [1.0, 1.0])
        else:
            # 推理时可调用：branch=None，merge权重你自定
            set_all_mb_lora_train_branch(self.llama_model, None)


# ==============================================================
# 3) 合成数据（更加“靠谱”的合成分布）
# ==============================================================

class SynthMultiModalDataset(Dataset):
    """
    生成：
      - "image": [tensor(B,3,H,W)] 的列表（与你的 forward/encode_img 兼容）
      - "input_text": List[str]，长度为 B
    阶段一：文本偏“精细语义”
    阶段二：在阶段一基础上追加“完整报告”语句
    """
    def __init__(self, tokenizer: LlamaTokenizer, num_samples: int, phase: int,
                 batch_size: int = 2, H: int = 224, W: int = 224, seed: int = 42):
        super().__init__()
        self.tok = tokenizer
        self.num_samples = num_samples
        self.phase = phase
        self.batch_size = batch_size
        self.H, self.W = H, W

        g = torch.Generator().manual_seed(seed)

        # 构造一些“部位/征象/程度/模式”的词表，用于拼接合成句子
        self.organs = ["left lung", "right lung", "heart", "diaphragm", "hilar region", "costophrenic angle"]
        self.findings = ["opacity", "nodule", "effusion", "atelectasis", "consolidation", "pneumothorax"]
        self.severity = ["mild", "moderate", "severe"]
        self.patterns = ["patchy", "diffuse", "focal", "interstitial", "lobar"]

        # 提前生成文本
        self.samples = []
        for i in range(num_samples):
            org = self.organs[i % len(self.organs)]
            fd  = self.findings[i % len(self.findings)]
            sev = self.severity[i % len(self.severity)]
            pat = self.patterns[i % len(self.patterns)]

            # 阶段一：精细语义（短句，偏关键词）
            fine = f"fine semantics: {sev} {pat} {fd} in the {org}."
            if phase == 1:
                text = fine
            else:
                # 阶段二：完整报告 = 精细语义 + 报告段落（增加一些报告体裁的常见短语）
                report = (
                    f"Final report: The image suggests {sev} {fd} with a {pat} pattern involving the {org}. "
                    f"No acute osseous abnormality. Correlate clinically and compare with prior imaging when available."
                )
                text = fine + " " + report

            # 组一个 batch 的文本
            texts = [text for _ in range(batch_size)]

            # 生成匹配的图像 batch（B,3,H,W），加入轻微结构噪声更贴近自然图像分布
            # 基础噪声
            img = torch.randn(batch_size, 3, H, W, generator=g) * 0.25
            # 模拟“亮度/对比度”变化
            img = img * (0.8 + 0.4 * torch.rand(batch_size, 1, 1, 1, generator=g)) + (torch.rand(batch_size, 1, 1, 1, generator=g) - 0.5) * 0.2
            # 加个简易“病灶”圆形区域（随机通道&位置）
            for b in range(batch_size):
                c = int(torch.randint(0, 3, (1,), generator=g))
                cy = int(torch.randint(H//4, 3*H//4, (1,), generator=g))
                cx = int(torch.randint(W//4, 3*W//4, (1,), generator=g))
                rad = int(torch.randint(min(H, W)//16, min(H, W)//8, (1,), generator=g))
                yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
                mask = ((yy - cy)**2 + (xx - cx)**2) <= (rad**2)
                img[b, c][mask] += 0.35 if "opacity" in fd or "consolidation" in fd else -0.35
            # 归一到[-1,1]大致范围
            img = img.clamp(-1.0, 1.0)

            self.samples.append({"image": [img], "input_text": texts})

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]


def make_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    """
    每个样本内部已经自带一个“B维度”的小 batch（image: [Tensor(B,3,H,W)]，texts: List[str] 长度=B），
    这里 DataLoader 用 batch_size=1，并用自定义 collate_fn 直接返回该样本，避免多重嵌套。
    """
    def _collate(one_item_batch):
        assert len(one_item_batch) == 1
        return one_item_batch[0]
    return DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_collate)


# ==============================================================
# 4) 训练循环（两阶段）
# ==============================================================

def train_stage(model: MMReportModel, loader: DataLoader, steps: int, lr: float = 2e-4, wd: float = 0.0, device="cuda"):
    model.to(device)
    model.train()

    # 只优化 requires_grad=True 的参数（即当前激活的 LoRA 分支权重，和未冻结的视觉侧参数）
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    step = 0
    while step < steps:
        for batch in loader:
            # 将图像移到 device；保持为 List[Tensor(B,3,H,W)]
            batch["image"] = [t.to(device) for t in batch["image"]]
            out = model(batch)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)

            step += 1
            if step % 5 == 0:
                print(f"step {step} | loss {loss.item():.4f}")
            if step >= steps:
                break


# ==============================================================
# 5) 主函数：组装参数 → 构建模型 → 合成数据 → 两阶段训练/测试
# ==============================================================

@dataclass
class Args:
    # 视觉编码器
    vision_model: str = "microsoft/swin-tiny-patch4-window7-224"
    vis_use_lora: bool = False
    vis_r: int = 8
    vis_alpha: int = 16
    freeze_vm: bool = True
    lora_dropout: float = 0.05

    # LLaMA 模型
    llama_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 先用小模型做连通性测试
    low_resource: bool = False  # 若换 7B，可设 True（8bit）

    # 多分支 LoRA 设置（文本端）
    use_mblora: bool = True
    num_lora_branches: int = 2
    active_lora_branch: int = 0
    llm_r: int = 8
    llm_alpha: int = 16
    target_keywords: tuple = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")

    # 文本拼接
    end_sym: str = "</s>"
    max_length: int = 256

    # 其他
    delta_file: Optional[str] = None
    global_only: bool = False


def main():
    args = Args()

    # 构建模型
    model = MMReportModel(args)

    # ========== 阶段一：训练 LoRA0，合并权重 [1,0] ==========
    print("\n[Stage-1] Train LoRA0 only; forward uses LoRA0 (w=[1,0])")
    model._activate_branch(0)  # 打开分支0训练 + 设合成权重 [1,0]

    tok = model.llama_tokenizer
    ds1 = SynthMultiModalDataset(tok, num_samples=20, phase=1, batch_size=2, H=224, W=224)
    loader1 = make_loader(ds1, batch_size=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_stage(model, loader1, steps=20, lr=2e-4, wd=0.0, device=device)

  
    print("\n[Stage-2] Freeze LoRA0, train LoRA1; forward uses LoRA0+LoRA1 (w=[1,1])")
    model._activate_branch(1)  

    ds2 = SynthMultiModalDataset(tok, num_samples=20, phase=2, batch_size=2, H=224, W=224)
    loader2 = make_loader(ds2, batch_size=1)
    train_stage(model, loader2, steps=20, lr=2e-4, wd=0.0, device=device)

    
    print("\n[Inference] Freeze all branches; use merge weights [1.0, 0.8]")
    set_all_mb_lora_train_branch(model.llama_model, None)
    set_all_mb_lora_merge_weights(model.llama_model, [1.0, 0.8])

    # 构造一个小 batch 做前传，确认不报错
    batch = {
        "image": [torch.randn(2, 3, 224, 224).to(device)],
        "input_text": [
            "fine semantics: mild focal opacity in the left lung. Final report: The image suggests mild opacity with a focal pattern involving the left lung. Correlate clinically.",
            "fine semantics: severe diffuse effusion in the right lung. Final report: The image suggests severe effusion with a diffuse pattern involving the right lung. Correlate clinically.",
        ],
    }
    model = model.to(device)
    out = model(batch)
    print(f"Smoke test loss: {out['loss'].item():.4f}")


if __name__ == "__main__":
    main()
