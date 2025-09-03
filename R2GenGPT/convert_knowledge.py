import json, os, pathlib
from copy import deepcopy
def load_split(file_path: str, split_name: str):
    """读取单个 split JSON，保留所有原字段，只追加必要键。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    processed = []
    for entry in raw:
        entry = deepcopy(entry)           # 不修改原对象，防止副作用

        # 1️⃣ 统一 id —— 如果本来就有 id 字段则保留，否则尝试其他常见字段名
        if 'id' not in entry:
            entry['id'] = entry.get('study_id') or entry.get('uid')

        # 2️⃣ 统一 image_path —— 若已存在就直接用；否则从 views 提取
        if 'image_path' not in entry:
            img_paths = []
            for view in entry.get('views', []):
                # 视图可能是 dict，也可能直接是字符串；都做兼容
                img = view if isinstance(view, str) else view.get('image') or view.get('path')
                if img:
                    p_dir, patient, parent, fname = pathlib.Path(img).parts[-4:]
                    img_paths.append(f"{p_dir}/{patient}/{parent}/{fname}")
            entry['image_path'] = img_paths

        # 3️⃣ 标注 split
        entry['split'] = split_name

        processed.append(entry)
    return processed

# ---------- 修改这里的文件名 / 路径 ----------
train = load_split("/home/yuhaowang/project/report_generation/TRRG/MedCLIP/mimic_train.json", "train")
valid = load_split("/home/yuhaowang/project/report_generation/TRRG/MedCLIP/mimic_val.json", "valid")
test  = load_split("/home/yuhaowang/project/report_generation/TRRG/MedCLIP/mimic_test.json", "test")

merged = {"train": train, "val": valid, "test": test}

out_path = "./merged_mimic.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print("✅ 合并完成，数据量：", {k: len(v) for k, v in merged.items()})
print("📄 已保存到：", out_path)
