# models/search.py
import copy
import os
import random
import torch

from train import train     # 你的训练过程:contentReference[oaicite:0]{index=0}
from test import test       # 返回 (avg_loss, result):contentReference[oaicite:1]{index=1}

def sample_arch(search_space, max_depth=3):
    """从搜索空间随机采样一个架构"""
    depth = random.choice(search_space["depth"])
    ops = [random.choice(search_space["op_candidates"]) for _ in range(depth)]
    hiddens = [random.choice(search_space["hidden_candidates"]) for _ in range(depth)]
    skip = random.choice(search_space["skip_candidates"])
    return {
        "depth": depth,
        "ops": ops,
        "hidden_dims": hiddens,
        "skip": skip
    }

def reset_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def random_search(
        make_model_fn,              # () -> 一个新的、未训练的GDN实例 (use_nas=True)
        train_dataloader,
        val_dataloader,
        base_train_config: dict,    # 原训练配置，函数内部会拷贝并缩短epoch
        num_samples: int = 20,
        short_epochs: int = 10,
        device: str = "cuda",
        tmp_save_dir: str = "./pretrained/nas_tmp/"
):
    """
    返回最优的 arch_config (以验证集MSE最小为目标)
    """
    os.makedirs(tmp_save_dir, exist_ok=True)

    best_val = float("inf")
    best_arch = None

    for i in range(num_samples):
        model = make_model_fn().to(device)

        # 从可搜索层拿到搜索空间
        ss = model.get_search_space()   # 通过get_search_space获取search_space
        if ss is None:
            print(f"Model does not support NAS. Skipping {i+1}/{num_samples}")
            continue

        arch = sample_arch(ss)

        # 应用到模型
        if model.searchable_gnn:
            model.searchable_gnn.build_arch(arch)

        # 训练若干 epoch（短训）
        cfg = copy.deepcopy(base_train_config)
        cfg["epoch"] = short_epochs
        tmp_path = os.path.join(tmp_save_dir, f"nas_{i}.pt")



        train(
            model=model,
            save_path=tmp_path,
            config=cfg,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )

        # 用 val_dataloader 评估 MSE
        val_loss, _ = test(model, val_dataloader)   # 返回第一个是avg MSE:contentReference[oaicite:2]{index=2}

        print(f"[{i+1}/{num_samples}] arch={arch}  val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_arch = arch

    print(f"[NAS] Best arch: {best_arch}, val_loss={best_val:.6f}")
    return best_arch
