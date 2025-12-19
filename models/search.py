import copy
import logging
import os
import random
from datetime import datetime

from test import test
from train import train

log_dir = "../log"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/nas_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

# ------------ helpers: 采样/归一化/交叉/变异 ------------
def _nearest_allowed_depth(d, allowed_depths):
    # 把任意整数 depth 对齐到允许集合里最接近的一个
    return min(allowed_depths, key=lambda x: abs(x - d))


def normalize_arch(arch, search_space):
    """
    归一化/修复一个 arch，确保：
      - 有效的 depth（在 search_space['depth'] 允许集合中）
      - len(ops) == len(hidden_dims) == depth
      - skip/op/hidden 都落在候选集合内
    """
    allowed_depths = sorted(list(search_space["depth"]))
    op_cands = list(search_space["op_candidates"])
    hid_cands = list(search_space["hidden_candidates"])
    skip_cands = list(search_space["skip_candidates"])

    ops = list(arch.get("ops", []))
    hids = list(arch.get("hidden_dims", []))
    skip = arch.get("skip", random.choice(skip_cands))
    # 初步 depth：优先 arch 指定；否则来自已有长度；再不行随机
    d = arch.get("depth", None)
    if d is None:
        if len(ops) > 0:
            d = len(ops)
        elif len(hids) > 0:
            d = len(hids)
        else:
            d = random.choice(allowed_depths)

    # 把 d 裁剪/贴近到允许集合
    d = _nearest_allowed_depth(max(1, d), allowed_depths)

    # 先把非法 op 替换掉
    ops = [o if o in op_cands else random.choice(op_cands) for o in ops]
    # 扩/截到 d
    if len(ops) < d:
        ops += [random.choice(op_cands) for _ in range(d - len(ops))]
    ops = ops[:d]

    # hidden 同理
    if len(hids) < d:
        hids += [random.choice(hid_cands) for _ in range(d - len(hids))]
    hids = hids[:d]

    if skip not in skip_cands:
        skip = random.choice(skip_cands)

    fixed = {
        "depth": d,
        "ops": ops,
        "hidden_dims": hids,
        "skip": skip
    }

    # 打个 debug
    print(f"    [normalize_arch] depth={d}, len(ops)={len(ops)}, len(hiddens)={len(hids)}, skip={skip}")
    logging.info(f"    [normalize_arch] depth={d}, len(ops)={len(ops)}, len(hiddens)={len(hids)}, skip={skip}")
    return fixed


def crossover(parent1, parent2, search_space):
    """
    单点交叉（对齐短父母），然后用 normalize_arch 修复长度/合法性。
    """
    l1, l2 = len(parent1["ops"]), len(parent2["ops"])
    # 如果任一父母深度为 1，直接拼接+修复，避免 empty range
    if min(l1, l2) <= 1:
        child = {
            "ops": (parent1["ops"] + parent2["ops"])[:max(l1, l2)],
            "hidden_dims": (parent1["hidden_dims"] + parent2["hidden_dims"])[:max(l1, l2)],
            "skip": random.choice([parent1.get("skip", "none"), parent2.get("skip", "none")]),
            "depth": max(l1, l2),
        }
        print(f"    [crossover] short parent -> raw child: {child}")
        logging.info(f"    [crossover] short parent -> raw child: {child}")
        return normalize_arch(child, search_space)

    cut = random.randint(1, min(l1, l2) - 1)
    # 目标深度从两个父母里挑一个
    target_d = random.choice([l1, l2])

    ops = parent1["ops"][:cut] + parent2["ops"][:]
    hids = parent1["hidden_dims"][:cut] + parent2["hidden_dims"][:]
    child = {
        "ops": ops[:target_d],
        "hidden_dims": hids[:target_d],
        "skip": random.choice([parent1.get("skip", "none"), parent2.get("skip", "none")]),
        "depth": target_d,
    }
    print(f"    [crossover] cut={cut}, target_d={target_d}, raw child: {child}")
    logging.info(f"    [crossover] cut={cut}, target_d={target_d}, raw child: {child}")
    return normalize_arch(child, search_space)

def arch_to_key(arch):
    """
    把一个架构 arch 转成可哈希的 key，方便用来判重
    """
    return (
        int(arch["depth"]),
        tuple(arch["ops"]),
        tuple(arch["hidden_dims"]),
        arch.get("skip", "none"),
    )


def mutate(arch, search_space, p_change_depth=0.3):
    """
    细粒度变异：随机改 op/hidden/skip；一定概率增删层（并修复）。
    """
    arch = copy.deepcopy(arch)
    allowed_depths = sorted(list(search_space["depth"]))
    op_cands = list(search_space["op_candidates"])
    hid_cands = list(search_space["hidden_candidates"])
    skip_cands = list(search_space["skip_candidates"])

    d = arch["depth"]
    if random.random() < p_change_depth:
        # 改深度（加/减一层）
        if random.random() < 0.5 and d < max(allowed_depths):
            insert_pos = random.randrange(d + 1)
            arch["ops"].insert(random.randrange(d + 1), random.choice(op_cands))
            arch["hidden_dims"].insert(random.randrange(d + 1), random.choice(hid_cands))
            arch["depth"] = d + 1
            print(f"    [mutate] +1 layer at {insert_pos}")
            logging.info(f"    [mutate] +1 layer at {insert_pos}")
        elif d > min(allowed_depths):
            remove_pos = random.randrange(d)
            idx = random.randrange(d)
            del arch["ops"][idx]
            del arch["hidden_dims"][idx]
            arch["depth"] = d - 1
            print(f"    [mutate] -1 layer at {remove_pos}")
            logging.info(f"    [mutate] -1 layer at {remove_pos}")
    else:
        # 不改深度，改某一层的 op / hidden
        idx = random.randrange(d)
        if random.random() < 0.5:
            old = arch["ops"][idx]
            arch["ops"][idx] = random.choice(op_cands)
            print(f"    [mutate] change op at {idx}: {old} -> {arch['ops'][idx]}")
            logging.info(f"    [mutate] change op at {idx}: {old} -> {arch['ops'][idx]}")
        else:
            old = arch["hidden_dims"][idx]
            arch["hidden_dims"][idx] = random.choice(hid_cands)
            print(f"    [mutate] change hidden at {idx}: {old} -> {arch['hidden_dims'][idx]}")
            logging.info(f"    [mutate] change hidden at {idx}: {old} -> {arch['hidden_dims'][idx]}")

    # 少量概率换 skip
    if random.random() < 0.2:
        old_skip = arch.get("skip", "none")
        arch["skip"] = random.choice(skip_cands)
        print(f"    [mutate] change skip: {old_skip} -> {arch['skip']}")
        logging.info(f"    [mutate] change skip: {old_skip} -> {arch['skip']}")

    return normalize_arch(arch, search_space)


# 从搜索空间随机采样一个架构
def sample_arch(search_space):
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

def tournament_select(pop_with_fitness, k=3):
    """
    锦标赛选择:
      pop_with_fitness: [(arch, fitness), ...]，fitness 越小越好（你这里=1-AUC）
      k: 每次参与锦标赛的个体数
    返回: 被选中的 arch
    """
    # 如果种群太小，直接退化成随机选
    if len(pop_with_fitness) <= k:
        arch, _ = min(pop_with_fitness, key=lambda x: x[1])
        return arch

    competitors = random.sample(pop_with_fitness, k)
    # fitness 越小越好
    competitors.sort(key=lambda x: x[1])
    return competitors[0][0]

# 遗传算法搜索策略
def genetic_algorithm(
        make_model_fn,
        train_dataloader,
        val_dataloader,
        base_train_config: dict,
        num_generations: int = 15,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        device: str = "cuda",
        tmp_save_dir: str = "./pretrained/nas_tmp/"
):

    os.makedirs(tmp_save_dir, exist_ok=True)

    # 先造一个模型拿 search_space，就不每次造了
    probe_model = make_model_fn().to(device)
    search_space = probe_model.get_search_space()
    del probe_model

    # 初始化种群（采样 + 归一化）
    population = [normalize_arch(sample_arch(search_space), search_space)
                  for _ in range(population_size)]

    best_val_loss = float("inf")
    best_arch = None

    # 遗传算法的进化过程
    for generation in range(num_generations):
        print(f"\n========== Generation {generation + 1} ==========")
        logging.info(f"\n========== Generation {generation + 1} ==========")
        print("Population architectures (before eval):")
        logging.info("Population architectures (before eval):")
        for idx, arch in enumerate(population):
            print(f"  [{idx + 1}] depth={arch['depth']} | "
                  f"ops={arch['ops']} | "
                  f"hidden={arch['hidden_dims']} | "
                  f"skip={arch['skip']}")
            logging.info(f"  [{idx + 1}] depth={arch['depth']} | "
                         f"ops={arch['ops']} | "
                         f"hidden={arch['hidden_dims']} | "
                         f"skip={arch['skip']}")

        # 评估种群的适应度（这里适应度=1-AUC，越小越好）
        val_losses = []  # 实际装的是 (arch, fitness)，只是变量名沿用
        for idx, arch in enumerate(population):
            print(f"\nEvaluating architecture {idx + 1}/{len(population)} ...")
            logging.info(f"\nEvaluating architecture {idx + 1}/{len(population)} ...")

            # 防御性再 normalize 一次，确保是合法结构
            arch = normalize_arch(arch, search_space)

            print(f"  -> Using arch: depth={arch['depth']}, "
                  f"len(ops)={len(arch['ops'])}, len(hiddens)={len(arch['hidden_dims'])}, "
                  f"skip={arch['skip']}")
            print(f"  -> ops: {arch['ops']}")
            print(f"  -> hidden_dims: {arch['hidden_dims']}")

            logging.info(f"  -> Using arch: depth={arch['depth']}, "
                         f"len(ops)={len(arch['ops'])}, len(hiddens)={len(arch['hidden_dims'])}, "
                         f"skip={arch['skip']}")
            logging.info(f"  -> ops: {arch['ops']}")
            logging.info(f"  -> hidden_dims: {arch['hidden_dims']}")

            # 统一取可搜索的 GNN 层：支持 searchable_gnn_layers 或 searchable_gnn
            model = make_model_fn().to(device)
            if getattr(model, "searchable_gnn_layers", None) is not None:
                nas_layers = list(model.searchable_gnn_layers)
            elif getattr(model, "searchable_gnn", None) is not None:
                nas_layers = [model.searchable_gnn]
            else:
                nas_layers = []

            # 对所有 NAS 层应用同一个 arch（多分支共享架构）
            for l_id, layer in enumerate(nas_layers):
                print(f"    -> build_arch for NAS layer {l_id}")
                logging.info(f"    -> build_arch for NAS layer {l_id}")
                layer.build_arch(arch)

            cfg = copy.deepcopy(base_train_config)
            cfg["epoch"] = min(10, cfg.get("epoch", 10))  # 短训
            tmp_path = os.path.join(tmp_save_dir, f"nas_gen{generation}_idx{idx}.pt")

            train_loss_list, val_metric_list,val_loss=train(model=model, save_path=tmp_path, config=cfg, train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader)
            print("val_loss:",val_loss)
            # 使用带标签的数据（search_val_dataloader）做异常检测评估
            # 1) 先在 search_val_dataloader 上跑一次 test，拿到预测/真值/标签
            val_losses.append((arch, val_loss))
            print(f"  Architecture {idx + 1} validation loss: {val_loss:.6f}")
            logging.info(f"  Architecture {idx + 1} validation loss: {val_loss:.6f}")

            print(f"  -> Validation losses: {val_losses}")

        # 如果这一代一个都没成功评估，直接 break
        if len(val_losses) == 0:
            print("[ERROR] No architectures evaluated in this generation.")
            logging.info("[ERROR] No architectures evaluated in this generation.")
            break

        # 按验证集损失排序，选择最好的架构
        val_losses.sort(key=lambda x: x[1])
        best_arch_in_pop, best_val = val_losses[0]

        print(f"\nGeneration {generation + 1} Results:")
        print(f"  Best validation loss in this gen: {best_val:.6f}")
        print(f"  Best architecture in this gen: {best_arch_in_pop}")

        logging.info(f"\nGeneration {generation + 1} Results:")
        logging.info(f"  Best validation loss in this gen: {best_val:.6f}")
        logging.info(f"  Best architecture in this gen: {best_arch_in_pop}")

        if best_val < best_val_loss:
            best_val_loss = best_val
            best_arch = best_arch_in_pop
            print(f"  >>> New global best found! val_loss={best_val_loss:.6f}")
            logging.info(f"  >>> New global best found! val_loss={best_val_loss:.6f}")

        # ===== 生成下一代种群（除了最后一代） =====
        if generation < num_generations - 1:
            # 生成下一代种群
            # 可以保留一个精英做直接继承
            next_generation = [best_arch_in_pop]  # 精英保留
            used_keys = {arch_to_key(best_arch_in_pop)}
            print(f"\nGenerating next generation...")
            logging.info("\nGenerating next generation...")

            # 锦标赛规模，通常取 2~5 之间
            tournament_k = min(3, len(val_losses))

            while len(next_generation) < population_size:
                # 使用锦标赛选择两个父母；val_losses: [(arch, fitness), ...]
                p1 = tournament_select(val_losses, k=tournament_k)
                p2 = tournament_select(val_losses, k=tournament_k)

                print(f"  [GA] pick parents (tournament k={tournament_k}):")
                logging.info(f"  [GA] pick parents (tournament k={tournament_k}):")

                print(f"    parent1: depth={p1['depth']}, skip={p1.get('skip', 'none')}")
                logging.info(f"    parent1: depth={p1['depth']}, skip={p1.get('skip', 'none')}")
                for i, (op, hid) in enumerate(zip(p1['ops'], p1['hidden_dims']), 1):
                    print(f"      L{i}: op={op:<10} hidden={hid}")
                    logging.info(f"      L{i}: op={op:<10} hidden={hid}")

                print(f"    parent2: depth={p2['depth']}, skip={p2.get('skip', 'none')}")
                logging.info(f"    parent2: depth={p2['depth']}, skip={p2.get('skip', 'none')}")
                for i, (op, hid) in enumerate(zip(p2['ops'], p2['hidden_dims']), 1):
                    print(f"      L{i}: op={op:<10} hidden={hid}")
                    logging.info(f"      L{i}: op={op:<10} hidden={hid}")

                # 交叉 + 变异
                child = crossover(p1, p2, search_space)

                if random.random() < mutation_rate:
                    print("    -> mutate child")
                    logging.info("    -> mutate child")
                    child = mutate(child, search_space)
                else:
                    print("    -> no mutation")
                    logging.info("    -> no mutation")

                child = normalize_arch(child, search_space)

                # ---------- 去重逻辑开始 ----------
                child_key = arch_to_key(child)
                retry = 0
                max_retry = 5

                # 如果重复,就多尝试几次变异
                while child_key in used_keys and retry < max_retry:
                    print("    -> duplicate child, re-mutate")
                    logging.info("    -> duplicate child, re-mutate")
                    child = mutate(child, search_space)
                    child = normalize_arch(child, search_space)
                    child_key = arch_to_key(child)
                    retry += 1

                # 如果还重复，就干脆随机采一个全新架构
                if child_key in used_keys:
                    print("    -> still duplicate after re-mutate, resample new arch")
                    logging.info("    -> still duplicate after re-mutate, resample new arch")
                    child = normalize_arch(sample_arch(search_space), search_space)
                    child_key = arch_to_key(child)
                # ---------- 去重逻辑结束 ----------

                print(f"    -> child after norm: depth={child['depth']}, "
                      f"len(ops)={len(child['ops'])}, len(hiddens)={len(child['hidden_dims'])}, "
                      f"skip={child['skip']}")
                logging.info(f"    -> child after norm: depth={child['depth']}, "
                             f"len(ops)={len(child['ops'])}, len(hiddens)={len(child['hidden_dims'])}, "
                             f"skip={child['skip']}")

                next_generation.append(child)

            population = next_generation
        else:
            # 最后一代：只评估，不再交叉 / 变异
            print("\nReached last generation, skip crossover/mutation.")
            logging.info("\nReached last generation, skip crossover/mutation.")
            # 不再更新 population，直接结束循环
            break


    print(f"\n========== Final Result ==========")
    print(f"Best Architecture Overall: {best_arch}")
    print(f"Best Validation Loss: {best_val_loss:.6f}")

    logging.info(f"\n========== Final Result ==========")
    logging.info(f"Best Architecture Overall: {best_arch}")
    logging.info(f"Best Validation Loss: {best_val_loss:.6f}")
    return best_arch


# 使用遗传算法进行架构搜索
def search_with_genetic_algorithm(
        make_model_fn,
        train_dataloader,
        val_dataloader,
        base_train_config: dict,
        num_generations: int = 15,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        device: str = "cuda",
        tmp_save_dir: str = "./pretrained/nas_tmp/"
):
    """
    使用遗传算法搜索最优架构
    """
    return genetic_algorithm(
        make_model_fn,
        train_dataloader,
        val_dataloader,
        base_train_config,
        num_generations=num_generations,
        population_size=population_size,
        mutation_rate=mutation_rate,
        device=device,
        tmp_save_dir=tmp_save_dir,
    )

