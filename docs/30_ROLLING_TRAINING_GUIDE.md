# Rolling Training Guide（滚动训练 · 总览）

> Phase 27 过渡边界：Rolling/CPCV-rolling 仍使用既有 window/state 编排，尚未迁入 static/CPCV
> execution service；但所有真实变更命令现在与 static/CPCV 共用 workspace training execution
> lease，避免并发覆盖 current record。`--show-state` 与 `--dry-run` 不获取该 lease。完整的
> plan/evidence/publication/closure 对齐仍属明确延后范围。
> `--dry-run` 是严格的文件系统预览：只显示配置、动作与模型选择，不初始化 Qlib、不计算实际
> window/fold 日期，也不创建 OperatorLog、state、manifest 或 lock；精确窗口在真实执行中解析。
> `--show-state` 同样使用无 lock 的只读 state load；`--clear-state` 会备份/删除 state，因此仍经过
> safeguard 并持有 shared execution lease。

> 30 系列文档专注于**滚动训练**——训练窗口随时间推进而滑动的训练范式。
>
> | 文档 | 内容 |
> |------|------|
> | **30（本文）** | 架构总览、四种模式、Key 系统、快速上手 |
> | [31  · Slide 模式](31_SLIDE_ROLLING.md) | 窗口数学、示例、适用场景 |
> | [32  · CPCV 模式](32_CPCV_ROLLING.md) | Walk-Forward CPCV、Fold 结构、Gap 分析、参数约束 |
> | [33  · 运维指南](33_OPERATIONS.md) | CLI 参数、状态管理、日常流程、故障排查 |

---

## 1. 四种训练模式共存

| # | 模式 | 脚本 | 训练方法 | 输出 Key |
|---|------|------|---------|---------|
| 1 | 静态 | `static_train.py --full` | 固定日期区间 slide | `model@static` |
| 2 | CPCV 单次 | `cv_train.py --all-enabled` | K-fold + purge/embargo | `model@cpcv` |
| 3 | 滚动 + slide | `rolling_train.py` | 每窗口 slide | `model@rolling` |
| 4 | 滚动 + CPCV | `rolling_train.py` | 每窗口 Walk-Forward CPCV | `model@cpcv_rolling` |

四种模式可共存于同一 workspace，互不影响。下游通过 `--training-mode` 选择对应模型。

### Train → Today Gap 对比

| 模式 | Gap | 瓶颈 |
|------|-----|------|
| 静态 slide（5yr train / 2yr valid / 2yr test） | ~4 年 | valid + test 固定区间 |
| CPCV 单次（10 groups / 2 test） | ~2 年 | test_set = 2 groups |
| 滚动 + slide（5yr train / 1yr valid / 3M step） | ~13 月 | valid = 1 年 |
| **滚动 + CPCV（5yr train / 3M step）** | **~4 月** | test_step + purge |

---

## 2. 架构：WHEN vs HOW 解耦

```
rolling_train.py      ← CLI + 流程编排 + 策略分派
    │
    ├─ training_method: "slide" | "cpcv"  (--training-method 可覆盖)
    │
    ├─ strategy_slide.py     ← 每窗口 train/valid/test 连续三段
    ├─ strategy_cpcv.py      ← 每窗口 K-fold Walk-Forward CPCV
    │
    └─ orchestration.py      ← 共享编排：窗口循环 + 拼接 + 保存 + 回测
```

```
rolling/
├── state.py              # 状态管理（slide 和 CPCV 各自独立文件）
├── memory.py             # 3 层内存清理
├── backtest.py           # 回测
├── orchestration.py      # 共享编排逻辑
├── strategy_slide.py     # Slide 策略
├── strategy_cpcv.py      # CPCV 策略
├── windows.py            # [兼容层]
├── training.py           # [兼容层]
└── prediction.py         # [兼容层]
```

---

## 3. 配置文件

`config/rolling_config.yaml`：

```yaml
rolling_start: "2015-01-01"     # T: 第一个窗口起点
train_years: 5                  # slide: 训练段 / CPCV: Train 域长度
valid_years: 1                  # slide: 验证段 / CPCV: 不使用（保留兼容）
test_step: "3M"                 # 滚动步长（nM 或 nY）

training_method: "cpcv"         # "slide" | "cpcv"

# CPCV 参数（仅 training_method=cpcv 时生效）
cpcv_n_groups: 10               # Train 域内分组数
cpcv_n_val_groups: 1            # 每 Fold 验证组数
cpcv_purge_steps: 3             # 对称 Purge（交易周）
cpcv_embargo_steps: 5           # 非对称 Embargo（交易周）
```

---

## 4. 快速上手

```bash
conda activate qlib_cupy
source workspaces/<name>/run_env.sh

# ---- Slide 滚动 ----
python quantpits/scripts/rolling_train.py --cold-start --dry-run \
  --models linear_Alpha158 --training-method slide

# ---- CPCV 滚动 ----
python quantpits/scripts/rolling_train.py --cold-start --dry-run \
  --models linear_Alpha158 --training-method cpcv --show-folds

# ---- 真训 ----
python quantpits/scripts/rolling_train.py --cold-start \
  --models linear_Alpha158 --training-method cpcv

# ---- 数据更新后补训 ----
python quantpits/scripts/rolling_train.py --merge --all-enabled

# ---- 仅预测（数据更新后不重训）----
python quantpits/scripts/rolling_train.py --predict-only --all-enabled

# ---- 回测 ----
python quantpits/scripts/rolling_train.py --backtest-only \
  --models linear_Alpha158 --training-method cpcv
```

Phase 27 的 `--dry-run` 不再展示依赖 Qlib calendar 的精确窗口或 fold；`--show-folds` 仅在真实
执行完成日期解析后生效。该取舍保证 dry-run 严格零写入、零 Qlib 初始化。

> `--training-method` 可覆盖 `rolling_config.yaml` 中的设置，无需改配置文件即可切换模式对比效果。

---

## 5. Key 系统

| 模式 | Key | `--training-mode` |
|------|-----|-------------------|
| 静态 | `model@static` | `static` |
| CPCV 单次 | `model@cpcv` | `cpcv` |
| 滚动 + slide | `model@rolling` | `rolling` |
| 滚动 + CPCV | `model@cpcv_rolling` | `cpcv_rolling` |

不同模式的 Key 共存于 `latest_train_records.json`。`--cold-start` 只清空当前模式的 state 文件，不影响其他模式的记录。

V2 记录为 rolling 与 CPCV rolling 分别保留模型级输出 experiment/recorder 身份；
`cpcv_rolling_experiment_name` 兼容字段在后续无关 merge 中也必须保留。
真实 rolling 命令会验证 combined recorder、持久化预测覆盖和 artifact workspace containment；
验证失败时不会降级发布未验证的 current pointer。

```bash
# 下游使用
python quantpits/scripts/brute_force_fast.py --training-mode rolling
python quantpits/scripts/brute_force_fast.py --training-mode cpcv_rolling
python quantpits/scripts/ensemble_fusion.py --from-config --training-mode rolling
```

---

## 6. 进一步阅读

- [31  · Slide 模式详解](31_SLIDE_ROLLING.md) — 窗口数学公式、完整示例
- [32  · CPCV 模式详解](32_CPCV_ROLLING.md) — Walk-Forward CPCV 设计原理、Fold 结构、参数约束
- [33  · 运维指南](33_OPERATIONS.md) — 全部 CLI 参数、状态管理、日常流程
