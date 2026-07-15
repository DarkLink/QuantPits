# Rolling Training Guide（滚动训练 · 总览）

> Phase 28A/28B/28C 已建立 import-pure CLI、filesystem-only authoritative Prepared Plan，以及
> lease 内的 Resolved Plan 与 legacy execution adapter。导入模块和
> 执行 `--help` 不要求预先 `source run_env.sh`，也不会改变当前目录。`--dry-run`、
> `--explain-plan` 与 `--json-plan` 来自同一个 typed plan，冻结 action、ordered targets、workflow
> 及 workspace 输入指纹、legacy state 分类、预期副作用和 plan fingerprint；该路径不初始化
> Qlib/MLflow，不获取 lease，也不写文件。registry 中的相对 workflow YAML 路径显式相对于所选
> workspace 解析，并且不允许逃逸 workspace。
>
> Prepared Plan 不伪造交易日历事实：anchor、slide windows 与 CPCV folds 会标为 runtime deferred。
> 真实命令直接用 Prepared `WorkspaceContext` 显示 safeguard，不导入 legacy `env`；随后在 shared lease
> 内复核 Prepared inputs，再激活同一个显式 workspace、初始化一次 Qlib，
> 冻结精确 anchor、ordered windows/CPCV folds、stable window keys 与 execution fingerprint。adapter 只消费
> Prepared targets 和 Resolved windows，不重新扫描 registry 或再次生成窗口。Rolling 仍使用 legacy
> unversioned state 与 legacy record merge，不具备新版 evidence/publication
> closure parity。

> Phase 27 过渡边界：Rolling/CPCV-rolling 仍使用既有 window/state 编排，尚未迁入 static/CPCV
> execution service；但所有真实变更命令现在与 static/CPCV 共用 workspace training execution
> lease，避免并发覆盖 current record。`--show-state` 与 `--dry-run` 不获取该 lease。完整的
> service-owned execution kernel、evidence/publication/closure 对齐仍属明确延后范围。
> `--dry-run` 是严格的文件系统 Prepared Plan：显示配置、输入指纹、动作、ordered targets、state
> 分类与预期副作用，但不初始化 Qlib、不计算实际 window/fold 日期，也不创建 OperatorLog、state、
> manifest 或 lock；精确窗口在真实执行中解析。
> `--show-state` 同样使用无 lock 的严格只读分类；`--clear-state` 会备份/删除 state，因此仍经过
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
quantpits/rolling/
├── command.py            # filesystem-only PreparedRollingRun
├── windows.py            # runtime ResolvedRollingRun + stable window keys
└── legacy.py             # exact-scope legacy execution adapter + baseline recheck

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

# 也可不 source，显式指定示例 workspace
python -m quantpits.scripts.rolling_train --workspace workspaces/Demo_Workspace --help

# 等价形式；direct script、python -m 与 main(argv=[...]) 使用同一个解析结果
python -m quantpits.scripts.rolling_train --workspace=workspaces/Demo_Workspace --help

# ---- Slide 滚动 ----
python quantpits/scripts/rolling_train.py --cold-start --dry-run \
  --models linear_Alpha158 --training-method slide

# 同一 authoritative Prepared Plan 的单文档 JSON 形式
python -m quantpits.scripts.rolling_train \
  --workspace workspaces/Demo_Workspace --cold-start --all-enabled --json-plan

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

Prepared Plan 不展示依赖 Qlib calendar 的精确窗口或 fold；`--show-folds` 在 plan 中只显示 deferred
原因，精确内容由真实执行在 lease 内解析并传给 legacy adapter。该取舍保证 plan route 严格零写入、零 Qlib/MLflow
初始化。多个 primary action（例如同时使用 `--cold-start --resume`）会在 safeguard、lease 和 backend
之前以 `rolling_action_conflict` 拒绝；`--show-state` 会区分 `missing`、`valid_legacy`、`corrupt` 与
`unsupported`，不会把损坏状态伪装成空状态。

`--workspace PATH`、`--workspace=PATH` 和程序化 `main(argv=[...])` 都以 Prepared context 作为唯一
workspace identity，不依赖进程 `sys.argv` 让 legacy `env` 再次选择 workspace。真实执行顺序为
explicit-context safeguard → shared lease → input baseline recheck → workspace activation → Qlib init →
Resolved Plan → legacy adapter → OperatorLog → lease release。`--clear-state` 不需要 Qlib，但仍在 lease 内
复核 state baseline 后备份并清除；state 已缺失时 outcome 与 OperatorLog 记录 `skipped`。`daily`、
`predict-only` 缺少有效 anchor，或 `retrain-last` 没有已完成窗口时，会在 backend 前以
`rolling_state_precondition_failed` 拒绝。OperatorLog、adapter outcome 与 CLI exit 共享同一个
command-level status。成功 action 仍标注 `legacy_partial_visibility`；它不代表每个 target×window 都已有
新版 immutable evidence 或 manifest/receipt closure。`predict-only` 没有生成 prediction 时记录 `skipped`。
`backtest-only` 在 current records 缺失/为空、缺少请求的 Rolling family，或选定 target 没有历史 Rolling
record 时，以 `rolling_backtest_precondition_failed` 失败并返回非零；只有找到 records 并实际调用 legacy
backtest 后，才会返回 `success / legacy_partial_visibility`。
真实执行还会在 Qlib/backend 初始化前解析所有声明写路径及其已有父目录；若 workspace 内的 symlink 令
state、record、history、MLflow 或 OperatorLog 写入实际落到 workspace 外，会以
`rolling_output_outside_workspace` 非零拒绝。Prepared Plan 会声明 current-record 的 history backup，避免
legacy backup 成为未声明副作用。

Phase 28 的全量 Python 测试与 workspace gate 由项目 owner 控制和执行。无写 gate 应使用
`Demo_Workspace` 或 owner 明确选择的一次性 validation workspace，并对配置、state、current records、
OperatorLog 与 MLflow 路径做前后快照；计划命令不得触发 safeguard、lease 或 backend 初始化。生产
workspace 保持只读。真实 adapter/bootstrap smoke 是 owner 验收项，只能在 owner 明确授权的可丢弃
validation workspace 执行；不得将私有 workspace identity、绝对路径或运行数据写入提交内容。

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
