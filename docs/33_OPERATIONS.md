# 33 · 运维指南

> 滚动训练的日常操作：CLI 参数、状态管理、工作流、故障排查。

---

## 1. CLI 参数速查

### 运行模式

| 参数 | 作用 | 适用场景 |
|------|------|---------|
| `--cold-start` | 清空当前模式 state，全量训练所有窗口 | 首次使用、彻底重建 |
| `--merge` | 保留 state，仅训练缺失窗口 | qlib 数据更新后补训、追加新模型 |
| `--retrain-models M1,M2` | 清空指定模型 state，全量重建 | 模型超参/代码变更 |
| `--retrain-last` | 清空最后窗口 state，重训 | 数据修正 |
| `--predict-only` | 不训练，用最新窗口模型预测 | 数据更新后快速出预测 |
| `--resume` | 从断点继续 | 训练中断恢复 |
| `--backtest` | 训练完成后执行回测 | 附加于其他模式 |
| `--backtest-only` | 仅回测，跳过训练/预测 | 对已有预测出回测报告 |

### 模型选择

| 参数 | 说明 |
|------|------|
| `--models m1,m2` | 按名称指定 |
| `--algorithm alg` | 按算法筛选 |
| `--dataset ds` | 按数据集筛选 |
| `--tag tag` | 按标签筛选 |
| `--all-enabled` | 所有 `enabled: true` 的模型 |
| `--skip m1,m2` | 排除指定模型 |

### 运行控制

| 参数 | 说明 |
|------|------|
| `--workspace PATH` | 显式选择 workspace；未指定时读取 `QLIB_WORKSPACE_DIR` |
| `--dry-run` | authoritative Prepared Plan 的兼容入口；精确窗口日期延迟到真实执行 |
| `--explain-plan` | 同一个 Prepared Plan 的 human-readable 形式 |
| `--json-plan` | 同一个 Prepared Plan 的单一 JSON 文档（含输入/state/effect/plan fingerprints） |
| `--run-id ID` | 真实执行的显式 lease operation identity |
| `--show-folds` | CPCV 模式下显示每窗口的 Fold 详情（train/valid 日期段） |
| `--training-method slide\|cpcv` | 覆盖 `rolling_config.yaml` 中的设置，无需改配置文件 |
| `--cache-size N` | Handler 缓存上限（MB），0=禁用，默认=自动。CPCV 模式 K-fold 共享 |
| `--no-pretrain` | 跳过预训练模型加载 |
| `--allow-stale-predict` | 允许 `--predict-only` 用旧权重预测新数据 |

Prepared Plan 是严格的文件系统只读操作，不初始化 Qlib/MLflow，不获取 shared lease，也不写
OperatorLog。它会冻结 registry 顺序的 targets 和 workspace-contained workflow，分类 legacy state，并
如实声明真实命令可能写入的 state、record、history、MLflow artifacts 和 OperatorLog。日历 anchor、
windows/CPCV folds 在 Prepared Plan 中保持 deferred；真实执行会在 shared lease 内复核 input baselines，
初始化一次 Qlib，并用 Resolved Plan 绑定精确 anchor、ordered windows/folds、stable window keys 和 execution
fingerprint。`--workspace PATH`、`--workspace=PATH` 与 `main(argv=[...])` 共享这个 Prepared context；
safeguard 不导入 legacy `env`，activation 只在 safeguard → lease → baseline recheck 后发生。legacy adapter
只消费该冻结范围，不重新扫描 registry 或再次生成窗口。

缺少 anchor 的 `daily`/`predict-only` 和没有已完成窗口的 `retrain-last` 会以
`rolling_state_precondition_failed` 在 backend 前失败。已缺失 state 的 `--clear-state` 以及没有生成预测的
`--predict-only` 明确记录为 `skipped`。OperatorLog、adapter outcome 与 CLI exit 使用同一个
command-level status；成功 action 仍保留 `legacy_partial_visibility`，不代表 per-window evidence parity。
`--backtest-only` 若缺少有效 current records、请求的 Rolling family 或选定 target 的历史 record，会以
`rolling_backtest_precondition_failed` 返回非零，并在 OperatorLog 中记录同一 failure；只有已找到 records
并调用 legacy backtest 的路径才记录 `success / legacy_partial_visibility`。
shared lease 内、backend 初始化前还会对所有声明写路径做 symlink-aware containment 检查。任何实际解析到
workspace 外的 state/record/history/MLflow/OperatorLog 路径都会以
`rolling_output_outside_workspace` 拒绝；不要通过跨 workspace symlink 共享可写运行状态。

全量回归与 workspace gate 由项目 owner 执行。无写验证只使用 `Demo_Workspace` 或 owner 明确选择的
一次性 validation workspace；生产 workspace 始终只读。真实 Rolling adapter/bootstrap smoke 必须单独
授权并只在可丢弃 validation workspace 执行；提交的代码和文档不得包含私有 workspace identity、绝对路径
或运行数据。

### 信息查看

| 参数 | 说明 |
|------|------|
| `--show-state` | 严格只读分类：`missing` / `valid_legacy` / `corrupt` / `unsupported` |
| `--clear-state` | 清除当前模式的训练状态（自动备份） |

> `--show-state` / `--clear-state` 支持 `--training-method` 选择目标模式。

---

## 2. 状态管理

### 状态文件

| 训练方法 | 状态文件 |
|---------|---------|
| slide | `data/rolling_state.json` |
| CPCV | `data/rolling_state_cpcv.json` |

两个文件完全独立。`--cold-start` 只清空当前模式的状态，不影响另一模式。

### 状态结构

```json
{
    "started_at": "2024-12-28 10:00:00",
    "rolling_config": {"test_step": "3M", "training_method": "cpcv", ...},
    "anchor_date": "2024-12-26",
    "training_method": "cpcv",
    "total_windows": 26,
    "completed_windows": {
        "0": {"linear_Alpha158": "abc123..."},
        "1": {"linear_Alpha158": "def456..."}
    }
}
```

### 跨模式隔离

| 操作 | slide 状态 | CPCV 状态 | `latest_train_records.json` |
|------|-----------|----------|---------------------------|
| `--cold-start --training-method slide` | **清空** | 不变 | slide key 覆盖，CPCV key 保留 |
| `--cold-start --training-method cpcv` | 不变 | **清空** | CPCV key 覆盖，slide key 保留 |
| `--clear-state --training-method slide` | **删除** | 不变 | 不变 |
| `--clear-state --training-method cpcv` | 不变 | **删除** | 不变 |

记录文件使用 `merge_train_records()` —— 不同 `@mode` 后缀的 key 永不冲突。

---

## 3. 日常工作流

### 首次冷启动

```bash
# Slide 模式
python quantpits/scripts/rolling_train.py --cold-start --all-enabled \
  --training-method slide

# CPCV 模式（建议先用 fast model 验证）
python quantpits/scripts/rolling_train.py --cold-start \
  --models linear_Alpha158 --training-method cpcv
```

### 每周数据更新后

```bash
# 1. 补训新窗口
python quantpits/scripts/rolling_train.py --merge --all-enabled

# 2. 仅预测（不重训）
python quantpits/scripts/rolling_train.py --predict-only --all-enabled

# 3. 回测
python quantpits/scripts/rolling_train.py --backtest-only --all-enabled
```

### 调参后重建

```bash
python quantpits/scripts/rolling_train.py --retrain-models alstm_Alpha158
```

### 对比两种模式

```bash
# Slide 滚动
python quantpits/scripts/rolling_train.py --cold-start --models linear_Alpha158 \
  --training-method slide
python quantpits/scripts/rolling_train.py --backtest-only --models linear_Alpha158 \
  --training-method slide

# CPCV 滚动
python quantpits/scripts/rolling_train.py --cold-start --models linear_Alpha158 \
  --training-method cpcv
python quantpits/scripts/rolling_train.py --backtest-only --models linear_Alpha158 \
  --training-method cpcv

# 两者的 key 不同（@rolling vs @cpcv_rolling），可共存于 latest_train_records.json
```

---

## 4. `--predict-only` 行为

| 场景 | 行为 |
|------|------|
| 无 gap（所有窗口已训练至 anchor_date） | 跳过，打印 "Nothing to predict" |
| 有 gap + `--allow-stale-predict` | 仅预测 gap 段 `[test_end+1d, anchor_date]` |
| 有 gap 无 flag | 跳过，提示 `--allow-stale-predict` 或 `--retrain-models` |

> Gap 检测同时看窗口索引差和日期差。即使索引相同，`anchor_date > test_end` 也会触发 gap 预测。

---

## 5. `--backtest-only` 输出

回测输出与训练时的 `PortAnaRecord` 完全一致：

```
The following are analysis results of benchmark return(1week).
                       risk
mean               0.002227
std                0.016799
annualized_return  0.115800
information_ratio  0.955909
max_drawdown      -0.066849

The following are analysis results of excess return without cost(1week).
...

The following are analysis results of indicators(week).
     value
ffr    1.0
pa     0.0
pos    0.0

[model] 回测完成! Ann_Ret: -17.58%, Excess: -17.58%, Max_DD: -15.93%, IR: -1.075
```

---

## 6. 故障排查

| 症状 | 原因 | 解决 |
|------|------|------|
| `purge_steps >= smallest CV group size` | 窗口时间范围太短 | 增大 `train_years` 或减小 `n_groups` |
| `Embargo pushes test_start past test_end` | embargo 太大 | 减小 `embargo_steps` |
| `NameError: RollingState is not defined` | 已修复 | — |
| `--predict-only` 加载了错误的模型文件 | 之前 bug（slide 策略被误用于 CPCV） | 已修复 |
| handler 缓存未生效 | `--cache-size 0` 禁用了缓存 | 去掉 `--cache-size 0` 或设正值 |
| CPCV predict-only 8 次重复加载数据 | 无 gap 时不应执行 predict | 已修复：gap=0 时跳过 |
