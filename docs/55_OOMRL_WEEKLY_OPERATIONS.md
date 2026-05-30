# OOM-RL 周度运维流程

完整的周度操作流程，涵盖从数据导入到订单生成的全部步骤，以及 RLFF 闭环反馈的介入点。

---

## 全流程概览

```
周六/周日（数据就绪后）

  1. 导入新数据                    qlib data update
         │
  2. 导入交易结算 + 跑 Post Trade   prod_post_trade.py
         │
  3. 跑 Deep Analysis              run_deep_analysis --critic
         │
  4. 评估 Action Items             阅读报告，判断哪些值得执行
         │
  5. Playground 验证               手动改 config + static_train
         │                          或 run_feedback_loop --execute
         │
  6. Promote 成功调整               promote_config.py 或手动改 production config
         │
  7. 训练/预测                     static_train.py
         │
  8. 融合 + 订单生成               ensemble_fusion.py → order_gen.py
```

**核心原则**：
- **所有调整先在 Playground 验证，验证通过才 promote 到 production**
- **不要跳过步骤 5 直接在 production 改配置**
- **deep_analysis 需要在 post_trade 之后跑**（需要最新的 cash/holdings）

---

## 逐步操作

### 步骤 1：导入最新市场数据

```bash
# 更新 qlib 数据到最新交易日
# 具体命令取决于数据源，通常为：
qlib data update --region cn --freq day
```

### 步骤 2：导入结算数据 + Post Trade

```bash
conda activate qlib_cupy
cd <project_root>
source workspaces/Demo_Workspace/run_env.sh

# 导入券商结算文件，更新现金和持仓
python -m quantpits.scripts.prod_post_trade
```

这一步会更新 `prod_config.json` 中的 `cash` 和 `holdings`，是后续分析的基础。

### 步骤 3：跑 Deep Analysis（Critic 模式）

```bash
python -m quantpits.scripts.run_deep_analysis --critic
```

**输出文件**（位于 `output/deep_analysis/`）：
- `action_items_{date}.json` — LLM 生成的 ActionItem 列表
- `feedback_report_{date}.json` — 闭环评估报告
- `deep_analysis_report_v5-new_{date}.md` — 完整分析报告

**关注指标**：
- `feedback_summary.accuracy` — 闭环质量（null 表示上周无建议被执行）
- `global_diagnosis.health_status` — 系统健康状态
- ActionItem 的 `confidence` 和 `risk_level`
- `scope_recommendations` — 被 scope 限制阻断了哪些建议

### 步骤 4：评估 Action Items

阅读 `action_items_{date}.json`，逐条判断。

**首要参考：model_knowledge.yaml**

```bash
# 先检查模型的历史调参经验
cat config/model_knowledge.yaml | grep -A20 "<model_name>"
```

该文件记录了每个模型的 `known_effective_params`（已知有效）和 `known_ineffective_params`（已知无效）。Action item 如果建议了 `known_ineffective_params` 中的方向 → 直接跳过，不执行。

**决策矩阵**：

| 评估维度 | 执行条件 | 不执行 |
|---------|---------|--------|
| **model_knowledge** | action 方向匹配 known_effective | 方向匹配 known_ineffective → 跳过 |
| **confidence** | ≥ 0.5 值得考虑，≥ 0.7 通常可靠 | < 0.4 仅参考 |
| **risk_level** | low 无脑执行 | high 需额外分析 |
| **diversifier** | 检查模型是否在 Orthogonal_Wildcards 组 | 低 IC 但 is_diversifier=true → 保留 |
| **action_type** | adjust_hyperparam / retrain 可在 Playground 验证 | disable_model 需 LOO delta 证据 |
| **scope** | hyperparams / model_selection 可直接执行 | combo_search 需手动跑搜索脚本 |

**特别注意**：
- `disable_model` + `confidence > 0.5` + 无 LOO delta → 违反约束，不执行
- `trigger_search` → 需要手动跑 `brute_force_ensemble.py`（search_adapter 尚未实现）
- `keep_as_diversifier` 诊断 → Per-Model 正确识别了正交分散器，无需干预

### 步骤 5：在 Playground 中验证

**永远不要在 production workspace 直接改配置。**

**单变量优先原则**：如果 action item 建议同时调整 ≥2 个参数（combo 调参），必须在 Playground 中逐个参数单独实验，分离各参数的独立效应。组合调参可能掩盖 "参数 A 有益但参数 B 有害" 的情况。

**历史教训**：krnn_Alpha360 的 dropout↑ + lr↓ combo 建议导致 IC 下降。单变量实验发现 dropout↑ 是反向的，仅 lr↓ 就 +10%。如果直接执行 combo 建议，会浪费一次训练资源且得出错误结论。

```bash
# 方式 A：手动验证（推荐，更可控）

# 5a. 同步 Playground
python -c "
from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
pm = PlaygroundManager('<project_root>/workspaces/Demo_Workspace')
pm.create_or_sync()
"

# 5b. 在 Playground 中修改对应模型的 workflow_config YAML
# 手动编辑 config/workflow_config_{model}.yaml

# 5c. 在 Playground 中重训
python -m quantpits.scripts.static_train \
    --workspace workspaces/Demo_Workspace_Playground \
    --models <model_name>

# 5d. 对比 IC/ICIR
# 训练输出中会显示 IC/ICIR，与 action_item 中的 expected_outcome 对比
```

```bash
# 方式 B：通过 feedback loop 自动执行（需要 adapter 支持）
python -m quantpits.scripts.run_feedback_loop \
    --action-items workspaces/Demo_Workspace/output/deep_analysis/action_items_{date}.json \
    --execute \
    --models <model1>,<model2> \
    --max-experiment-rounds 1

# 预览模式（--report-only），先看看会改什么
python -m quantpits.scripts.run_feedback_loop \
    --action-items workspaces/Demo_Workspace/output/deep_analysis/action_items_{date}.json \
    --report-only
```

**验证标准**：
- IC 改善 ≥ 5% 或 ICIR 改善 → 通过
- IC 基本持平但 best_epoch 改善（不再 epoch 0）→ 部分通过（架构性修复）
- IC 下降 → 不通过，回滚

### 步骤 6：Promote 成功的调整

```bash
# 方式 A：手动 promote（推荐）
# 将 Playground 中验证通过的 config 内容复制到 production
cp workspaces/Demo_Workspace_Playground/config/workflow_config_{model}.yaml \
   workspaces/Demo_Workspace/config/workflow_config_{model}.yaml

# 方式 B：通过 promote_config.py
python -m quantpits.scripts.promote_config \
    --source workspaces/Demo_Workspace_Playground \
    --target workspaces/Demo_Workspace \
    --models <model1>,<model2>
```

**Promote 规则**：
- 只有 IC/ICIR 明确改善的调整才 promote
- 效果模糊的（IC 持平但 risk=medium）留到下一轮观察
- 效果变差的（IC 下降）不 promote，在 Playground 中回滚

### 步骤 7：训练/预测

```bash
# 对 promote 过的模型做全量训练，其余 --predict-only
python -m quantpits.scripts.static_train \
    --workspace workspaces/Demo_Workspace \
    --models <promoted_model1>,<promoted_model2>

# 其余模型仅预测
python -m quantpits.scripts.static_train \
    --workspace workspaces/Demo_Workspace \
    --predict-only --all-enabled
```

### 步骤 8：融合 + 订单生成

```bash
# 融合模型预测为组合信号
python -m quantpits.scripts.ensemble_fusion

# 生成买卖订单
python -m quantpits.scripts.order_gen
```

---

## Playground 安全模型

```
Production (Demo_Workspace)          Playground (Demo_Workspace_Playground)
┌─────────────────────────┐       ┌──────────────────────────────┐
│ config/                 │  ───→ │ config/        (完整副本)    │
│   workflow_config_*.yaml│  sync │   workflow_config_*.yaml     │
│   model_registry.yaml   │       │   model_registry.yaml        │
│ data/                   │       │ data/          (混合策略)    │
│   *.jsonl               │  copy │   *.jsonl      (实体副本)    │
│   order_history/        │  link │   order_history/(符号链接)    │
│   history/              │  link │   history/     (符号链接)    │
│ output/                 │       │ output/        (独立输出)    │
│ mlruns/                 │       │ mlruns/        (独立追踪)    │
└─────────────────────────┘       └──────────────────────────────┘
```

- **config/**: 全量复制 — adapter 可以自由修改
- **data/*.jsonl**: 实体副本 — 隔离训练写入
- **data/history/**: 符号链接 — 共享只读历史数据
- **output/ + mlruns/**: 完全独立 — 不影响生产

创建/同步 Playground：
```bash
python -c "
from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
pm = PlaygroundManager('<project_root>/workspaces/Demo_Workspace')
print(pm.create_or_sync())
"
```

---

## Scope 控制

Deep Analysis 的 Critic 受 `feedback_scope.json` 控制，决定 LLM 可以干预哪些环节：

| Scope | 当前状态 | Adapter |
|-------|---------|---------|
| `hyperparams` | ✅ enabled | TrainingAdapter |
| `model_selection` | ✅ enabled | ModelSelectionAdapter |
| `combo_search` | ✅ enabled | search_adapter（尚未实现，需手动执行） |
| `strategy_params` | ❌ disabled | fusion_adapter（尚未实现） |

Scope 在 `workspaces/Demo_Workspace/config/feedback_scope.json` 中配置。

---

## 常见问题

### Q: Playground 训练报错 "lr must be float"

A: YAML 中 `5e-05` 会被部分解析器当作字符串。使用 `0.00005` 代替科学计数法。

### Q: feedback_summary.accuracy 一直是 null

A: 说明 Action Items 从未被真正执行过。至少执行 3 个调整，下一轮 deep_analysis 就有 evaluable feedback。

### Q: Synthesizer 说 "dropout 已在 XX 日期调整为 0.5" 但实际不是

A: 这是幻觉。`action_item_history.jsonl` 中所有条目的 `executed` 字段默认为 `false`。Synthesizer 现在能区分 "建议过" 和 "已执行"。如果仍然出现幻觉，检查 `config/skills/synthesizer_system.md` 的 "建议已执行 vs 仅建议的区分" 规则。

### Q: gats_Alpha158_origin_N 每次都被建议 disable

A: 该模型 avg_corr=0.03，属于 Orthogonal_Wildcards 组（is_diversifier=true）。低 IC 是预期行为（正交分散价值 > 单独预测能力）。不执行 disable 建议。Synthesizer 的 LOO delta 硬约束会将 confidence 限制在 0.5。

### Q: Triage 显示 "0 combos → Per-Combo"

A: 当所有模型都有 historical flags 时，Triage 可能认为 "成员都有问题，先修模型再分析组合"，从而跳过组合路由。这是 LLM 的一次性决策波动，不代表代码 bug。如果连续两轮出现，需要检查 Triage 的 routing 逻辑。即使 combo 被跳过，Synthesizer 通常会自行补一个 `trigger_search` 来填补组合分析缺口。

### Q: trigger_search ActionItem 怎么执行

A: search_adapter 尚未实现，需要手动运行：
```bash
python -m quantpits.scripts.brute_force_ensemble --help
# 或
python -m quantpits.scripts.minentropy_ensemble --help
```

---

## 快速参考

```bash
# 完整周度操作（最小命令集）
conda activate qlib_cupy
source workspaces/Demo_Workspace/run_env.sh

# 1-2. 数据 + 结算
python -m quantpits.scripts.prod_post_trade

# 3. 分析
python -m quantpits.scripts.run_deep_analysis --critic

# 4. 查看 action items
cat workspaces/Demo_Workspace/output/deep_analysis/action_items_$(date +%Y-%m-%d).json | python -m json.tool | less

# 5. Playground 验证（根据 action items 选择模型）
python -c "from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager; PlaygroundManager('<project_root>/workspaces/Demo_Workspace').create_or_sync()"
# ...手动改 Playground config...
python -m quantpits.scripts.static_train --workspace workspaces/Demo_Workspace_Playground --models <model>

# 6. Promote（确认 IC 改善后）
# ...手动 cp config 到 production...

# 7-8. 生产流程
python -m quantpits.scripts.static_train --models <promoted_models>
python -m quantpits.scripts.static_train --predict-only --all-enabled
python -m quantpits.scripts.ensemble_fusion
python -m quantpits.scripts.order_gen
```

---

## 相关文档

- [50_DEEP_ANALYSIS_GUIDE.md](50_DEEP_ANALYSIS_GUIDE.md) — 深度分析系统
- [51_OOMRL_FEEDBACK_OVERVIEW.md](51_OOMRL_FEEDBACK_OVERVIEW.md) — 闭环反馈概览
- [53_OOMRL_CRITIC_GUIDE.md](53_OOMRL_CRITIC_GUIDE.md) — LLM Critic 指南
- [54_OOMRL_FEEDBACK_LOOP.md](54_OOMRL_FEEDBACK_LOOP.md) — 反馈闭环执行
- [04_POST_TRADE_GUIDE.md](04_POST_TRADE_GUIDE.md) — Post Trade 指南
- [01_TRAINING_GUIDE.md](01_TRAINING_GUIDE.md) — 训练指南
- [03_ENSEMBLE_FUSION_GUIDE.md](03_ENSEMBLE_FUSION_GUIDE.md) — 融合指南
- [06_ORDER_GEN_GUIDE.md](06_ORDER_GEN_GUIDE.md) — 订单生成指南
